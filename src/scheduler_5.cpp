#ifndef SCHEDULER_5_CPP
#define SCHEDULER_5_CPP

#include "scheduler_5.h"

// TODO: have a status bit to terminate threads!

Scheduler_5::Scheduler_5(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
	this->matrix = mtx;
	this->params = params;
	this->stats = stats; 
	this->llb = llb;

	o_tiled_rows = matrix->getTiledORowSize();
	o_tiled_cols = matrix->getTiledOColSize();
	b_tiled_rows = matrix->getTiledBRowSize();

	// BW logger to have bw usage in cycle level accuracy
	// This is a cool way, I am considering during the same for PEs
	bw_logger = new float[MAX_TIME];

	// PE times
	pe_time = new uint64_t[params->getPECount()];
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	std::fill(bw_logger, bw_logger+MAX_TIME, 0.0);

	// Each row that commit is recorded in these arrays 
	vecCommittedRows_iidx = new int[b_tiled_rows];
	vecCommittedRows_cycle = new uint64_t[b_tiled_rows];
	vecCommittedRows_ASize = new uint64_t[b_tiled_rows];
	vecCommittedRows_OSize = new uint64_t[b_tiled_rows];

	// Bytes that can be transferred every cycle
	bytes_per_ns = (float)params->getBandwidth() / params->getFrequency();

	// Keep the number of committed rows need to be retired
	num_vecCommittedRows = 0;

	// Creating the wroker threads to do the computation in parallel
	int rc;
	num_worker_threads = params->getNumWorkerThreads();

	// Each thread reports the number of cycles it took to 
	//   do the computation
	mutex = new pthread_mutex_t[num_worker_threads];
	thread_info_out = new int[num_worker_threads];
	thread_info_in = new int*[num_worker_threads];
	for(int th_idx= 0; th_idx<num_worker_threads; th_idx++){
		// 0:Status=Waiting, Processing, Done. 1:i, 2:j, 3:k
		thread_info_in[th_idx] = new int[4];
		std::fill(thread_info_in[th_idx], thread_info_in[th_idx] + 4, 0);

		thread_info_out[th_idx] = 0;
		mutex[th_idx] = PTHREAD_MUTEX_INITIALIZER;
	}
	//std::fill(thread_info_out, thread_info_out + num_worker_threads, 0);
	//std::fill(mutex, mutex + num_worker_threads, PTHREAD_MUTEX_INITIALIZER);

	threads = new pthread_t[num_worker_threads];
	for(int i=0; i< num_worker_threads; i++){
		if ((rc = pthread_create(&threads[i], NULL, &(Scheduler_5::multiplierHelperThread), this)))
				printf("pthread creation failed: %d\n", rc);
	}

	// TODO 1: Add setstacksize to pthread
	// TODO 2: Change this! make it dynamic
	omp_set_num_threads(4);

	return;
}

Scheduler_5::~Scheduler_5(){
	delete [] bw_logger;
	delete [] pe_time;
	delete [] vecCommittedRows_iidx;
	delete [] vecCommittedRows_cycle;
	delete [] vecCommittedRows_ASize;
	delete [] vecCommittedRows_OSize;
	delete [] threads;
	delete [] mutex;
	delete [] thread_info_out;
	// Two level
	//delete [] thread_info_in;

	return;
}

// Reset all the internal stats
void Scheduler_5::Reset(){
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	std::fill(bw_logger, bw_logger+MAX_TIME, 0.0);
	num_vecCommittedRows = 0;
	bytes_per_ns = (float)params->getBandwidth() / params->getFrequency();

	return;
}

/* Dataflow in pseudo code is :
 *
 * 4 for(int k_idx = 0; k_idx<ceil[K/a_reuse]; k_idx++)
 *     o_reuse = calculateOReuse(a_reuse);
 * 3   for(int j_idx = 0; j_idx<ceil[J/o_reuse]; j_idx++)
 * 2     for(int i_idx = 0; i_idx<I; i_idx++) // row granularity sync; similar to ExTensor
 * 1       for(int k_inner=0; k_inner<min(a_reuse,K-k_idx*a_reuse); k_inner++)
 * 0         for(int j_inner=0; j_inner<min(o_reuse, J-j_idex*o_reuse; j_inner++))
 *
 * For row based synchronization we need two vectors 
 * 1) A and O rows in LLB (every time we will add some to fill up the LLB portion) 
 * 2) A and O row ID and finish time, so if it is evicted then the rest of the PEs
 *			added later should start not earlier than that time.  
 *
 */
void Scheduler_5::Run(){


	int a_reuse = params->getAReuse();

	b_llb_horizontalSum = new int[b_tiled_rows]; 
	vecCyclesComp = new int[a_reuse * b_tiled_rows];
	OSize_preComp = new uint64_t[a_reuse];

	// Iterations over B cols / O cols / a_reuse
	for(int k_idx = 0; k_idx < o_tiled_cols; k_idx+=a_reuse){
		int j_idx_start = 0, j_idx_stop  = 0;
		// Iteration over B rows / A cols / o_reuse
		while(j_idx_start < b_tiled_rows){

			// Calculate O_reuse and fetch B tiles
			CalcOReuse(k_idx, j_idx_start, j_idx_stop); 
			params->setOReuse(j_idx_stop-j_idx_start);
			// Calculate the horizontal sum of B PE tiles in the LLB tile
			//   This sum is used to decide whether to bring PE tiles of A
			//   It basically says whether there is any nnz PE tile in each row
			CalcBLLBHorizontalSum(j_idx_start, j_idx_stop, 
					k_idx, std::min(k_idx+a_reuse, o_tiled_cols), b_llb_horizontalSum);
			// You need to load a new LLB tile of B
			load_b_tiles = 1;
			// It shows what is the first row in A and O which has not been fetched yet
			first_row_notFetched = 0;
			// iteration over A rows / O rows
			//   there is no i_idx++ since it goes as far as LLB assists
			for(int i_idx = 0; i_idx < o_tiled_rows;/* */){
				int numRows = 0;
				// fill up LLB with A and O rows!
				// It fetches A and O rows as much as LLB allows. It first evicts the row with
				//   the smallest end cycle time and gives its space to new rows. The number of new rows
				//   is returned in numRows (range [0,o_tiled_rows] ).
				FetchAORows(numRows, i_idx, j_idx_start, j_idx_stop, k_idx, b_llb_horizontalSum);
				// Processing each row one by one; since row sync gets rows in different starting cycles		
				for(int i_row = 0; i_row< numRows; i_row++){
					// Run and schedule the multiplication of one row of A by an LLB tile of B
					Schedule(i_idx, j_idx_start, j_idx_stop, k_idx, b_llb_horizontalSum);
					i_idx++;
					load_b_tiles = 0;
					// If we have exceeded the LLB capacity already stop it! empty some A and O rows
					//	and proceed when we have enough space
					if(llb->GetSize() >= llb->GetCapacity()){
						break;
					}
				}
			}

			j_idx_start = j_idx_stop;
			// Flush everything inside LLB since we are going to bring new data
			llb->EvictMatrixFromLLB('B', DONT_UPDATE_TRAFFIC);
			llb->EvictMatrixFromLLB('A', DONT_UPDATE_TRAFFIC);
			llb->EvictMatrixFromLLB('O', UPDATE_TRAFFIC);
			num_vecCommittedRows = 0;
			// All PEs should finish their tasks before fetching new B tiles
			SyncPETimes();
		}
	}

	delete [] b_llb_horizontalSum;
	delete [] threads;
	delete [] vecCyclesComp;
	delete [] OSize_preComp;
	return;
}

// Gets a fiber of A and O and a rectangle LLB tile of B and schedules them
//   The ending cycles of each row is recorded in vecCommittedRows for 
//   synnchronization, fetching, and committing rows 
void Scheduler_5::Schedule(int i_idx, int j_idx_start, int j_idx_stop, int k_idx, 
		int *b_llb_horizontalSum){
	
	int a_reuse = params->getAReuse();
	int o_reuse = params->getOReuse();

	uint64_t action_bytes = 0, batch_ending_cycle= 0, excessCycles = 0;
	int cycles_comp = 0;

	uint64_t OSize_postComp =0, OSize_fetched = 0;	
	batch_ending_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
	uint64_t acc_b_size = AccumulateSize('B', j_idx_start, j_idx_stop, k_idx, std::min(k_idx+ a_reuse, o_tiled_cols ));
	if (acc_b_size == 0){
		vecCommittedRows_iidx[num_vecCommittedRows] = i_idx;
		vecCommittedRows_cycle[num_vecCommittedRows] = batch_ending_cycle;
		num_vecCommittedRows++;
		return;
	}

	std::fill(vecCyclesComp, vecCyclesComp+ a_reuse*o_reuse, 0);
	for(int k_inner=k_idx; k_inner<std::min(k_idx+a_reuse,o_tiled_cols); k_inner++){
		if(ShouldIFetchThisOTile(i_idx, j_idx_start, j_idx_stop, k_inner))
			OSize_preComp[k_inner-k_idx] = matrix->getCOOSize('O', i_idx, k_inner);
	}

	//printf("%d\n",i_idx);
	omp_set_nested(1);
	omp_set_dynamic(0);
	omp_set_num_threads(40);
	//mkl_set_num_threads_local(5);

	//mkl_set_num_threads_local(8);
	// We cannot parasllelize over J since they are writing to the same output tile
	for(int j_inner = j_idx_start; j_inner < j_idx_start+o_reuse; j_inner++){
		// Parallelizing over K is pretty safe since each thread works on a different output tile
		for(int k_inner = k_idx; k_inner < std::min(k_idx+a_reuse, o_tiled_cols); k_inner++){
			//DoTheTileMultiplication(i_idx, j_idx_start, j_inner, k_idx, k_inner);
			checkForFinishedThreads(j_idx_start, k_idx);
			if((matrix->getCSFSize('B', j_inner, k_inner) != 0) &
				(matrix->getCSFSize('A', i_idx, j_inner) != 0)){
			submitTileMultiplicatin(i_idx, j_inner, k_inner);
			}
			else{
				int entryAddr = a_reuse*(j_inner-j_idx_start) + (k_inner - k_idx);
				vecCyclesComp[entryAddr] = 0;
			}
		}
		// make sure you have all the data to avoid race condition 
		//   for writing to the same output tile
		checkUntilAllThreadsFinished(j_idx_start, k_idx);
	}
	
	// Iterate over each B fiber
	for(int k_inner = k_idx; k_inner < std::min(k_idx+a_reuse, o_tiled_cols); k_inner++){
		// Iterate over each (A row, corresponding B column) tiles
		for(int j_inner = j_idx_start; j_inner < j_idx_start+o_reuse; j_inner++){
			action_bytes = 0;
			// Add up the B tile needs to be fetched from memory
			if(load_b_tiles){
				// If need to load B tiles add them to the memory traffic usage
				action_bytes += matrix->getCSFSize('B', j_inner, k_inner);
			}	
			if((k_inner == k_idx) & (b_llb_horizontalSum[j_inner]>0)){
				// Add A tiled size to the traffic computation
				action_bytes += matrix->getCSFSize('A', i_idx, j_inner);
			}
			int entryAddr = a_reuse*(j_inner-j_idx_start) + (k_inner - k_idx);
			cycles_comp = vecCyclesComp[entryAddr];
			// Update bandwidth and update PE units when either 
			//   we have loaded sth or did some computation
			if((action_bytes) | (cycles_comp)){
				scheduleAndUpdateBW(cycles_comp, action_bytes, batch_ending_cycle);
			}
		}
		// LLB size needs to be changed because the output size might have 
		//   changed during the computation
		uint64_t OSize_preComp_curr = OSize_preComp[k_inner-k_idx];
		OSize_postComp = matrix->getCOOSize('O', i_idx, k_inner);
		llb->AddToLLB('O', Req::write, OSize_postComp-OSize_preComp_curr ,
			 	DONT_UPDATE_TRAFFIC);
		vecCommittedRows_OSize[i_idx] += (OSize_postComp-OSize_preComp_curr);
		if(vecCommittedRows_OSize[i_idx]>0){
			// Update the bandwidth usage for the output psum read
			excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle, OSize_preComp_curr);
		}
	}

	// Use the bandwidth for writing back the output to memory
	excessCycles = 0;	
	if(vecCommittedRows_OSize[i_idx]>0){
		action_bytes = vecCommittedRows_OSize[i_idx];
		// Update the bandwidth usage for the output write-back
		excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle, action_bytes);
	}

	// Add the end_cycle, #row pair to the committed rows vector
	vecCommittedRows_iidx[num_vecCommittedRows] = i_idx;
	vecCommittedRows_cycle[num_vecCommittedRows] = batch_ending_cycle;
	num_vecCommittedRows++;

	uint64_t cycles_stats = std::max(*std::max_element(pe_time, pe_time+params->getPECount()),
			batch_ending_cycle + excessCycles);
	stats->Set_cycles(cycles_stats);
	stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());
	return;
}

void Scheduler_5::checkForFinishedThreads(int j_start, int k_start){
	int a_reuse = params->getAReuse();
	for(int t_idx = 0; t_idx < num_worker_threads; t_idx++){
		//if not done!
		if(thread_info_in[t_idx][0] != 2)
			continue;
		// mutex section
		pthread_mutex_lock( &mutex[t_idx]);
		int comp_cycles = thread_info_out[t_idx];
		thread_info_in[t_idx][0] = 0;
		int k_inner = thread_info_in[t_idx][3];
		int j_inner = thread_info_in[t_idx][2];
		pthread_mutex_unlock(&mutex[t_idx]);
		int entryAddr = a_reuse*(j_inner-j_start) + (k_inner - k_start);
		vecCyclesComp[entryAddr] = comp_cycles;
	}
	return;
}

void Scheduler_5::checkUntilAllThreadsFinished(int j_start, int k_start){
	int a_reuse = params->getAReuse();
	for(int t_idx = 0; t_idx < num_worker_threads; t_idx++){
		//Wait until the thread is not processing anymore
		while(thread_info_in[t_idx][0] == 1);
		// mutex section
		if(thread_info_in[t_idx][0] == 2){
			pthread_mutex_lock( &mutex[t_idx]);
			int comp_cycles = thread_info_out[t_idx];
			thread_info_in[t_idx][0] = 0;
			int k_inner = thread_info_in[t_idx][3];
			int j_inner = thread_info_in[t_idx][2];
			pthread_mutex_unlock(&mutex[t_idx]);
			int entryAddr = a_reuse*(j_inner-j_start) + (k_inner - k_start);
			vecCyclesComp[entryAddr] = comp_cycles;
		}
	}
	return;
}

void Scheduler_5::submitTileMultiplicatin(int i_idx, int j_inner, int k_inner){
	// Wait until you submit the job to a miserable thread
	while(1){
		for(int t_idx = 0; t_idx < num_worker_threads; t_idx++){
			if(thread_info_in[t_idx][0] == 0){
				pthread_mutex_lock( &mutex[t_idx]);
				thread_info_in[t_idx][1] == i_idx;
				thread_info_in[t_idx][2] == j_inner;
				thread_info_in[t_idx][3] == k_inner;
				thread_info_in[t_idx][0] == 1;
				pthread_mutex_unlock(&mutex[t_idx]);
				return;
			}
		}
	}
	return;
}

// This function gets the cycles and bytes that a tile multiplication takes
//  Then updates the bw logger, and adds the extra cycles to the computation
//  and sschedules it 
void Scheduler_5::scheduleAndUpdateBW(int cycles_comp, uint64_t action_bytes, 
		uint64_t &batch_ending_cycle){

	uint64_t starting_cycle = 0, ending_cycle = 0, excessCycles = 0;	
	
	if(params->getIntersectModelDist() == intersect_dist::sequential){
		// Start and end cycle for currrent tile multiplication
		starting_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
		ending_cycle = cycles_comp + starting_cycle;
		// Update the bandwidth usage. This is something I am very excited about, at the end 
		// I can plot	this and look at per cycles bandwidth usage! I should do this for PEs as well
		excessCycles = updateBWLogAndReturnExcess(starting_cycle, ending_cycle, action_bytes);

		*std::min_element(pe_time, pe_time+params->getPECount()) +=	
			((uint64_t)cycles_comp + excessCycles);
		ending_cycle += excessCycles;
		if(ending_cycle>batch_ending_cycle)
			batch_ending_cycle = ending_cycle;
	}
	else if(params->getIntersectModelDist() == intersect_dist::parallel){
		// find the maximum number of cycles a PE needs to do computation
		uint64_t cycles_per_pe = (uint64_t)ceil((double)cycles_comp/params->getPECount());
		// Start and end cycle for currrent tile multiplication
		starting_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
		ending_cycle = cycles_per_pe + starting_cycle;
		// Update the bandwidth usage. This is something I am very excited about, at the end 
		// I can plot	this and look at per cycles bandwidth usage! I should do this for PEs as well
		excessCycles = updateBWLogAndReturnExcess(starting_cycle, ending_cycle, action_bytes);
		ending_cycle += excessCycles;
		uint64_t ending_cycle_w_excess = ending_cycle;

		for(uint64_t i = (uint64_t)cycles_comp; i>0;/* */){
			uint64_t cycles_curr_pe = (i > cycles_per_pe)? cycles_per_pe	:	i;
			i-=cycles_curr_pe;
			uint64_t min_pes = *std::min_element(pe_time, pe_time+params->getPECount());
			uint64_t max_cycles = std::max(min_pes + cycles_curr_pe, ending_cycle_w_excess);
			*std::min_element(pe_time, pe_time+params->getPECount()) = max_cycles;
			if (max_cycles > ending_cycle)
				ending_cycle = max_cycles;
		}
		if(ending_cycle>batch_ending_cycle)
			batch_ending_cycle = ending_cycle;
	}
	else{
		printf("You need to use either sequential or parallel distribution model!\n");
		exit(1);
	}

	return;
}

void * Scheduler_5::multiplierHelperThread(void * ptr) {
	
	Scheduler_5 * sched = (Scheduler_5 *) ptr;
	sched->multiplierThread();
	// this is where you do all the work
}

void * Scheduler_5::multiplierThread(){
	int myID = gettid()% num_worker_threads;

	printf("Thread %d is here!\n",myID);
	int ready = 0, i_idx , j_idx , k_idx;
	while(1){
		pthread_mutex_lock( &mutex[myID]);
		if(thread_info_in[myID][0] == 1){
			ready = 1; 
			i_idx = thread_info_in[myID][1]; 
			j_idx = thread_info_in[myID][2]; 
			k_idx = thread_info_in[myID][3];
		}
		pthread_mutex_unlock(&mutex[myID]);
		if (!ready){
			continue;
		}
		int a_reuse = params->getAReuse();
		uint64_t bytes_rd = 0, bytes_wr = 0;
		int cycles_comp;
		// Do the actual calculation
		if((matrix->getCSFSize('B', j_idx, k_idx) != 0) &
				(matrix->getCSFSize('A', i_idx, j_idx) != 0)){
			matrix->CSRTimesCSR(i_idx, j_idx, k_idx, 
					&cycles_comp, &bytes_rd, &bytes_wr);
			stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);
		}
		pthread_mutex_lock( &mutex[myID]);
		// I am done! Mark as finished processing
		thread_info_in[myID][0] = 2;
		thread_info_out[myID] = cycles_comp;
		pthread_mutex_unlock(&mutex[myID]);
	}
}

// Gets the A, B, and O tile addresses, does the MKL computation, updates the 
//   cyclesComp array, and updates the busy cycles stats
void Scheduler_5::DoTheTileMultiplication(int i_idx, int j_start, int j_inner,
	 	int k_start, int k_inner){
	
	int a_reuse = params->getAReuse();
	int entryAddr = a_reuse*(j_inner-j_start) + (k_inner - k_start);
	uint64_t bytes_rd = 0, bytes_wr = 0;
	int cycles_comp;
	// Do the actual calculation
	if((matrix->getCSFSize('B', j_inner, k_inner) != 0) &
			(matrix->getCSFSize('A', i_idx, j_inner) != 0)){
		matrix->CSRTimesCSR(i_idx, j_inner, k_inner, 
				&vecCyclesComp[entryAddr], &bytes_rd, &bytes_wr);
		cycles_comp = vecCyclesComp[entryAddr];
		stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);
	}
	
	return;
}


// This is a beautiful bw logger that gets the start cycle and end cycle 
//   of each tile multiplication and in a cycle accurate way says how
//   many extra cycles it is going to take.
// Its role is to keep track of bandwidth
uint64_t Scheduler_5::updateBWLogAndReturnExcess(uint64_t starting_cycle, 
		uint64_t ending_cycle, uint64_t action_bytes){

	total_traffic += action_bytes;
	//printf("%f - %f, %d\n", action_bytes_f, bytes_per_ns, action_bytes);
	//exit(1);

	float action_bytes_f = float(action_bytes);
	for(int i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		float rem_cap = bytes_per_ns - bw_logger[i_idx];
		if((action_bytes_f > 0) & (rem_cap == 0))
			continue;
		if(action_bytes_f > rem_cap){
			bw_logger[i_idx] = bytes_per_ns;
			action_bytes_f -= rem_cap;
		}
		else{
			bw_logger[i_idx] += action_bytes_f;
			action_bytes_f = 0;
			if((uint64_t)i_idx > ending_cycle)
				return ((uint64_t)i_idx-ending_cycle);
			else 
				return 0;
			//return std::max(0, i_idx - ending_cycle);
		}
	}
	printf("%d bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}

uint64_t Scheduler_5::updateBWLogAndReturnExcess(uint64_t starting_cycle, 
		uint64_t ending_cycle, uint64_t action_bytes,
		uint64_t a_bytes, uint64_t b_bytes, uint64_t o_r_bytes, uint64_t o_w_bytes){
	
	uint64_t sum_all = a_bytes+b_bytes+o_r_bytes+o_w_bytes; 
	if (sum_all != action_bytes){
		printf("a: %lu, b: %lu, o_r: %lu, o_w: %lu, sum: %lu, action_bytes: %lu\n",
				a_bytes, b_bytes, o_r_bytes, o_w_bytes, sum_all, action_bytes );
		exit(1);
	}
	a_bwl_traffic += a_bytes;
	b_bwl_traffic += b_bytes;
	o_bwl_traffic_read += o_r_bytes;
	o_bwl_traffic_write += o_w_bytes;

	total_traffic += action_bytes;
	//float action_bytes_f = action_bytes;

	//printf("%f - %f, %d\n", action_bytes_f, bytes_per_ns, action_bytes);
	//exit(1);

	float action_bytes_f = float(action_bytes);
	for(int i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		float rem_cap = bytes_per_ns - bw_logger[i_idx];
		if((action_bytes_f > 0) & (rem_cap == 0))
			continue;
		if(action_bytes_f > rem_cap){
			bw_logger[i_idx] = bytes_per_ns;
			action_bytes_f -= rem_cap;
		}
		else{
			bw_logger[i_idx] += action_bytes_f;
			action_bytes_f = 0;
			return std::max((uint64_t)0, i_idx - ending_cycle);
		}
	}
	printf("%d bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}


// This synchronization happens when a row related calculation is over
//		So, when we are bringing a new row, we make sure that the timing has been updated
//		min_val is the end_cycle of the committed row, which will be the start_cycle of
//		the fetched row
void Scheduler_5::SyncPETimesWithMinCycle(uint64_t min_val){
	for(int idx=0; idx<params->getPECount(); idx++){
		pe_time[idx] = std::max(pe_time[idx], min_val);
	} 
	return; 
}

// All of the PEs should finish their work before fetching the next	B tiles
void Scheduler_5::SyncPETimes(){

	int max_val = *std::max_element(pe_time, pe_time+params->getPECount());
	std::fill(pe_time, pe_time+params->getPECount(), max_val);

	return;
}

// Gets the Column boundary and start address of matrix B row,
//   returns the start and stop address or the matrix B row, i.e., o_reuse
// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
void Scheduler_5::CalcOReuse(int k_idx, int j_idx_start, int & j_idx_stop){
	int a_reuse = params->getAReuse();
	j_idx_stop = j_idx_start;
	// Add rows until it either runs out of memory or reaches the last row
	for(int idx = j_idx_start; idx<b_tiled_rows; idx++){
		// Find the size of the new row 
		uint64_t extra_size = AccumulateSize('B', idx, idx+1, k_idx, std::min(k_idx+a_reuse, o_tiled_cols));
		// It means that it could not fit the new row in LLB and failed
		if(llb->DoesFitInLLB('B', extra_size) == 0) {return;}
		llb->AddToLLB('B', Req::read, extra_size, UPDATE_TRAFFIC);
		j_idx_stop++;
	}
	return;
}

// It fetches A and O rows as much as LLB allows. It first evicts the row with
//   the smallest end cycle time and gives its space to new rows. The number of new rows
//   is returned in numRows (range [0,o_tiled_rows] ).
void Scheduler_5::FetchAORows(int & numRows,
		int i_idx, int j_idx_start, int j_idx_stop, int k_idx, int *b_llb_horizontalSum){

	int a_reuse = params->getAReuse();
	uint64_t a_row_size, o_row_size;
	// This should normally free only one row; The reason I have used while is that in many cases
	//   the row size is zero (empty row), which we should skip!
	while(num_vecCommittedRows>0){
		uint64_t * smallest_end_cycle = std::min_element(vecCommittedRows_cycle, 
				vecCommittedRows_cycle+num_vecCommittedRows);
		int distance_from_start = std::distance(vecCommittedRows_cycle, smallest_end_cycle);
		SyncPETimesWithMinCycle(vecCommittedRows_cycle[distance_from_start]);
		// Find the A and O row size of the corresponding row and columns
		a_row_size = vecCommittedRows_ASize[vecCommittedRows_iidx[distance_from_start]];
		o_row_size = vecCommittedRows_OSize[vecCommittedRows_iidx[distance_from_start]];

		// Remove the rows from LLB memory; A is flushed and O is written back
		uint64_t sizebefore = llb->GetSize();
		llb->RemoveFromLLB('A', Req::read, a_row_size, DONT_UPDATE_TRAFFIC);
		llb->RemoveFromLLB('O', Req::write, o_row_size, UPDATE_TRAFFIC);

		//printf("removed row %d - size before %lu, now %lu\n", 
		//		vecCommittedRows_iidx[distance_from_start], sizebefore, llb->GetSize());
		
		// remove the min from the list
		Swap(vecCommittedRows_iidx[distance_from_start], vecCommittedRows_iidx[num_vecCommittedRows-1]);
		Swap(vecCommittedRows_cycle[distance_from_start], vecCommittedRows_cycle[num_vecCommittedRows-1]);

		num_vecCommittedRows--;

		// If the size was zero, then kudos! Get out of while loop
		if((a_row_size + o_row_size) > 0)
			break;
	}
	
	numRows = 0;
	for (int idx = first_row_notFetched; idx < o_tiled_rows; idx++){
		// I modified the accumulate functions to take into account smart PE tile fetch
		//   They will only be brought into LLB if they will be used in any computation
		a_row_size = AccumulateSize_AwrtB(idx, j_idx_start, j_idx_stop, b_llb_horizontalSum);
		o_row_size = AccumulateSize_OwrtAB(idx, j_idx_start, j_idx_stop, k_idx, 
				std::min(k_idx+a_reuse, o_tiled_cols));

		if ((llb->DoesFitInLLB('A', a_row_size)) & (llb->DoesFitInLLB('O', o_row_size))){
			llb->AddToLLB('A', Req::read, a_row_size, UPDATE_TRAFFIC);
			llb->AddToLLB('O', Req::read, o_row_size, UPDATE_TRAFFIC);
			
			vecCommittedRows_ASize[idx] = a_row_size;
			vecCommittedRows_OSize[idx] = o_row_size;

			first_row_notFetched = idx+1; 
		}
		// We can not fit any more rows, return 
		else{break;}
	}
	numRows = first_row_notFetched - i_idx;
	//printf("numRows :%d ,%d, %lu\n",numRows, num_vecCommittedRows, llb->GetSize());

	return;
}

// Find horizonatal sum of B tiles in LLB; this is used in
//   deciding whether to bring A PE tiles to LLB or not
// If horizontalsum of the row is zero then do not bring the 
//   corresponding PE tile of A
void Scheduler_5::CalcBLLBHorizontalSum(int j_start, int j_end, 
		int k_start, int k_end, int * b_llb_horizontalSum){
	
	for(int j_idx = j_start; j_idx<j_end; j_idx++){
		if (AccumulateSize('B', j_idx, j_idx+1, k_start, k_end) > 0)
			b_llb_horizontalSum[j_idx] = 1;
		else
			b_llb_horizontalSum[j_idx] = 0;
	}

	return;
}


// Finds the size of A PE tiles wrt B tiles
//  looks at the B tiles in the LLB to see whether 
//  they should be loaded or not 
uint64_t Scheduler_5::AccumulateSize_AwrtB(int i_idx, 
		int j_start, int j_end, int *b_llb_horizontalSum){

	uint64_t size = 0;
	for(int j_idx=j_start; j_idx<j_end; j_idx++){
			if(b_llb_horizontalSum[j_idx]){
				size += matrix->getCSFSize('A', i_idx, j_idx);}}
	return size;
}

// Finds the size of O PE tiles wrt A & B tiles
//  looks at the A&B intersection in the LLB to see whether 
//  they should be loaded or not 
uint64_t Scheduler_5::AccumulateSize_OwrtAB(int i_idx,	
		int j_start, int j_end, int k_start, int k_end){

	uint64_t size = 0;
	for(int k_idx = k_start; k_idx<k_end; k_idx++){
		if(ShouldIFetchThisOTile(i_idx, j_start, j_end, k_idx))
			size+=matrix->getCOOSize('O', i_idx, k_idx);
	}
	return size;
}

// Gets the i, k and j range for matrix A and B 
//   then says whther the specific O tile should be fetched or
//   not based on the intersection
uint32_t Scheduler_5::ShouldIFetchThisOTile(int i_idx, 
		int j_idx_start, int j_idx_end, int k_idx){
		
	for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++){
		if((matrix->getCSRSize('A', i_idx, j_idx)>0) &
				(matrix->getCSRSize('B', j_idx, k_idx)>0)){
			return 1;
		}
	}
	return 0;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the size that block of tiles would occupy in LLB
uint64_t Scheduler_5::AccumulateSize(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t size = 0;
	switch (mat_name){
		case 'A':{ 
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++){
				for(int j_idx=d2_start; j_idx<d2_end; j_idx++){
					size += matrix->getCSFSize('A', i_idx, j_idx);}}
			break;}
		case 'B':{ 
			for(int j_idx=d1_start; j_idx<d1_end; j_idx++){
				for(int k_idx=d2_start; k_idx<d2_end; k_idx++){
					size += matrix->getCSFSize('B', j_idx, k_idx);}}
			break;}
		case 'O':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++){
				for(int k_idx=d2_start; k_idx<d2_end; k_idx++){
					size += matrix->getCOOSize('O', i_idx, k_idx);}}
			break;}
		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}
	return size;
}

template<typename T>
void Scheduler_5::Swap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

void Scheduler_5::printPEs(){
	for(int i=0; i<params->getPECount(); i++)
		printf("%lu ", pe_time[i]);
	printf("\n");
	return;
}

void Scheduler_5::PrintBWUsage(){
	//uint64_t size = 0;
	double size = 0;
	for (uint64_t i=0; i< stats->Get_cycles(); i++)
		size += bw_logger[i];
	printf("BW logger shows : %lu bytes, %f GBs\n", (uint64_t)size, (double)size/(1024.0*1024.0*1024.0));
	/*
	printf("BW logger a: %lu, b: %lu, o_r: %lu, o_w: %lu\n", a_bwl_traffic, b_bwl_traffic, o_bwl_traffic_read, o_bwl_traffic_write);

	printf("total_traffic %lu, a_read %lu, b read %lu, o_read %lu, o_write %lu\n", 
			total_traffic, a_traffic, b_traffic, o_traffic_read, o_traffic_write);
	*/
	return;
}

void Scheduler_5::PrintBWLog(){
	FILE * pFile;
	pFile = fopen ("bw_log.txt","w");
	for (uint64_t i=0; i< stats->Get_cycles(); i++)
		fprintf(pFile, "%f\n", bw_logger[i]);
	fclose(pFile);
	return;
}

pid_t Scheduler_5::gettid( void )
{
	return syscall( __NR_gettid );
}

#endif
