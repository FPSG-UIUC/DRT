#ifndef SCHEDULER_4_CPP
#define SCHEDULER_4_CPP

#include "scheduler_4.h"

Scheduler_4::Scheduler_4(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
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

	sum_j = new uint64_t[b_tiled_rows];
	sum_i = new uint64_t[o_tiled_rows];
	// Bytes that can be transferred every cycle
	bytes_per_ns = (float)params->getBandwidth() / params->getFrequency();

	// Keep the number of committed rows need to be retired
	num_vecCommittedRows = 0;

	/*
	a_traffic = 0; b_traffic = 0; o_traffic_read = 0; o_traffic_write = 0; total_traffic = 0;
	a_bwl_traffic =0; b_bwl_traffic = 0; o_bwl_traffic_read = 0; o_bwl_traffic_write = 0;
	*/
	return;
}

Scheduler_4::~Scheduler_4(){
	delete [] bw_logger;
	delete [] pe_time;
	delete [] vecCommittedRows_iidx;
	delete [] vecCommittedRows_cycle;
	delete [] vecCommittedRows_ASize;
	delete [] vecCommittedRows_OSize;
	delete [] sum_j;
	delete [] sum_i;
	return;
}

// Reset all the internal stats
void Scheduler_4::Reset(){
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
void Scheduler_4::Run(){

	// loop over B rows (J) / A colds (J)
	//printf("i %d, j %d\n",o_tiled_rows, b_tiled_rows);

	int a_reuse = params->getAReuse();
	a_reuse_const = a_reuse;

	int * b_llb_horizontalSum = new int[b_tiled_rows]; 

	// Iterations over B cols / O cols / a_reuse
	for(int k_idx = 0; k_idx < o_tiled_cols; /*k_idx+=a_reuse*/){
		int j_idx_start = 0, j_idx_stop  = 0;
		// Iteration over B rows / A cols / o_reuse
		while(j_idx_start < b_tiled_rows){

			// Calculate O_reuse and fetch B tiles
			CalcOReuse(k_idx, j_idx_start, j_idx_stop);
			a_reuse = params->getAReuse();
			// Calculate the horizontal sum of B PE tiles in the LLB tile
			//   This sum is used to decide whether to bring PE tiles of A
			//   It basically says whether there is any nnz PE tile in each row
			CalcBLLBHorizontalSum(j_idx_start, j_idx_stop, 
					k_idx, std::min(k_idx+a_reuse, o_tiled_cols), b_llb_horizontalSum);
			
			PreCalculateAORowsSize(j_idx_start, j_idx_stop, 
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
		k_idx += a_reuse;
		params->setAReuse(a_reuse_const);
	}
	delete [] b_llb_horizontalSum;

	return;
}

// Gets a fiber of A and O and a rectangle LLB tile of B and schedules them
//   The ending cycles of each row is recorded in vecCommittedRows for 
//   synnchronization, fetching, and committing rows 
void Scheduler_4::Schedule(int i_idx, int j_idx_start, int j_idx_stop, int k_idx, 
		int *b_llb_horizontalSum){
	
	int a_reuse = params->getAReuse();
	uint64_t starting_cycle = 0, ending_cycle = 0, excessCycles = 0, 
					 bytes_rd = 0, bytes_wr = 0, action_bytes = 0, batch_ending_cycle= 0;
	int cycles_comp = 0;

	//printf("%d\n", i_idx);	
	uint64_t OSize_preComp = 0, OSize_postComp =0, OSize_fetched = 0;	
	batch_ending_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
	//int acc_b_size = std::accumulate(&b_llb_horizontalSum[j_idx_start], &b_llb_horizontalSum[j_idx_stop], 0.0);
	//if (acc_b_size == 0){
	if (llb->GetBSize() == 0){
		vecCommittedRows_iidx[num_vecCommittedRows] = i_idx;
		vecCommittedRows_cycle[num_vecCommittedRows] = batch_ending_cycle;
		num_vecCommittedRows++;
		return;
	}
	// Iterate over each B fiber
	for(int k_inner = k_idx;	k_inner < std::min(k_idx+a_reuse, o_tiled_cols);	k_inner++){
		// Remove the pre-computation size and add the post calculation size
		OSize_preComp = 0;
		// A small trick to boost the performance
		int oTileSize = matrix->getCOOSize('O', i_idx, k_inner);
		if(oTileSize == 0)
			OSize_preComp = 0;
		else if(ShouldIFetchThisOTile(i_idx, j_idx_start, j_idx_stop, k_inner))
			OSize_preComp = matrix->getCOOSize('O', i_idx, k_inner);

		// Iterate over each (A row, corresponding B column) tiles
		for(int j_inner = j_idx_start; j_inner < j_idx_stop; j_inner++){
			action_bytes = 0;
			/*a_update=0; b_update=0; o_r_update=0; o_w_update=0;*/
			bytes_rd = 0; bytes_wr= 0; cycles_comp = 0;

			// Add up the B tile needs to be fetched from memory
			if(load_b_tiles){
				// If need to load B tiles add them to the memory traffic usage
				action_bytes += matrix->getCSFSize('B', j_inner, k_inner);
				/*
				b_traffic += matrix->getCSFSize('B', j_inner, k_inner);
				b_update =  matrix->getCSFSize('B', j_inner, k_inner);
				*/
			}	
			if((k_inner == k_idx) & (b_llb_horizontalSum[j_inner]>0)){
				// Add A tiled size to the traffic computation
				action_bytes += matrix->getCSFSize('A', i_idx, j_inner);
				/*
				a_traffic += matrix->getCSFSize('A', i_idx, j_inner);
				a_update = matrix->getCSFSize('A', i_idx, j_inner);
				*/
			}

			// Do the actual calculation
			if((matrix->getCSFSize('B', j_inner, k_inner) != 0) &
					(matrix->getCSFSize('A', i_idx, j_inner) != 0)){
				matrix->CSRTimesCSR(i_idx, j_inner, k_inner, &cycles_comp, &bytes_rd, &bytes_wr);
				stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);
			}
			
			// Update bandwidth and update PE units when either 
			//   we have loaded sth or did some computation
			
			if ((action_bytes) | (cycles_comp)){
				if(params->getIntersectModelDist() == intersect_dist::sequential){
					// Start and end cycle for currrent tile multiplication
					starting_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
					ending_cycle = cycles_comp + starting_cycle;
					// Update the bandwidth usage. This is something I am very excited about, at the end 
					// I can plot	this and look at per cycles bandwidth usage! I should do this for PEs as well
					excessCycles = updateBWLogAndReturnExcess(starting_cycle, ending_cycle, action_bytes);
					/*excessCycles = updateBWLogAndReturnExcess(starting_cycle, ending_cycle, action_bytes,
						 	a_update, b_update, o_r_update, o_w_update);*/

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
							//(cycles_curr_pe + excessCycles);
					}
					if(ending_cycle>batch_ending_cycle)
						batch_ending_cycle = ending_cycle;
			
				}
				else{
					printf("You need to use either sequential or parallel distribution model!\n");
					exit(1);
				}
			}
		}
		// LLB size needs to be changed because the output size might have 
		//   changed during the computation
		OSize_postComp = matrix->getCOOSize('O', i_idx, k_inner);
		llb->AddToLLB('O', Req::write, OSize_postComp-OSize_preComp, DONT_UPDATE_TRAFFIC);
		vecCommittedRows_OSize[i_idx] += (OSize_postComp-OSize_preComp);
		if(vecCommittedRows_OSize[i_idx]>0){
			// Update the bandwidth usage for the output psum read
			/*a_update=0; b_update=0; o_r_update=OSize_preComp; o_w_update=0;*/

			excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle, OSize_preComp);
			//excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle, 
			//   OSize_preComp, a_update, b_update, o_r_update, o_w_update);

			//o_traffic_read += OSize_preComp;
		}
	}

	// Use the bandwidth for writing back the output to memory
	excessCycles = 0;	
	if(vecCommittedRows_OSize[i_idx]>0){
		/*a_update=0; b_update=0; o_r_update=0; o_w_update= vecCommittedRows_OSize[i_idx];*/
		action_bytes = vecCommittedRows_OSize[i_idx];
		// Update the bandwidth usage for the output write-back
		excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle, action_bytes);
		/*excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle, 
				action_bytes, a_update, b_update, o_r_update, o_w_update);*/

		//o_traffic_write += action_bytes;
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

// This is a beautiful bw logger that gets the start cycle and end cycle 
//   of each tile multiplication and in a cycle accurate way says how
//   many extra cycles it is going to take.
// Its role is to keep track of bandwidth
uint64_t Scheduler_4::updateBWLogAndReturnExcess(uint64_t starting_cycle, 
		uint64_t ending_cycle, uint64_t action_bytes){

	total_traffic += action_bytes;
	//float action_bytes_f = action_bytes;

	//printf("%f - %f, %d\n", action_bytes_f, bytes_per_ns, action_bytes);
	//exit(1);

	float action_bytes_f = float(action_bytes);
	for(int i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		if((action_bytes_f > 0) & (bytes_per_ns ==  bw_logger[i_idx]))
			continue;

		float rem_cap = bytes_per_ns - bw_logger[i_idx];
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

uint64_t Scheduler_4::updateBWLogAndReturnExcess(uint64_t starting_cycle, 
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
void Scheduler_4::SyncPETimesWithMinCycle(uint64_t min_val){
	for(int idx=0; idx<params->getPECount(); idx++){
		pe_time[idx] = std::max(pe_time[idx], min_val);
	} 
	return; 
}

// All of the PEs should finish their work before fetching the next	B tiles
void Scheduler_4::SyncPETimes(){

	int max_val = *std::max_element(pe_time, pe_time+params->getPECount());
	std::fill(pe_time, pe_time+params->getPECount(), max_val);

	return;
}

// Gets the Column boundary and start address of matrix B row,
//   returns the start and stop address or the matrix B row, i.e., o_reuse
// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
void Scheduler_4::CalcOReuse(int k_idx, int j_idx_start, int & j_idx_stop){
	int a_reuse = params->getAReuse();
	j_idx_stop = j_idx_start;
	// Add rows until it either runs out of memory or reaches the last row
	for(int idx = j_idx_start; idx<b_tiled_rows; idx++){
		// Find the size of the new row 
		uint64_t extra_size = AccumulateSize('B', idx, idx+1, k_idx, std::min(k_idx+a_reuse, o_tiled_cols));
		// It means that it could not fit the new row in LLB and failed
		if(llb->DoesFitInLLB('B', extra_size) == 0) {break;}
		llb->AddToLLB('B', Req::read, extra_size, UPDATE_TRAFFIC);
		j_idx_stop++;
	}
	
	// if the if statements is correct, it means B partition of LLB still has room
	// Thus, let's increase the a_reuse value
	if((j_idx_start == 0) & (j_idx_stop == b_tiled_rows)){
		while(k_idx+a_reuse<o_tiled_cols){
			uint64_t extra_size = AccumulateSize('B', j_idx_start, j_idx_stop, 
					k_idx+a_reuse, k_idx+a_reuse+1);
			// It means that it could not fit the new row in LLB and failed
			if(llb->DoesFitInLLB('B', extra_size) == 0) {break;}
			llb->AddToLLB('B', Req::read, extra_size, UPDATE_TRAFFIC);
			a_reuse++;
		}
		params->setAReuse(a_reuse);
	}
	

	params->setOReuse(j_idx_stop- j_idx_start);

	return;
}


void Scheduler_4::PreCalculateAORowsSize(int j_start, int j_end, 
		int k_start, int k_end, int * b_llb_horizontalSum){

	#pragma omp parallel for
	for(int i_idx = 0; i_idx < o_tiled_rows; i_idx++){
		vecCommittedRows_ASize[i_idx] = AccumulateSize_AwrtB(i_idx, j_start, j_end, b_llb_horizontalSum);
		vecCommittedRows_OSize[i_idx] = AccumulateSize_OwrtAB(i_idx, j_start, j_end, k_start, k_end);
	}

	return;
}


// It fetches A and O rows as much as LLB allows. It first evicts the row with
//   the smallest end cycle time and gives its space to new rows. The number of new rows
//   is returned in numRows (range [0,o_tiled_rows] ).
void Scheduler_4::FetchAORows(int & numRows,
		int i_idx, int j_idx_start, int j_idx_stop, int k_idx, int *b_llb_horizontalSum){

	int a_reuse = params->getAReuse();
	uint64_t a_row_size, o_row_size;
	// This should normally free only one row; The reason I have used while is that in many cases
	//   the row size is zero (empty row), which we should skip!
	while(num_vecCommittedRows>0){
		uint64_t * smallest_end_cycle = std::min_element(vecCommittedRows_cycle, 
				vecCommittedRows_cycle+num_vecCommittedRows);
		int distance_from_start = std::distance(vecCommittedRows_cycle, smallest_end_cycle);
		uint64_t ending_cycle = vecCommittedRows_cycle[distance_from_start];
		// Find the A and O row size of the corresponding row and columns
		a_row_size = vecCommittedRows_ASize[vecCommittedRows_iidx[distance_from_start]];
		o_row_size = vecCommittedRows_OSize[vecCommittedRows_iidx[distance_from_start]];
		
		// remove the min from the list
		Swap(vecCommittedRows_iidx[distance_from_start], vecCommittedRows_iidx[num_vecCommittedRows-1]);
		Swap(vecCommittedRows_cycle[distance_from_start], vecCommittedRows_cycle[num_vecCommittedRows-1]);

		num_vecCommittedRows--;

		// If the size was greater than zero, then kudos! Get out of while loop
		if((a_row_size + o_row_size) > 0){
			// Remove the rows from LLB memory; A is flushed and O is written back
			llb->RemoveFromLLB('A', Req::read, a_row_size, DONT_UPDATE_TRAFFIC);
			llb->RemoveFromLLB('O', Req::write, o_row_size, UPDATE_TRAFFIC);
			SyncPETimesWithMinCycle(ending_cycle);

			break;
		}
	}
	
	numRows = 0;
	for (int idx = /*i_idx*/ first_row_notFetched; idx < o_tiled_rows; idx++){
		// I modified the accumulate functions to take into account smart PE tile fetch
		//   They will only be brought into LLB if they will be used in any computation
		a_row_size = AccumulateSize_AwrtB(idx, j_idx_start, j_idx_stop, b_llb_horizontalSum);
		o_row_size = AccumulateSize_OwrtAB(idx, j_idx_start, j_idx_stop, k_idx, 
				std::min(k_idx+a_reuse, o_tiled_cols));

		a_row_size = vecCommittedRows_ASize[idx];
		o_row_size = vecCommittedRows_OSize[idx];

		// We have enough space)
		/*
		if(!llb->DoesFitInLLB('A', a_row_size))
			printf("A is full! %lu %lu %lu\n", llb->GetASize(), llb->GetBSize(), llb->GetOSize());
		if(!llb->DoesFitInLLB('O', o_row_size))
			printf("O is full! %lu %lu %lu\n", llb->GetASize(), llb->GetBSize(), llb->GetOSize());
		*/
		if ((llb->DoesFitInLLB('A', a_row_size)) & (llb->DoesFitInLLB('O', o_row_size))){
			llb->AddToLLB('A', Req::read, a_row_size, UPDATE_TRAFFIC);
			llb->AddToLLB('O', Req::read, o_row_size, UPDATE_TRAFFIC);
			
			//vecCommittedRows_ASize[idx] = a_row_size;
			//vecCommittedRows_OSize[idx] = o_row_size;

			//numRows++; 
			first_row_notFetched = idx+1; //numRows = idx- i_idx+ 1;
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
void Scheduler_4::CalcBLLBHorizontalSum(int j_start, int j_end, 
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
uint64_t Scheduler_4::AccumulateSize_AwrtB(int i_idx, 
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
uint64_t Scheduler_4::AccumulateSize_OwrtAB(int i_idx,	
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
uint32_t Scheduler_4::ShouldIFetchThisOTile(int i_idx, 
		int j_idx_start, int j_idx_end, int k_idx){

	// A little hack. It we are bringing a whole a row, then
	//   output certainly gonna be touched
	if (j_idx_start == 0)
		return 1;
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
uint64_t Scheduler_4::AccumulateSize(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t size_tot = 0;
	switch (mat_name){
		case 'A':{
			//#pragma omp parallel for 
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				sum_i[i_idx-d1_start] = matrix->accumulateCSFSize('A',i_idx, d2_start, d2_end);
			size_tot = std::accumulate(sum_i, sum_i+d1_end-d1_start, 0.0);
				//for(int j_idx=d2_start; j_idx<d2_end; j_idx++){
				//	size += matrix->getCSFSize('A', i_idx, j_idx);}}
			break;}
		case 'B':{ 
			//#pragma omp parallel for 
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				sum_j[i_idx-d1_start] = matrix->accumulateCSFSize('B',i_idx, d2_start, d2_end);
			size_tot = std::accumulate(sum_j, sum_j+d1_end-d1_start, 0.0);

			//for(int j_idx=d1_start; j_idx<d1_end; j_idx++){
			//	for(int k_idx=d2_start; k_idx<d2_end; k_idx++){
			//		size += matrix->getCSFSize('B', j_idx, k_idx);}}
			break;}
		case 'O':{
			//#pragma omp parallel for 
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				sum_i[i_idx-d1_start] = matrix->accumulateCOOSize('O',i_idx, d2_start, d2_end);
			size_tot = std::accumulate(sum_i, sum_i+d1_end-d1_start, 0.0);

			//for(int i_idx=d1_start; i_idx<d1_end; i_idx++){
			//	for(int k_idx=d2_start; k_idx<d2_end; k_idx++){
			//		size += matrix->getCOOSize('O', i_idx, k_idx);}}
			break;}
		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}

/*
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
*/
	return size_tot;
}

template<typename T>
void Scheduler_4::Swap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

void Scheduler_4::printPEs(){
	for(int i=0; i<params->getPECount(); i++)
		printf("%lu ", pe_time[i]);
	printf("\n");
	return;

}

void Scheduler_4::PrintBWUsage(){
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
void Scheduler_4::PrintBWLog(){
	FILE * pFile;
	pFile = fopen ("bw_log.txt","w");
	for (uint64_t i=0; i< stats->Get_cycles(); i++)
		fprintf(pFile, "%f\n", bw_logger[i]);
	fclose(pFile);
	return;
}

#endif
