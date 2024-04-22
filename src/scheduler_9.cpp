#ifndef SCHEDULER_9_CPP
#define SCHEDULER_9_CPP

#include "scheduler_9.h"

// constructor -> intializer of the scheduler
Scheduler_9::Scheduler_9(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
	this->matrix = mtx;
	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Dimensions of inputs and the output tensors
	a_rows = matrix->GetARowSize();
	a_cols = matrix->GetAColSize();
	b_cols = matrix->GetBColSize();

	// #Bytes that can be transferred every cycle
	top_bytes_per_ns = (float)params->getTopBandwidth() / params->getFrequency();
	middle_bytes_per_ns = (float)params->getMiddleBandwidth() / params->getFrequency();

	// BW logger to have bw usage in cycle level accuracy
	//	This is a cool approach.
	// The top variable is for the top level DRAM->LLB
	// The middle variable is for the middle level LLB->PE
	top_bw_logger = new float[MAX_TIME];
	middle_bw_logger = new float[MAX_TIME];

	// Initialize PE arrays and BW_logger to all zeros
	std::fill(top_bw_logger, top_bw_logger+MAX_TIME, 0.0);
	std::fill(middle_bw_logger, middle_bw_logger+MAX_TIME, 0.0);

	// PE times
	pe_time = new uint64_t[params->getPECount()];
	std::fill(pe_time, pe_time+params->getPECount(), 0);

	pe_utilization_logger = new uint64_t[params->getPECount()];
	std::fill(pe_utilization_logger, pe_utilization_logger+params->getPECount(), 0);

	// If using nnz-based scheduler, then allocate and initialize related array
	if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
		nnz_counts = new uint64_t[params->getPECount()];
		std::fill(nnz_counts, nnz_counts+params->getPECount(), 0);
	}

	// IMPORTANT NOTE: vecCommittedACols_jidx & vecCommittedACols_cycle
	//   do not work with absolute row address. That is why we have
	//   the iidx vector to keep track of the row address
	vecCommittedACols_jidx = new int[a_cols];
	vecCommittedACols_cycle = new uint64_t[a_cols];
	// Keep the number of committed rows need to be retired
	num_vecCommittedCols = 0;
	// reset the round robin scheduler counter
	round_robin_slot = 0;

	return;
}

// Destructor -> delete [] bunch of dynamically allocated arrays
Scheduler_9::~Scheduler_9(){
	delete [] top_bw_logger;
	delete [] middle_bw_logger;
	delete [] pe_time;
	delete [] pe_utilization_logger;
	delete [] nnz_counts;

	delete [] vecCommittedACols_jidx;
	delete [] vecCommittedACols_cycle;

	return;
}

// Reset all the internal stats; Used when there are multiple runs in one main file
//	Usually used for bandwidth scaling sweep
void Scheduler_9::Reset(){

	// Reset PE_time and BW_logger
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	std::fill(top_bw_logger, top_bw_logger+MAX_TIME, 0.0);
	// Sometimes bandwidth changes before reset, so update it
	top_bytes_per_ns = (float)params->getTopBandwidth() / params->getFrequency();

	return;
}

/* Dataflow in pseudo code is :
 *
 * for(j=0; j<a_cols; j++)
 *	for(i=0; i<a_rows; i++)
 *		for(k=0; k<b_cols; k++)
 *			a[i][k] += a[i][j]*b[j][k];
 *
 */
int Scheduler_9::Run(){

	//TODO: Make the reduce per output matrix row! Do it the last!

	uint64_t pproduct_write_back = 0, pproduct_count =0;
	uint32_t * a_size_per_pe = new uint32_t[a_cols];
	uint32_t * b_size_row = new uint32_t[a_cols];
	uint32_t * o_size_per_pe = new uint32_t[a_cols];
 	uint32_t * macc_count_per_pe = new uint32_t[a_cols];
	uint32_t * a_elements_in_col = new uint32_t[a_cols];

	int * pe_indices = new int[a_rows];

	// Pre-calculate what is the size of A, B, and O per PE of each
	//	A col/B row. It also finds out how many scalars per A row
	//	and MACCs per pe corresponding to the col A scalars
	PreCalculatePerColStats(a_size_per_pe, b_size_row, o_size_per_pe,
			macc_count_per_pe, a_elements_in_col);

	// Iterate over cols A/ rows B
	for(int j_idx = 0; j_idx < a_cols; j_idx++){
		// The size that the next col A/row B computation will occupy
		uint64_t size_next_task = ((a_size_per_pe[j_idx] + o_size_per_pe[j_idx]) *
				a_elements_in_col[j_idx]) + b_size_row[j_idx] ;

		if(size_next_task > llb->GetCapacity()){
			printf("Task size exceeds maximum!\n A: %lu, B:%lu, O: %lu\n",
					a_size_per_pe[j_idx]*a_elements_in_col[j_idx], b_size_row[j_idx],
					o_size_per_pe[j_idx]*a_elements_in_col[j_idx]);

		}
		// If there is not enough room in LLB: wait until enough space is available!
		while(!llb->DoesFitInLLB_Minimal(size_next_task)){
			RetireOneAColumnTask(a_size_per_pe, b_size_row, o_size_per_pe, a_elements_in_col);
		}//while

		// Pick the PEs that are supposed to perform the outer products for j_idx
		PickPEsAccordingToPolicy(a_elements_in_col[j_idx], pe_indices);
		// Find the earliest time that data can be fetched from DRAM
		uint64_t batch_starting_cycle = FindBatchStartingCycle(
				a_elements_in_col[j_idx], pe_indices);
		// Add data to LLB
		llb->AddToLLB_Minimal(size_next_task);

		// Increment number of partial products that need to be brought to
		//		the chip to be reduced
		pproduct_count += macc_count_per_pe[j_idx] * a_elements_in_col[j_idx];
		// Top traffic that need to be read so computation can start
		//		(it consists of A and B entries)
		uint64_t top_traffic_read = (a_size_per_pe[j_idx] * a_elements_in_col[j_idx])
		 	+ b_size_row[j_idx];
		// Update top bw usage
		uint64_t endingCycle_top = UpdateBWLog(batch_starting_cycle,
			top_traffic_read, top_bw_logger, top_bytes_per_ns);

		// The size of write, when the computation is over
		uint64_t top_traffic_write = o_size_per_pe[j_idx] * a_elements_in_col[j_idx];

		//*** Schedule each scalar in the A column; Update pe_time, middle_noc_bw ***
		for(uint32_t i_idx = 0; i_idx < a_elements_in_col[j_idx]; i_idx++){
			// Chose the PE that is going to be updated
			uint64_t * chosen_pe = &pe_time[pe_indices[i_idx]];

			uint64_t middle_traffic = top_traffic_read + top_traffic_write;
			// For the middle DOT we should start when computation takes place
			uint64_t endingCycle_middle = 0;
			// Update the middle DOT traffic for the A scalar
			if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
				endingCycle_middle = UpdateBWLog(*chosen_pe,
					middle_traffic, middle_bw_logger, middle_bytes_per_ns);
			}
			// Update the PE logger time
			*chosen_pe = std::max( std::max(endingCycle_top, endingCycle_middle)
						,*chosen_pe + macc_count_per_pe[j_idx]);

			// PE time becomes the maximum of pe_time, top_bw_logger, and middle_bw_logger
			uint64_t max_time_accessed_in_batch =
				std::max(max_time_accessed_in_batch, *chosen_pe);
		} // for i_idx -> Scheduling scalars of A column

		// Find the earliest time that data can be written to DRAM
		uint64_t batch_ending_cycle = FindBatchEndingCycle(
				a_elements_in_col[j_idx], pe_indices);

		// Update the Top NoC (main memory)
		endingCycle_top = UpdateBWLog(batch_ending_cycle,
			top_traffic_write, top_bw_logger, top_bytes_per_ns);

		// Update the commit -> retire related arrays
		vecCommittedACols_cycle[num_vecCommittedCols] = batch_ending_cycle;
		vecCommittedACols_jidx[num_vecCommittedCols] = j_idx;


		// Incerement the size of data that needs to be read for the final
		//	reduce/merge phase
		pproduct_write_back += top_traffic_write;
		// Increment the number of committed cols awaiting to be retired
		num_vecCommittedCols++;
	} // for j_idx

	//retire (flush/writeback) all the remaining cols A / rows B
	while(num_vecCommittedCols){
		RetireOneAColumnTask(a_size_per_pe, b_size_row, o_size_per_pe, a_elements_in_col);
	}

	// The output data size that is going to be written back as final result
	uint64_t final_write_back = matrix->CalculateNotTiledMatrixProduct();
	// The number of accumulate operations to reduce partial products
	uint64_t reduce_accumulate_cycles = pproduct_count -
		matrix->GetNotTiledOutputNNZCount();
	// Pick the PEs that are supposed to perform the outer products for j_idx
	PickPEsAccordingToPolicy(1, pe_indices);
	uint64_t * chosen_pe = & pe_time[pe_indices[0]];
	// Update the Top NoC (main memory) -> This update is only for read bytes

	uint64_t endingCycle_top = UpdateBWLog(*chosen_pe,
		pproduct_write_back, top_bw_logger, top_bytes_per_ns);
	// Update the Top NoC (main memory)
	uint64_t endingCycle_middle = 0;
	if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
		// The data that needs to be read to PEs and then written back to LLB
		uint64_t middle_traffic = pproduct_write_back + final_write_back;
		endingCycle_middle = UpdateBWLog(*chosen_pe,
				middle_traffic, middle_bw_logger, middle_bytes_per_ns);
	}
	// Update the PE logger time
	*chosen_pe = std::max( std::max(endingCycle_top, endingCycle_middle)
			,*chosen_pe + reduce_accumulate_cycles);
	// Update the Top NoC (main memory) -> This update is only for read bytes
	endingCycle_top = UpdateBWLog(*chosen_pe,
		final_write_back, top_bw_logger, top_bytes_per_ns);

	uint64_t ending_cycle_memory = std::max(endingCycle_top, endingCycle_middle);

	stats->Set_cycles(std::max(ending_cycle_memory, *chosen_pe));
	stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());

	delete [] a_size_per_pe;
	delete [] b_size_row;
	delete [] o_size_per_pe;
 	delete [] macc_count_per_pe;
	delete [] a_elements_in_col;

	delete [] pe_indices;
	return 0;
}

// Retires the earliest finished outstanding committed column of A/Row of B
void Scheduler_9::RetireOneAColumnTask(uint32_t * a_size_per_pe, uint32_t * b_size_row,
	 	uint32_t * o_size_per_pe, uint32_t * a_elements_in_col){
	// Find the earliest column of A that is committed but not retired yet
	uint64_t * earliest_end_cycle = std::min_element(vecCommittedACols_cycle,
		vecCommittedACols_cycle+num_vecCommittedCols);
	// Find the ptr address of the earliest end cycle
	int distance_from_start = std::distance(vecCommittedACols_cycle, earliest_end_cycle);
	// We need to keep ending cycle for SyncPETimesWithMin Cycle in 10 lines
	uint64_t ending_cycle = vecCommittedACols_cycle[distance_from_start];
	// The absolute j_idx of the retiring column
	int col_idx = vecCommittedACols_jidx[distance_from_start];
	// Find the A and O row size of the corresponding row and columns
	uint32_t llb_size_task = ((a_size_per_pe[col_idx] + o_size_per_pe[col_idx])
			*	a_elements_in_col[col_idx] )+ b_size_row[col_idx];
	// remove the min from the list
	Swap(vecCommittedACols_jidx[distance_from_start],
			vecCommittedACols_jidx[num_vecCommittedCols-1]);
	Swap(vecCommittedACols_cycle[distance_from_start],
			vecCommittedACols_cycle[num_vecCommittedCols-1]);

	// One row is fetched, therefore available slots is decremented by one
	num_vecCommittedCols--;
	// Retire the column and flush/write back the corresponding data from the LLB
	llb->RemoveFromLLB_Minimal(llb_size_task);
	// Updates the PE time according to the retired task!
	SyncPETimesWithMinCycle(ending_cycle);


	return;
}


// This is a beautiful bw logger that gets the start cycle
//   of each tile multiplication and in a cycle accurate way says
//   when the data transfer is finished
// Its role is to keep track of bandwidth either for the top level
//  or middle level, depending on the bw_logger and bytes_per_ns assignment
uint64_t Scheduler_9::UpdateBWLog(uint64_t starting_cycle, uint64_t action_bytes,
		float *bw_logger, float bytes_per_ns){

	double action_bytes_f = double(action_bytes);
	for(uint64_t i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		float rem_cap = bytes_per_ns - bw_logger[i_idx];
		// Move on until finding the first DRAM bw available cycle
		if((action_bytes_f > 0) & (rem_cap == 0))
			continue;
		// Use the available BW, but this is not the end
		if(action_bytes_f > (double)rem_cap){
			bw_logger[i_idx] = bytes_per_ns;
			action_bytes_f -= (double)rem_cap;
		}
		// Last cycle needed to transfer the specified data
		else{
			bw_logger[i_idx] += action_bytes_f;
			return i_idx;
		}
	}
	printf("Starting cycle: %lu, Bytes: %lu, remaining bytes: %f, bandwidth: %f bytes/ns\n",
			starting_cycle, action_bytes, action_bytes_f, bytes_per_ns);
	printf("%d bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}



// Pre calculate the A, B, and O size per PE of each col A/Row B
// This function also figures out how many non-zeros are in each column of A
//		and how many MACC (more precisely multiply) operations each corresponding
//		PE unit needs to perform
void Scheduler_9::PreCalculatePerColStats(uint32_t * a_size_per_pe, uint32_t * b_size_row,
		uint32_t * o_size_per_pe,	uint32_t * macc_count_per_pe, uint32_t * a_elements_per_col){

	for(int j_idx=0; j_idx< a_cols; j_idx++){
		// This function just looks at the numbers and does not do actual multiplication
		matrix->OuterProduct(j_idx,	a_size_per_pe[j_idx], b_size_row[j_idx],
				o_size_per_pe[j_idx], macc_count_per_pe[j_idx], a_elements_per_col[j_idx]);
	}
	return;
}

// Find the earliest (pe_time) end time among the chosen pe indices
uint64_t Scheduler_9::FindBatchEndingCycle(
		uint32_t	a_elements_in_col, int * pe_indices){

	// If no A scalar is in the column, then return the dummy 0 value!
	if(a_elements_in_col == 0)
		return 0;

	// Iterate over the pe_times of pe_indices to find the
	//	earliest starting time
	uint64_t batch_ending_cycle = pe_time[pe_indices[0]];
	for(uint32_t pe_idx = 1; pe_idx < a_elements_in_col; pe_idx++){
		if (batch_ending_cycle > pe_time[pe_indices[pe_idx]])
			batch_ending_cycle = pe_time[pe_indices[pe_idx]];
	}
	return batch_ending_cycle;
}

// Find the earliest (pe_time) start time among the chosen pe indices
uint64_t Scheduler_9::FindBatchStartingCycle(
		uint32_t	a_elements_in_col, int * pe_indices){

	// If no A scalar is in the column, then return the dummy 0 value!
	if(a_elements_in_col == 0)
		return 0;

	// Iterate over the pe_times of pe_indices to find the
	//	earliest starting time
	uint64_t batch_starting_cycle = pe_time[pe_indices[0]];
	for(uint32_t pe_idx = 1; pe_idx < a_elements_in_col; pe_idx++){
		if (batch_starting_cycle < pe_time[pe_indices[pe_idx]])
			batch_starting_cycle = pe_time[pe_indices[pe_idx]];
	}
	return batch_starting_cycle;
}


// This synchronization happens when a row related calculation is over
//		So, when we are bringing a new row, we make sure that the timing has been updated
//		min_val is the end_cycle of the committed row, which will be the start_cycle of
//		the fetched row
void Scheduler_9::SyncPETimesWithMinCycle(uint64_t min_val){
	for(int idx=0; idx<params->getPECount(); idx++){
		pe_time[idx] = std::max(pe_time[idx], min_val);
	}
	return;
}

// Returns count_needed pe indices according to the scheduling policy
void Scheduler_9::PickPEsAccordingToPolicy(
		int count_needed, int * pe_indices){

	int vec_counter = 0;

	static_distributor dist_policy = params->getStaticDistributorModelMiddle();
	if(dist_policy == static_distributor::round_robin){
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices[vec_counter++] = (pe_idx+round_robin_slot)%params->getPECount();
		}
		round_robin_slot = (round_robin_slot+count_needed)%params->getPECount();
	}
	else if(dist_policy == static_distributor::oracle){
		QuickSort2Desc(pe_time, pe_utilization_logger, 0, params->getPECount()-1);
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices[vec_counter++] = pe_idx%params->getPECount();
		}
	}
	else if(dist_policy == static_distributor::nnz_based){
		QuickSort3Desc(nnz_counts, pe_time, pe_utilization_logger, 0, params->getPECount()-1);
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices[vec_counter++] = pe_idx%params->getPECount();
		}
	}
	else{
		printf("Static Scheduler Policy Is Not Available!\n");
		exit(1);
	}

	return;
}


// Sorting a, b, and c according to a (b and c are dependent)
// This function is used for finding to corresponding PE units, and pe_utilization_logger
//	a is the nnz count for each pe, and pe is the pe progress time
template<class A, class B, class C> void Scheduler_9::QuickSort3Desc(
		A a[], B b[], C c[], int l, int r){

	int i = l;
	int j = r;
	A v = a[(l + r) / 2];
	A w = b[(l + r) / 2];
	do {
		while(1){
			if(a[i] < v)i++;
			else if((a[i] == v) & (b[i] < w))i++;
			else break;
		}
		while(1){
			if(a[j] > v)j--;
			else if((a[j] == v) & (b[j] > w))j--;
			else break;
		}

		if (i <= j){
			Swap(a[i], a[j]);
			Swap(b[i], b[j]);
			Swap(c[i], c[j]);
			i++;
			j--;
		};
	} while (i <= j);
	if (l < j)QuickSort3Desc(a, b, c, l, j);
	if (i < r)QuickSort3Desc(a, b, c, i, r);
}

// Sorting a and b according to a (b is dependent)
// This function is used for finding to corresponding PE units
//	a is the nnz count for each pe, and pe is the pe progress time
template<class A, class B> void Scheduler_9::QuickSort2Desc(
		A a[], B b[], int l, int r){

	int i = l;
	int j = r;
	A v = a[(l + r) / 2];
	A w = b[(l + r) / 2];
	do {
		while(1){
			if(a[i] < v)i++;
			else if((a[i] == v) & (b[i] < w))i++;
			else break;
		}
		while(1){
			if(a[j] > v)j--;
			else if((a[j] == v) & (b[j] > w))j--;
			else break;
		}

		if (i <= j){
			Swap(a[i], a[j]);
			Swap(b[i], b[j]);
			i++;
			j--;
		};
	} while (i <= j);
	if (l < j)QuickSort2Desc(a, b, l, j);
	if (i < r)QuickSort2Desc(a, b, i, r);
}


// Swap two variables
template<typename T> void Scheduler_9::Swap(T &a, T &b){

	T t = a;
	a = b;
	b = t;
}

void Scheduler_9::PrintBWUsage(){
	//uint64_t size = 0;
	double size_top = 0, size_middle = 0;
	for (uint64_t i=0; i<= stats->Get_cycles(); i++){
		size_top += top_bw_logger[i];
		size_middle += middle_bw_logger[i];
	}
	for(auto i=stats->Get_cycles()+1; i<stats->Get_cycles()+1000; i++){
		if(top_bw_logger[i] !=0) { printf("Shiiiit!\n"); break;}
	}
	printf("Top bandwidth logger: %lu bytes, %f GBs\n", (uint64_t)size_top, (double)size_top/(1024.0*1024.0*1024.0));
	printf("Middle bandwidth logger: %lu bytes, %f GBs\n", (uint64_t)size_middle, (double)size_middle/(1024.0*1024.0*1024.0));

	return;
}


#endif
