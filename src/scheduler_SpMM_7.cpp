#ifndef SCHEDULER_7_SPMM_CPP
#define SCHEDULER_7_SPMM_CPP

#include "scheduler_SpMM_7.h"

/*	TODO:
 *	(Done) 1- change AccumulateSize interface! Add dense option to it!
 *		Change all the AccumulateSizes that are used in the scheduler
 *	(Done) 2- CalcOReuse
 *	(Done) 3- PreCalculateAORowSize
 *  (Done) 4- FetchAORows
 *	(Done) 5- Schedule
 *
 */

/* There are three DOT levels that instead of calling them DRAM, LLB, PE, Registers
 *   I have called them TOP (DRAM->LLB), MIDDLE (LLB->PE), BOTTOM (PE->Registers)
 *
 * Each level has its own modules
 *	Top level: Extract (TBuild, Search, Mbuild, Scanners), Intersection
 *	Middle Level: Extract (Search, MBuild, Scanners), Intersection
 *	Bottom Lebel: Extract (Scanners), Intersection unit
 */


// constructor -> intializer of the scheduler
Scheduler_SpMM_7::Scheduler_SpMM_7(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
	this->matrix = mtx;
	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Dimensions of inputs and the output tensors
	if(params->getCompKernel() == kernel::SpMM){
		o_tiled_rows = matrix->getTiledARowSize();
		o_tiled_cols = (int)ceil((float)params->getNumDenseCols()/params->getTileSize());
		b_tiled_rows = matrix->getTiledAColSize();
	}
	else{
		o_tiled_rows = matrix->getTiledORowSize();
		o_tiled_cols = matrix->getTiledOColSize();
		b_tiled_rows = matrix->getTiledBRowSize();
	}
	// BW logger to have bw usage in cycle level accuracy
	//	This is a cool approach, I am considering doing the same for PEs
	bw_logger = new float[MAX_TIME];
	middle_bw_logger = new float[MAX_TIME];

	// PE times
	pe_time = new uint64_t[params->getPECount()];

	// Initialize PE arrays and BW_logger to all zeros
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	std::fill(bw_logger, bw_logger+MAX_TIME, 0.0);

	// Each row that commit is recorded in these arrays
	//	They are mainly committed, but waiting to be retired
	//	to be evicted from the LLB memory finally

	// IMPORTANT NOTE: vecCommittedRows_iidx & vecCommittedRows_cycle
	//   do not work with absolute row address. That is why we have
	//   the iidx vector to keep track of the row address
	vecCommittedRows_iidx = new int[o_tiled_rows];
	vecCommittedRows_cycle = new uint64_t[o_tiled_rows];

	// Continue of IMPORTANT NOTE: In contrast,
	//   vecCommittedRows_ASize & vecCommitted_OSize both work with
	//   absolute row addresses. if you are accessing array entry
	//   i_idx then it corresponds to row i_idx of matrix
	vecCommittedRows_ASize = new uint64_t[o_tiled_rows];
	vecCommittedRows_OSize = new uint64_t[o_tiled_rows];

	// #Bytes that can be transferred every cycle
	bytes_per_ns = (float)params->getBandwidth() / params->getFrequency();
	middle_bytes_per_ns = (float)params->getMiddleBandwidth() / params->getFrequency();

	// Keep the number of committed rows need to be retired
	num_vecCommittedRows = 0;

	/* Top Level (DRAM -> LLB)
	 *	Extract and Intersect variables
	 */
	// Vectors to record the extract overhead (t_build, search_mbuild)
	//	to take into account overheads in high-fidelity simulation
	vec_O_ExtractOverhead_top = new uint64_t[o_tiled_rows];
	vec_A_ExtractOverhead_top = new uint64_t[o_tiled_rows];
	// Pointer assignment; Just an alias to make the code readable
	extractor_intersect_done_top = vec_A_ExtractOverhead_top;

	// Get the parallelism for builing basic tile in top level for tensors A, B, & O
	//	DRAM->LLB
	par_tbuild_A_top = params->getParallelismBasicTileBuildTop_A();
	par_tbuild_B_top = params->getParallelismBasicTileBuildTop_B();
	par_tbuild_O_top = params->getParallelismBasicTileBuildTop_O();

	// A time-keeper for each basic tile builder in LLB of tensors A & O
	tbuild_A_time_top = new uint64_t[par_tbuild_A_top];
	tbuild_O_time_top = new uint64_t[par_tbuild_O_top];
	// Set all the entries to 0
	std::fill(tbuild_A_time_top, tbuild_A_time_top +par_tbuild_A_top, 0);
	std::fill(tbuild_O_time_top, tbuild_O_time_top +par_tbuild_O_top, 0);
	// Logs when the top (DRAM->LLB) is being used
	//   only if it is CAMBased! If it is instant or ideal
	//   it will take 0 or 1 cycle, respectively.
	if (params->getIntersectModelTop() == intersect::skipModel){
		intersect_logger_top = new uint8_t[MAX_TIME];
		std::fill(intersect_logger_top, intersect_logger_top+MAX_TIME, 0);
		b_intersect_vector_top = new int[o_tiled_rows];
		intersect_overhead_top = new uint8_t[o_tiled_rows];
	}

	// Get the parallelism for search in LLB of tensor B
	//   Also its log2 value!
	par_search_B_top = params->getParallelismSearchTop();
	log2_search_B_top = 0;
	while (par_search_B_top >>= 1) ++log2_search_B_top;
	par_search_B_top = params->getParallelismSearchTop();

	int row_size = params->getTileSize();
	log2_mbuild_B_top = 0;
	while (row_size >>= 1) ++log2_mbuild_B_top;


	if((params->getIntersectModelMiddle() != intersect::instantModel)
			| (params->getSearchModelMiddle() != search_tiles::instant)){
		par_middle = params->getParallelismMiddle();
	}
	else
		par_middle = 1;

	// Used only for the round-robin scheduler
	round_robin_slot = 0;
	// DEBUGGING variables. DON'T REMOVE
	a_traffic = 0; b_traffic = 0; o_traffic_read = 0; o_traffic_write = 0; total_traffic = 0;
	a_bwl_traffic =0; b_bwl_traffic = 0; o_bwl_traffic_read = 0; o_bwl_traffic_write = 0;
	total_traffic_middle = 0;


	return;
}

// Destructor -> delete [] bunch of dynamically allocated arrays
Scheduler_SpMM_7::~Scheduler_SpMM_7(){
	delete [] bw_logger;
	delete [] pe_time;
	delete [] vecCommittedRows_iidx;
	delete [] vecCommittedRows_cycle;
	delete [] vecCommittedRows_ASize;
	delete [] vecCommittedRows_OSize;
	delete [] vec_O_ExtractOverhead_top;
	delete [] vec_A_ExtractOverhead_top;
	delete [] tbuild_A_time_top;
	delete [] tbuild_O_time_top;
	if (params->getIntersectModelTop() == intersect::skipModel){
		delete [] intersect_logger_top;
		delete [] b_intersect_vector_top;
		delete [] intersect_overhead_top;
	}
	return;
}

// Reset all the internal stats; Used when there are multiple runs in one main file
//	Usually used for bandwidth scaling sweep
void Scheduler_SpMM_7::Reset(){
	// Reset PE_time and BW_logger
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	std::fill(bw_logger, bw_logger+MAX_TIME, 0.0);
	std::fill(middle_bw_logger, middle_bw_logger+MAX_TIME, 0.0);

	// Reset number of committed vectors (no vector waiting to be retired)
	num_vecCommittedRows = 0;
	// Sometimes bandwidth changes before reset, so update it
	bytes_per_ns = (float)params->getBandwidth() / params->getFrequency();
	middle_bytes_per_ns = (float)params->getMiddleBandwidth() / params->getFrequency();
	// Reset tp DOT basic tile build scheduler arrays
	std::fill(tbuild_A_time_top, tbuild_A_time_top +par_tbuild_A_top, 0);
	std::fill(tbuild_O_time_top, tbuild_O_time_top +par_tbuild_O_top, 0);
	// Reset top DOT intersect unit time logger
	if (params->getIntersectModelTop() == intersect::skipModel){
		std::fill(intersect_logger_top, intersect_logger_top+MAX_TIME, 0);
	}
	// Reset round-robin scheduler slot holder
	round_robin_slot = 0;

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
 *	NOTE: Pseudo-code shows the loop-nests precisely, but in implementation
 *	        there are some changes, but they follow the same logic and order
 *
 * For row based synchronization we need two vectors
 * 1) A and O rows in LLB (every time we will add some to fill up the LLB portion)
 * 2) A and O row ID and finish time, so if it is evicted then the rest of the PEs
 *			added later should start not earlier than that time.
 *
 */
int Scheduler_SpMM_7::Run(){

	// If we have a static tiling mechanism (ExTensor), then do some initialization
	if(params->getTilingMechanism() == tiling::t_static){
		// I, J, and K of the LLB (Top DOT) tiles
		I_top_tile = params->getITopTile();
		J_top_tile = params->getJTopTile();
		K_top_tile = params->getKTopTile();
		// O_reuse and a_reuse are constant values in static tiling
		params->setAReuse(K_top_tile);
		params->setOReuse(J_top_tile);
		// Shows how many row slots we have in total
		total_row_slots = I_top_tile;
		// Shows how many of the rows are available to load data
		remaining_row_slots = I_top_tile;

		printf("I_top: %d, J_top: %d, K_top: %d\n", I_top_tile, J_top_tile, K_top_tile);
	}

	int a_reuse = params->getAReuse();
	a_reuse_const = a_reuse;

	// Iterations over B cols / O cols / a_reuse
	for(int k_idx = 0; k_idx < o_tiled_cols; /* EMPTY */){
		// Range for the a_reuse = [j_idx_start, j_idx_stop)
		int j_idx_start = 0, j_idx_stop  = 0;
		// Iteration over B rows (in LLB tile granularity) / A cols / o_reuse
		while(j_idx_start < b_tiled_rows){
			// Calculate O_reuse and fetch B tiles
			//  Calculate B_extract top level overhead as well
			CalcOReuse(k_idx, j_idx_start, j_idx_stop);
			//printf("CalcOReuse\n");
			// Load the latest updated a_reuse (it might be bigger than user defined)
			a_reuse = params->getAReuse();
			// Find out what is the A and O row size we need to fetch
			//	after LLB intersection
			//	Also, calculates extract overhead for A and O rows
			//	in addition to the intersection overhead
			PreCalculateAORowsSize(j_idx_start, j_idx_stop,
					k_idx, std::min(k_idx+a_reuse, o_tiled_cols));
			//printf("PreCalculateAORowSize\n");

			// It shows what is the first row in A and O which has not been fetched yet
			first_row_notFetched = 0;
			// iteration over A rows / O rows
			//   there is no i_idx++ since it goes as far as LLB assists
			for(int i_idx = 0; i_idx < o_tiled_rows;/* EMPTY */){
				int numRows = 0;
				// fill up LLB with A and O rows!
				// It fetches A and O rows as much as LLB allows. It first evicts the row with
				//   the smallest end cycle time and gives its space to new rows. The number of new rows
				//   is returned in numRows (range [0,o_tiled_rows] ).
				// Also for the fetched rows calculate the cycle where extract and intersect are over
				//   in the top level (DRAM-> LLB)
				FetchAORows(numRows, i_idx, j_idx_start, j_idx_stop, k_idx);
				// Processing each row one by one; since row sync gets rows in different starting cycles
				for(int i_row = 0; i_row< numRows; i_row++){
					// Run and schedule the multiplication of one row of A by an LLB tile of B
					Schedule(i_idx, j_idx_start, j_idx_stop, k_idx);
					// For loop increment
					i_idx++;
					// If the LLB usage surpasses the LLB capacity, then terminate early with error
					//   I have disabled the checks for LLB size to let it nicely exit and let the rest of
					//   the program to proceed (other I, J, and K configurations)
					if(params->getTilingMechanism() == tiling::t_static){
						if(llb->GetSize() > llb->GetCapacity()){
							printf("LLB FULL\n");
							stats->Set_cycles(0);
							stats->Set_runtime(0);
							return 1;}
					}
					// If we have exceeded the LLB capacity already stop it! empty some A and O rows
					//	and proceed when we have enough space
					if(llb->GetSize() >= llb->GetCapacity()){	break;}
				}
			}
			if(params->getTilingMechanism()==tiling::t_static){
				remaining_row_slots = total_row_slots;
			}

			// Update B LLB col value
			j_idx_start = j_idx_stop;
			// Flush everything inside LLB since we are going to bring new data
			llb->EvictMatrixFromLLB('B', DONT_UPDATE_TRAFFIC);
			llb->EvictMatrixFromLLB('A', DONT_UPDATE_TRAFFIC);
			llb->EvictMatrixFromLLB('O', UPDATE_TRAFFIC);
			// Mark all A and B tiles as not present in LLB
			//	When 0, the tile is not in LLB
			//	When 1, the tile is present in LLB
			std::fill(matrix->a_csr_outdims->vals,
					matrix->a_csr_outdims->vals + matrix->a_csr_outdims->nnz, 0.0);
			std::fill(matrix->b_csc_outdims->vals,
					matrix->b_csc_outdims->vals + matrix->b_csc_outdims->nnz, 0.0);

			// All rows retired; start a new life :D
			num_vecCommittedRows = 0;
			// All PEs should finish their tasks before fetching new B tiles
			SyncPETimes();
		}
		// Outermost for loop counter
		k_idx += a_reuse;
		// B LLB column is changed, load back the user defined a_reuse val
		params->setAReuse(a_reuse_const);
	}

	return 0;
}

// Gets a fiber of A and O and a rectangle LLB tile of B and schedules them
//   The ending cycles of each row is recorded in vecCommittedRows for
//   synnchronization, fetching, and committing rows
void Scheduler_SpMM_7::Schedule(int i_idx, int j_idx_start, int j_idx_stop, int k_idx){

	//printf("i_idx:%d, j_idx: %d-%d, k_idx: %d\n", i_idx, j_idx_start, j_idx_stop, k_idx);

	int a_reuse = params->getAReuse();
	int cycles_comp = 0, o_tile_intersect_count = 0;
	uint64_t batch_starting_cycle = 0, batch_ending_cycle= 0, mem_finish_cycle = 0,
					 action_bytes = 0, OSize_preComp = 0, OSize_postComp =0;
	/* DEBUGGING CODE */
	uint64_t a_update=0, b_update=0, o_r_update=0, o_w_update=0;

	batch_ending_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
	// Early termination if B size is zero; It does not happen anymore
	/* I added the second early termination clause that actually happens
	 * and can save computation*/
	// If either A LLB row or B LLB tile are empty(null) then early terminate
	if ((llb->GetBSize() == 0) | (vecCommittedRows_ASize[i_idx] == 0)){
		vecCommittedRows_iidx[num_vecCommittedRows] = i_idx;
		vecCommittedRows_cycle[num_vecCommittedRows] = batch_ending_cycle;
		num_vecCommittedRows++;
		return;
	}
	// 1- Take into account top level Extract and Intersection overheads
	// Batch starting cycle is when the extract and intersection phases are
	//   done for the top level. Now, The middle level (LLB->PE)
	//   extract and intersect should be added to this batch_starting cycle as well
	batch_starting_cycle = std::max(extractor_intersect_done_top[i_idx], batch_ending_cycle);

	/********************************************/
	/* FWA: Find start and end of the A vector! */
	// lower and upper bound of A_CSR and B_CSC fiber to intersect
	//	find the intersect between [lower_bound, upper_bound]
	int lower_bound = j_idx_start; int upper_bound = j_idx_stop;
	// Find start and end of the A_CSR fiber
	// Positions
	int a_pos_start = matrix->a_csr_outdims->pos[i_idx];
	int a_pos_end = matrix->a_csr_outdims->pos[i_idx+1];
	// Indexes (J values of row A)
	int * vecA_begin = &(matrix->a_csr_outdims->idx[a_pos_start]);
	int * vecA_end = &(matrix->a_csr_outdims->idx[a_pos_end-1]);
	// Move to the strating point of row A according to lower_bound
	while ((*vecA_begin<lower_bound) & (vecA_begin<=vecA_end)) vecA_begin++;
	int vecA_offset = vecA_begin -  &(matrix->a_csr_outdims->idx[a_pos_start]);
	/* FWA: Until here!  ************************/

	std::vector<int> j_indexes;
	std::vector<int> a_entries_count;
	// Iterates over the A LLB tile row and divides the consecutive basic tiles
	//   in a way that each group fits into the middle level buffer (PE buffer)
	//   For example: tile 0..5 -> Buffer 0, tile 6..8-> Buffer 1, ...
	findJIndexesInTheRange(i_idx, j_idx_start, j_idx_stop,
			vecA_begin, vecA_end, j_indexes, a_entries_count);

	// A datastructure to keep track of extract and intersection overheads
	//	for each middle level (PE) DOT unit that is being used
	// For Bottom DOT, there are par_middle operators
	uint32_t * overheads_middle_perDOT = new
	 	uint32_t[a_entries_count.size() * par_middle + 1];

	std::fill(overheads_middle_perDOT,
			overheads_middle_perDOT + (a_entries_count.size()*par_middle),
		 	0);

	//printf("a_entries_count.size(): %d, par_middle: %d\n", a_entries_count.size(),par_middle);
/*
	printf("%d:(%d) ",i_idx,a_entries_count.size());
	for(int i=0; i<a_entries_count.size();i++)
		printf("%d ",a_entries_count.at(i));
	printf("\n");
*/


	// Iterate over all B column fibers (or O coulmns)
	for(int k_inner = k_idx; k_inner < std::min(k_idx+a_reuse, o_tiled_cols); k_inner++){
		// Variables for output tile BW related calculations
		OSize_preComp = 0; o_tile_intersect_count = 0;

		for(int t_idx = 0; t_idx<a_entries_count.size()/*j_indexes.size()-1*/;t_idx++){

			uint64_t middle_DOT_traffic = 0;
			/********************************************/
			/* FWB: Find start and end of the B vector! */
			// Update lower and upper bound according to vector A
			lower_bound = j_indexes.at(t_idx);
			upper_bound = j_indexes.at(t_idx+1);
			// Find start and end of the B_CSC fiber
			int b_pos_start = matrix->b_csc_outdims->pos[k_inner];
			int b_pos_end = matrix->b_csc_outdims->pos[k_inner+1];
			int * vecB_begin = &(matrix->b_csc_outdims->idx[b_pos_start]);
			int * vecB_end = &(matrix->b_csc_outdims->idx[b_pos_end-1]);
			while ((*vecB_begin<lower_bound) & (vecB_begin<=vecB_end)) vecB_begin++;
			int vecB_offset = vecB_begin -  &(matrix->b_csc_outdims->idx[b_pos_start]);
			/* FWB: Until here!  ************************/

			// Find the intersection of an A_CSR and a B_CSC fiber
			std::vector<int> intersect_vector, intersect_overhead_vector, a_idx, b_idx;
			int len_B_fiber = 0;
			intersectTwoVectors(vecA_begin, vecA_end, vecB_begin, vecB_end,
					lower_bound, upper_bound, intersect_vector, intersect_overhead_vector,
					a_idx, b_idx, len_B_fiber);

			// There is no intersection, so no computation. Skip!
			if(intersect_vector.size() == 0){
				continue;
			}
			// Record the initial size of output before being touched!
			if((o_tile_intersect_count == 0)){
				if(params->getOFormat() == CSX::Dense){
					if((vecCommittedRows_ASize[i_idx]!=0) & (j_idx_start!=0)){
						OSize_preComp = params->getDenseTileSize();
					}
				}
				else{
					OSize_preComp = matrix->getCOOSize('O', i_idx, k_inner);
				}
			}
			// Just an indication that output tile is modified
			o_tile_intersect_count += intersect_vector.size();

			// Everything starts after m_build of vector a!
			//  The cost of m_build vector a is a_entries_count.at(t_idx)
			if(overheads_middle_perDOT[t_idx*par_middle] == 0){
				std::fill(&overheads_middle_perDOT[t_idx*par_middle],
					&overheads_middle_perDOT[(t_idx+1)*par_middle-1],
					a_entries_count.at(t_idx));
			}

			uint32_t overhead_middle_extract = 0;
			uint32_t overhead_middle_intersect = 0;
			// Middle level extract unit overhead
			CalculateMiddleLevelExtarctOverhead(a_entries_count.at(t_idx),
					len_B_fiber, intersect_vector.size(), overhead_middle_extract);
			// Middle level intersection unit overhead
			CalculateMiddleLevelIntersectOverhead(
					intersect_overhead_vector, overhead_middle_intersect);

			uint32_t * middle_operator = std::min_element(&overheads_middle_perDOT[t_idx*par_middle],
					&overheads_middle_perDOT[(t_idx+1)*par_middle-1]);
			*middle_operator += (overhead_middle_extract + overhead_middle_intersect);

			uint64_t top_middle_time_done = extractor_intersect_done_top[i_idx] +
				 uint64_t(*middle_operator);
/*
			uint64_t temp_pe = *std::min_element(pe_time, pe_time+params->getPECount());
			printf("pe min: %lu top level time: %lu, overhead_middle_acc:%d, overhead_middle_intersect:%d, overhead_middle_extract:%d\n",
					temp_pe, extractor_intersect_done_top[i_idx],
					overheads_middle_perDOT[t_idx], overhead_middle_intersect, overhead_middle_extract);
	*/
			int counter=0;
			for (std::vector<int>::iterator it = intersect_vector.begin();
					it != intersect_vector.end(); ++it, counter++){
				int j_inner = *it;

				/* DEBUGGING VARIABLES */
				a_update=0; b_update=0; o_r_update=0; o_w_update=0;

				action_bytes = 0; cycles_comp = 0;

				// Bring the A tile if it is not in the LLB already
				if(matrix->a_csr_outdims->vals[a_pos_start+vecA_offset+a_idx[counter]] == 0){
					matrix->a_csr_outdims->vals[a_pos_start+vecA_offset+a_idx[counter]] = 1;
					// Add A tiled size to the traffic computation
					if(params->getAFormat() == CSX::Dense){
						a_update = params->getDenseTileSize();
					}
					else{
						a_update = matrix->getCSFSize('A', i_idx, j_inner);
					}
					action_bytes += a_update;
					/* DEBUGGING CODE */
					//a_update = matrix->getCSFSize('A', i_idx, j_inner);
					a_traffic += a_update;

					// Middle DOT traffic
					middle_DOT_traffic += a_update;
				}
				// Bring the B tile if it is not in the LLB already
				if(matrix->b_csc_outdims->vals[b_pos_start+vecB_offset+b_idx[counter]] == 0){
					matrix->b_csc_outdims->vals[b_pos_start+vecB_offset+b_idx[counter]] = 1;
					// If need to load B tiles add them to the memory traffic usage
					if(params->getBFormat() == CSX::Dense){
						b_update = params->getDenseTileSize();
					}
					else{
						b_update =  matrix->getCSFSize('B', j_inner, k_inner);
					}
					action_bytes += b_update;
					/* DEBUGGING CODE */
					//b_update =  matrix->getCSFSize('B', j_inner, k_inner);
					b_traffic += b_update;
				}
				// Since we have A-stationay in middle DOT, B basic tiles will be streamed
				//	all again and again
				//b_update =matrix->getCSFSize('B', j_inner, k_inner);
				if(params->getBFormat() == CSX::Dense){
					middle_DOT_traffic += params->getDenseTileSize();
				}
				else{
					middle_DOT_traffic += matrix->getCSFSize('B', j_inner, k_inner);
				}

				// Do the actual calculation
				uint64_t bytes_rd = 0, bytes_wr =0; // Obsolete variables
				uint32_t pproduct_size= 0;
				if(params->getCompKernel() == kernel::SpMSpM){
					matrix->CSRTimesCSR(i_idx, j_inner, k_inner, &cycles_comp, &bytes_rd, &bytes_wr);
					// middle DOT traffic update for output partial products
					middle_DOT_traffic += bytes_wr;
				}
				else if(params->getCompKernel() == kernel::SpMM){
					matrix->CSRTimesDense(i_idx, j_inner, k_inner, &cycles_comp, pproduct_size);
					middle_DOT_traffic += pproduct_size;
				}
				// Add up the busy cycles; Used for starts and also sanity check
				stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);


				// Update bandwidth and update PE units when either
				//   we have loaded sth or did some computation
				if ((action_bytes) | (cycles_comp) | (middle_DOT_traffic)){
					// Normal case where one PE multiplication is distributed to only one
					//	intersect unit and PE unit
					if(params->getIntersectModelDist() == intersect_dist::sequential){
						uint64_t *target_pe_ptr;
						if(params->getStaticDistributorModelMiddle() == static_distributor::oracle){
						// Start and end cycle for currrent tile multiplication
						// Other than current PE time, take into account overheads!
							//uint64_t * min_pe_ptr = std::min_element(pe_time, pe_time+params->getPECount());
							target_pe_ptr = std::min_element(pe_time, pe_time+params->getPECount());
						}
						else if(params->getStaticDistributorModelMiddle() == static_distributor::round_robin){
							target_pe_ptr = &pe_time[round_robin_slot];
							round_robin_slot = (++round_robin_slot) % (params->getPECount());
						}
						else{
							printf("No such dealer is available!\n"); exit(1);
						}
						uint64_t starting_cycle = std::max(*target_pe_ptr ,	top_middle_time_done);
						uint64_t ending_cycle = (uint64_t)cycles_comp + starting_cycle;
						// Update the bandwidth usage. This is something I am very excited about, at the end
						// I can plot	this and look at per cycles bandwidth usage! I should do this for PEs as well
						uint64_t temp_finish_cycle_top = 0, temp_finish_cycle_middle = 0, temp_finish_cycle = 0;
						//excessCycles = updateBWLogAndReturnExcess(starting_cycle,
						//	ending_cycle, action_bytes, temp_finish_cycle);
						/* DEBUGGING CODE */
						uint64_t excessCycles_middle = 0;
//						uint64_t excessCycles_middle = updateBWLogAndReturnExcessMiddleDOT(starting_cycle,
//							ending_cycle, middle_DOT_traffic, temp_finish_cycle_middle);

/*						uint64_t excessCycles_top = updateBWLogAndReturnExcess(*target_pe_ptr, ending_cycle, action_bytes,
									temp_finish_cycle_top, a_update, b_update, o_r_update, o_w_update);
*/
						uint64_t excessCycles_top = updateBWLogAndReturnExcess(batch_starting_cycle, ending_cycle, action_bytes,
									temp_finish_cycle_top, a_update, b_update, o_r_update, o_w_update);


						ending_cycle += std::max(excessCycles_top, excessCycles_middle);

						*target_pe_ptr = ending_cycle;
						if(ending_cycle>batch_ending_cycle)
							batch_ending_cycle = ending_cycle;
						temp_finish_cycle = std::max(temp_finish_cycle_middle, temp_finish_cycle_top);
						if(temp_finish_cycle > mem_finish_cycle)
							mem_finish_cycle = temp_finish_cycle;
					}
					else{
						printf("You need to use either sequential or parallel distribution model!\n");
						exit(1);
					}
				} // End of if statement
			} // End of for loop on one A_CSR and one B_CSC fiber intersection

		} // End of t_idx loop

		// LLB size needs to be changed because the output size might have
		//   changed during the computation
		if(o_tile_intersect_count){
			if((params->getOFormat() == CSX::Dense)){
				if(vecCommittedRows_ASize[i_idx])
					OSize_postComp = params->getDenseTileSize();
			}
			else
				OSize_postComp = matrix->getCOOSize('O', i_idx, k_inner);
		}
		else
			OSize_postComp = 0;

		if(params->getCompKernel() == kernel::SpMM){
			llb->AddToLLB('O', Req::write, OSize_postComp-OSize_preComp, DONT_UPDATE_TRAFFIC);
			vecCommittedRows_OSize[i_idx] = OSize_postComp;
			/*vecCommittedRows_OSize[i_idx] = OSize_postComp;
			if(OSize_preComp != OSize_postComp)
				llb->AddToLLB('O', Req::write, OSize_postComp, DONT_UPDATE_TRAFFIC);
*/
		}
		else{
			vecCommittedRows_OSize[i_idx] += (OSize_postComp-OSize_preComp);
			llb->AddToLLB('O', Req::write, OSize_postComp-OSize_preComp, DONT_UPDATE_TRAFFIC);

		}
		// Update the bandwidth usage for the output psum read
		if(o_tile_intersect_count){
		//if(intersect_vector.size()){
			uint64_t temp_finish_cycle = 0;
			//excessCycles = updateBWLogAndReturnExcess(
			//	batch_ending_cycle, batch_ending_cycle, OSize_preComp, temp_finish_cycle);
			/*	DEBUGGING CODE */
			a_update=0; b_update=0; o_r_update=OSize_preComp; o_w_update=0;
/*			updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle,
			   OSize_preComp, temp_finish_cycle, a_update, b_update, o_r_update, o_w_update);
*/			updateBWLogAndReturnExcess(batch_starting_cycle, batch_ending_cycle,
			   OSize_preComp, temp_finish_cycle, a_update, b_update, o_r_update, o_w_update);

			o_traffic_read += OSize_preComp;
			if(temp_finish_cycle > mem_finish_cycle)
				mem_finish_cycle = temp_finish_cycle;
		}
	}

	// Use the bandwidth for writing back the output to memory
	uint64_t excessCycles = 0;
	if(vecCommittedRows_OSize[i_idx]>0){
		action_bytes = vecCommittedRows_OSize[i_idx];
		uint64_t temp_finish_cycle = 0;
		// Update the bandwidth usage for the output write-back
		//excessCycles = updateBWLogAndReturnExcess(
		//	batch_ending_cycle, batch_ending_cycle, action_bytes, temp_finish_cycle);
		/*	DEBUGGING CODE */
		a_update=0; b_update=0; o_r_update=0; o_w_update= vecCommittedRows_OSize[i_idx];
		uint64_t excessCycles = updateBWLogAndReturnExcess(batch_ending_cycle, batch_ending_cycle,
				action_bytes, temp_finish_cycle, a_update, b_update, o_r_update, o_w_update);
		o_traffic_write += action_bytes;
		if(temp_finish_cycle > mem_finish_cycle)
			mem_finish_cycle = temp_finish_cycle;
	}

	// Add the end_cycle, #row pair to the committed rows vector
	vecCommittedRows_iidx[num_vecCommittedRows] = i_idx;
	vecCommittedRows_cycle[num_vecCommittedRows] = batch_ending_cycle;
	num_vecCommittedRows++;

	uint64_t cycles_stats = std::max(*std::max_element(pe_time, pe_time+params->getPECount()),
		mem_finish_cycle); //	batch_ending_cycle + excessCycles);
	if(cycles_stats>stats->Get_cycles())
	{
		stats->Set_cycles(cycles_stats);
		stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());
	}

	delete [] overheads_middle_perDOT;
	return;
}

// Iterates over the A LLB tile row and divides the consecutive basic tiles
//   in a way that each group fits into the middle level buffer (PE buffer)
//   For example: tile 0..5 -> Buffer 0, tile 6..8-> Buffer 1, ...
// The functions gets the id of the row, the start and end J idx,
//   the A row vector. Produces all the j_idxs (starts of the groups)
//   and number of tiles in each group
void Scheduler_SpMM_7::findJIndexesInTheRange(int i_idx, int lower_bound, int upper_bound,
		int * vecA_begin, int * vecA_end,
		std::vector<int> & j_indexes, std::vector<int> & a_entries_count){
	// IF we have instant model for the middle level nothing is needed
	if((params->getSearchModelMiddle() == search_tiles::instant) |
			(params->getMetadataBuildModelMiddle() == metadata_build::instant)){
		j_indexes.push_back(lower_bound); j_indexes.push_back(upper_bound);
		a_entries_count.push_back(0);
		return;
	}
	// Middle buffer (PE buffer) size allocated to tensor A (config. value)
	int a_size_lim = (int)((float)params->getPEBufferSize() *
			params->getATensorPercPEBuffer());
	// Helper value to keep track of middle buffer already filled
	int size_buff_a = 0;
	// Helper value to keep track of number of A basic tiles already
	//   in the current middle unit (PE DOT unit)
	int count = 0;
	int total_tile_count = 0;
	// Starting idx
	j_indexes.push_back(*vecA_begin);
	for(int * it = vecA_begin; it<=vecA_end; it++){
		int j_idx = *it;
		//printf("j_idx: %d - [%d, %d]\n", j_idx, *vecA_begin, *vecA_end);
		// Early termination: All the tiles are included, get out of the for loop
		if( j_idx>=upper_bound ){
			// FIXME: next line looks irrelevant, I commented it out
			//a_entries_count.push_back(count);
			break;
		}
		total_tile_count++;
		// Next tile size
		int tile_size = 0;
		if(params->getAFormat() == CSX::Dense){
			tile_size = params->getDenseTileSize();
		}
		else{
			tile_size = matrix->getCSFSize('A', i_idx, j_idx);

		}
		// The tile does not fit, thus, the tile size is
		//	bigger than PE buffer size
		if(tile_size > a_size_lim){
			printf("Fatal Error: PE Buffer is smaller than the basic tile size!\n");
			exit(1);
		}

		// if fits in the buffer continue accumulating
		if((size_buff_a + tile_size) < a_size_lim){
			size_buff_a += tile_size;
			count++;
			//printf("j_idx: %d added, count: %d, size: %d\n", j_idx, count, size_buff_a);
		}
		// If it does not fit then record j and count values
		else{
			j_indexes.push_back(j_idx);
			a_entries_count.push_back(count);
			//printf("Full with size: %d\n", size_buff_a);
			size_buff_a = 0;	count = 0;
			// It does not fit! now fetch the did not fit ones
			count++; size_buff_a += tile_size;
			//printf("j_idx: %d added, count: %d, size: %d\n", j_idx, count, size_buff_a);
		}

	}
	if(count > 0){
		a_entries_count.push_back(count);
		j_indexes.push_back(upper_bound);
		//printf("Full with size: %d\n", size_buff_a);
	}
	// Commented this out and put a correct version in the above if statement
	//j_indexes.push_back(*vecA_end);

	// DEBUGGING code
/*
	printf("j_indexes size: %lu, a_entries_count size: %lu\n",
				j_indexes.size(), a_entries_count.size());

	for(int i=0; i<a_entries_count.size();i++)
	{
		printf("Start j: %d, End j: %d, Count: %d\n",
				j_indexes[i], j_indexes[i+1], a_entries_count[i]);
	}
	exit(1);
	*/
	return;
}


// Bandwidth logger for the middle DOT
uint64_t Scheduler_SpMM_7::updateBWLogAndReturnExcessMiddleDOT(uint64_t starting_cycle,
		uint64_t ending_cycle, uint64_t action_bytes,uint64_t & finish_cycle ){

	total_traffic_middle += action_bytes;

	float action_bytes_f = float(action_bytes);
	for(uint64_t i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		finish_cycle = i_idx;
		float rem_cap = middle_bytes_per_ns - middle_bw_logger[i_idx];
		if((action_bytes_f > 0) & (rem_cap == 0))
			continue;
		if(action_bytes_f > rem_cap){
			middle_bw_logger[i_idx] = middle_bytes_per_ns;
			action_bytes_f -= rem_cap;
		}
		else{
			middle_bw_logger[i_idx] += action_bytes_f;
			action_bytes_f = 0;

			if((uint64_t)i_idx > ending_cycle)
				return ((uint64_t)i_idx-ending_cycle);
			else
				return 0;
		}
	}

	printf("%d bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}


// This is a beautiful bw logger that gets the start cycle and end cycle
//   of each tile multiplication and in a cycle accurate way says how
//   many extra cycles it is going to take.
// Its role is to keep track of bandwidth
uint64_t Scheduler_SpMM_7::updateBWLogAndReturnExcess(uint64_t starting_cycle,
		uint64_t ending_cycle, uint64_t action_bytes,uint64_t & finish_cycle ){

	total_traffic += action_bytes;

	float action_bytes_f = float(action_bytes);
	for(uint64_t i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		finish_cycle = i_idx;
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
			// Shit this is not correct! what happens when i_idx<ending_cycle in uint64_t?
			//return std::max((uint64_t)0, i_idx - ending_cycle);
			if((uint64_t)i_idx > ending_cycle)
				return ((uint64_t)i_idx-ending_cycle);
			else
				return 0;
		}
	}
/*
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
		}
	}
	*/
	printf("%d bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}

uint64_t Scheduler_SpMM_7::updateBWLogAndReturnExcess(uint64_t starting_cycle,
		uint64_t ending_cycle, uint64_t action_bytes, uint64_t & finish_cycle,
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
	for(auto i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		finish_cycle = i_idx;
		float rem_cap = bytes_per_ns - bw_logger[i_idx];
		if((action_bytes_f != 0) & (rem_cap == 0))
			continue;
		if(action_bytes_f > rem_cap){
			bw_logger[i_idx] = bytes_per_ns;
			action_bytes_f -= rem_cap;
		}
		else{
			bw_logger[i_idx] += action_bytes_f;
			action_bytes_f = 0;
			// Shit this is not correct! what happens when i_idx<ending_cycle in uint64_t?
			//return std::max((uint64_t)0, i_idx - ending_cycle);
			// Correct version
			if((uint64_t)i_idx > ending_cycle)
				return ((uint64_t)i_idx-ending_cycle);
			else
				return 0;

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
void Scheduler_SpMM_7::SyncPETimesWithMinCycle(uint64_t min_val){
	for(int idx=0; idx<params->getPECount(); idx++){
		pe_time[idx] = std::max(pe_time[idx], min_val);
	}
	return;
}

// All of the PEs should finish their work before fetching the next	B tiles
void Scheduler_SpMM_7::SyncPETimes(){

	int max_val = *std::max_element(pe_time, pe_time+params->getPECount());
	std::fill(pe_time, pe_time+params->getPECount(), max_val);

	return;
}

// Gets the Column boundary and start address of matrix B row,
//   returns the start and stop address or the matrix B row, i.e., o_reuse
// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
void Scheduler_SpMM_7::CalcOReuse(int k_idx, int j_idx_start, int & j_idx_stop){

	int a_reuse = params->getAReuse();
	// Static tiling case (ExTensor); just fetch as many B basic tiles that are in the
	//   statically set region
	if(params->getTilingMechanism() == tiling::t_static){
		j_idx_stop = std::min(j_idx_start+J_top_tile, b_tiled_rows);
		uint64_t extra_size = AccumulateSize('B', j_idx_start, j_idx_stop,
				k_idx, std::min(k_idx+K_top_tile, o_tiled_cols), params->getBFormat());
		llb->AddToLLB('B', Req::read, extra_size, UPDATE_TRAFFIC);
	}
	// Dynamic reflexive tiling case (TACTile approach)
	else if(params->getTilingMechanism() == tiling::t_dynamic){
		j_idx_stop = j_idx_start;
		// Add rows until it either runs out of memory or reaches the last row
		for(int idx = j_idx_start; idx<b_tiled_rows; idx++){
			// Find the size of the new row
			uint64_t extra_size = AccumulateSize('B', idx, idx+1,
					k_idx, std::min(k_idx+a_reuse, o_tiled_cols), params->getBFormat());
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
						k_idx+a_reuse, k_idx+a_reuse+1, params->getBFormat());
				// It means that it could not fit the new row in LLB and failed
				if(llb->DoesFitInLLB('B', extra_size) == 0) {break;}
				llb->AddToLLB('B', Req::read, extra_size, UPDATE_TRAFFIC);
				a_reuse++;
			}
			params->setAReuse(a_reuse);
		}
		params->setOReuse(j_idx_stop- j_idx_start);
	}
	else{
		printf("Tiling choice is not available!\n");
		exit(1);
	}

	// Overhead for Dense data is zero! There is nothing
	//  unpredictable. We know how many tiles in what range to bring
	if(params->getBFormat() == CSX::Dense){
		overhead_tbuild_top_B = 0;
		overhead_extractor_top_B = 0;
		extractor_top_B_done = overhead_extractor_top_B +
			*std::min_element(pe_time, pe_time+params->getPECount());
		// This variable is used for scheduling tensor A tbuild phase
		//	(building the basic tiles for LLB)
		tbuild_top_B_done = overhead_tbuild_top_B +
			*std::min_element(pe_time, pe_time+params->getPECount());
	}
	else{
		/***************************************************************/
		// Calculate the basic tile build overhead top level (DRAM->LLB))
		// Important output : overhead_basic_tile_build
		uint64_t overhead_basic_tile_build = 0;
		// Instant and not tile build will take 0 cycles
		if((params->getBasicTileBuildModelTop() == basic_tile_build::instant)
				|(params->getBasicTileBuildModelTop() == basic_tile_build::noTilebuild))
			overhead_basic_tile_build = 0;
		// Parallel and serial cases
		else{
			int length_array = std::min(k_idx+a_reuse,o_tiled_cols) - k_idx;
			uint64_t * per_col_runtime =	new uint64_t[length_array];
			// Get the overhead of building basic tiles in each column
			for(int idx = k_idx; idx< k_idx+length_array; idx++){
				uint64_t nnz = AccumulateNNZ('B', j_idx_start, j_idx_stop, idx, idx+1);
				uint64_t nnr = AccumulateNNR('B', j_idx_start, j_idx_stop, idx, idx+1);
				per_col_runtime[idx-k_idx] = nnz+nnr;
				//per_col_runtime.push(nnz+nnr);
				overhead_basic_tile_build += (nnz+nnr);
			}
			// if we are doing it parallel then schedule it with
			//	column granularity
			if(params->getBasicTileBuildModelTop() == basic_tile_build::parallel){
				int par_factor = par_tbuild_B_top;
				int *sched_arr = new int[par_factor];
				std::fill(sched_arr, sched_arr+par_factor, 0);
				for (int p_idx = 0; p_idx < length_array; p_idx++){
					*std::min_element(sched_arr, sched_arr+par_factor) += per_col_runtime[p_idx];
				}
				overhead_basic_tile_build = *std::max_element(sched_arr,
						sched_arr+par_factor);
				delete [] sched_arr;
			}
			delete [] per_col_runtime;
		}

		/*  Calculate the search overhead top level (DRAM->LLB) */
		// Important output : overhead_search
		uint64_t overhead_search = 0;
		uint64_t serial_overhead_search = 0;
		uint64_t max_overhead_column = 0;
		// Instant and not tile build will take 0 cycles
		if(params->getSearchModelTop() == search_tiles::instant)
			overhead_search = 0;
		else{
			for(int idx = k_idx; idx< std::min(k_idx+a_reuse,o_tiled_cols); idx++){
				uint64_t overhead_column = AccumulateNNZTiles('B', j_idx_start, j_idx_stop, idx, idx+1);
				serial_overhead_search += overhead_column;
				max_overhead_column = std::max(max_overhead_column, overhead_column);
			}
			// Parallel case
			if(params->getSearchModelTop() == search_tiles::parallel){
				uint64_t temp = (serial_overhead_search%par_search_B_top == 0)?
					serial_overhead_search/ par_search_B_top:
					serial_overhead_search/ par_search_B_top + 1;
				overhead_search = temp + log2_search_B_top;
			}
			// Serial case
			else{
				overhead_search = serial_overhead_search;
			}
		}

		// Calculate the metadata build overhead of LLB
		// Important output : overhead_mbuild, overhead_posbuild
		uint64_t overhead_mbuild = 0; // idx of metadata
		uint64_t overhead_posbuild = 0; // pos of metadata
		// Instant will take 0 cycles
		if(params->getMetadataBuildModelTop() == metadata_build::instant)
			{overhead_mbuild = 0; overhead_posbuild = 0;}
		else if(params->getMetadataBuildModelTop() == metadata_build::serial){
			overhead_mbuild = serial_overhead_search;
			overhead_posbuild = 0;
		}
		else{
			overhead_mbuild = max_overhead_column;
			overhead_posbuild = uint64_t(log2_mbuild_B_top + 1);
		}

		// overhead = max(tbuild,search,mbuilt) + overhead_posbuild;
		overhead_tbuild_top_B = overhead_basic_tile_build;
		overhead_extractor_top_B = std::max(overhead_basic_tile_build,
				overhead_search);
		overhead_extractor_top_B = std::max(overhead_extractor_top_B, overhead_mbuild);
		overhead_extractor_top_B += overhead_posbuild;
		// This variable is used to know when extracting tensor B for LLB is over
		extractor_top_B_done = overhead_extractor_top_B +
			*std::min_element(pe_time, pe_time+params->getPECount());
		// This variable is used for scheduling tensor A tbuild phase
		//	(building the basic tiles for LLB)
		tbuild_top_B_done = overhead_tbuild_top_B +
			*std::min_element(pe_time, pe_time+params->getPECount());
		/*
		printf("B Overheads\n\tBasic tile build: %lu\n",overhead_basic_tile_build);
		printf("\tSearch: %lu\n",overhead_search);
		printf("\tMbuild: %lu\n",overhead_mbuild);
		printf("\tPosbuild: %lu\n",overhead_posbuild);
		printf("\tExtract %lu\n",overhead_extractor_top_B);
		printf("B extract finish time: %lu\n",extractor_top_B_done);
		*/
	}
	return;
}

// Find out what is the A and O row size we need to fetch
//	after LLB intersection
void Scheduler_SpMM_7::PreCalculateAORowsSize(int j_start, int j_end,
		int k_start, int k_end){

	/************************************/
	/* Pre-calculating A and O row size */
	int * b_llb_horizontalSum;
	if(params->getBFormat() != CSX::Dense)
	{
		b_llb_horizontalSum	= new int[b_tiled_rows];
		// Calculate the horizontal sum of B PE tiles in the LLB tile
		//   This sum is used to decide whether to bring PE tiles of A
		//   It basically says whether there is any nnz PE tile in each row
		CalcBLLBHorizontalSum(j_start, j_end,
			k_start, k_end, b_llb_horizontalSum);
	}
	#pragma omp parallel for
	for(int i_idx = 0; i_idx < o_tiled_rows; i_idx++){
		// Find the size of the A row size needs to be fetched
		//   taking into account whether B row is empty or not
		if(params->getBFormat() == CSX::Dense){
			vecCommittedRows_ASize[i_idx] = AccumulateSize('A', i_idx, i_idx+1,
				j_start, j_end, params->getAFormat());
		}
		else{
			vecCommittedRows_ASize[i_idx] = AccumulateSize_AwrtB(i_idx,
				j_start, j_end, b_llb_horizontalSum);
		}
		// Finds the size of O PE tiles wrt A & B tiles
		//  looks at the A&B intersection in the LLB to see whether
		//  they should be loaded or not
		if(params->getOFormat() == CSX::Dense){
			vecCommittedRows_OSize[i_idx] = AccumulateSize('O', i_idx, i_idx+1,
				k_start, k_end, params->getOFormat());
			// if A is empty then there will be no output change (intersection part), so do not read it
			// If data is clean don't read it
			if((vecCommittedRows_ASize[i_idx] == 0) | (j_start == 0))
				vecCommittedRows_OSize[i_idx] = 0;
		}
		else{
			vecCommittedRows_OSize[i_idx] = AccumulateSize_OwrtAB(i_idx,
				j_start, j_end, k_start, k_end);
		}
		//printf("%d --> %lu , %lu\n",i_idx, vecCommittedRows_ASize[i_idx], vecCommittedRows_OSize[i_idx]);
	}
	if(params->getBFormat() != CSX::Dense)
	{
		delete [] b_llb_horizontalSum;
	}

	/************************************************/
	/* Calculate A and O top level extract overhead */

	// Top level Tile Extract Overhead for A & O (DRAM->LLB))
	std::fill(vec_A_ExtractOverhead_top, vec_A_ExtractOverhead_top+o_tiled_rows, 0);
	std::fill(vec_O_ExtractOverhead_top, vec_O_ExtractOverhead_top+o_tiled_rows, 0);

	// Do not do the extraction part for ExTensor!
	//	Everything has been done statically
	if(params->getTilingMechanism() == tiling::t_static){
		return;
	}

	// Phase 1: Basic Tile Build (t build)
	if((params->getBasicTileBuildModelTop()==basic_tile_build::serial) |
			(params->getBasicTileBuildModelTop()==basic_tile_build::parallel)){

		#pragma omp parallel for
		for(int i_idx = 0; i_idx < o_tiled_rows; i_idx++){

			uint64_t nnz_a = AccumulateNNZ('A', i_idx, i_idx+1,	j_start, j_end);
			uint64_t nnr_a = AccumulateNNR('A', i_idx, i_idx+1,	j_start, j_end);
			uint64_t nnz_o = AccumulateNNZ('O', i_idx, i_idx+1,	k_start, k_end);
			uint64_t nnr_o = AccumulateNNR('O', i_idx, i_idx+1,	k_start, k_end);
			vec_A_ExtractOverhead_top[i_idx]= nnz_a+nnr_a;
			vec_O_ExtractOverhead_top[i_idx]= nnz_o+nnr_o;
		}
	}
	// Phase 2 & 3: Search & metadata build
	if((params->getSearchModelTop() != search_tiles::instant) |
			(params->getMetadataBuildModelTop() != metadata_build::instant)){

		#pragma omp parallel for
		for(int i_idx = 0; i_idx < o_tiled_rows; i_idx++){
			uint64_t overhead_search_A = 0, overhead_search_O = 0,
				overhead_mbuild_A = 0, overhead_mbuild_O = 0;
			uint64_t nnz_tiles_row_A = AccumulateNNZTiles('A', i_idx, i_idx+1,	j_start, j_end);
			uint64_t nnz_tiles_row_O = AccumulateNNZTiles('O', i_idx, i_idx+1,	k_start, k_end);
			// 0 overhead in instant mode
			if(params->getSearchModelTop() != search_tiles::instant){
				overhead_search_A = nnz_tiles_row_A;
				overhead_search_O = nnz_tiles_row_O;
			}
			// 0 overhead in instant mode
			if(params->getMetadataBuildModelTop() != metadata_build::instant){
				overhead_mbuild_A = nnz_tiles_row_A;
				overhead_mbuild_O = nnz_tiles_row_O;
			}
			// These include absolute overhead cycles; Later on I use parallelism
			//   and passed cycle to calculate the exact timing
			vec_A_ExtractOverhead_top[i_idx] = std::max(vec_A_ExtractOverhead_top[i_idx],
					std::max(overhead_search_A, overhead_mbuild_A));
			vec_O_ExtractOverhead_top[i_idx] = std::max(vec_O_ExtractOverhead_top[i_idx],
					std::max(overhead_search_O, overhead_mbuild_O));
			//printf("A[%d] Extract Overhead %lu\n",i_idx,vec_A_ExtractOverhead_top[i_idx]);
			//printf("O[%d] Extract Overhead %lu\n",i_idx,vec_O_ExtractOverhead_top[i_idx]);
		}
	}

	/************************************************/
	/* Calculate A and O top level extract overhead */
	if(params->getIntersectModelTop() == intersect::skipModel){
		// It all should happen instantaneously
		std::fill(intersect_overhead_top, intersect_overhead_top + o_tiled_rows, 0);

		/*

		// Intersect Vector B Start and End assignment -> top level (DRAM->LLB)
		int lower_bound = j_start; int upper_bound = j_end;
		int * vecB_begin = b_intersect_vector_top;
		int * vecB_end = &(b_intersect_vector_top[b_inter_count_top-1]);
		while ((*vecB_begin<lower_bound) & (vecB_begin<=vecB_end)) vecB_begin++;

		#pragma omp parallel for
		for(int i_idx = 0; i_idx < o_tiled_rows; i_idx++){
			// Intersect vector A start and assignment-> top level (DRAM->LLB)
			int a_pos_start = matrix->a_csr_outdims->pos[i_idx];
			int a_pos_end = matrix->a_csr_outdims->pos[i_idx+1];
			int * vecA_begin = &(matrix->a_csr_outdims->idx[a_pos_start]);
			int * vecA_end = &(matrix->a_csr_outdims->idx[a_pos_end-1]);
			while ((*vecA_begin<lower_bound) & (vecA_begin<=vecA_end)) vecA_begin++;

			intersect_overhead_top[i_idx] = getOverheadIntersectTwoVectors(
					vecA_begin, vecA_end, vecB_begin, vecB_end, lower_bound, upper_bound);
		}
		*/
	}
	return;
}


// It fetches A and O rows as much as LLB allows. It first evicts the row with
//   the smallest end cycle time and gives its space to new rows. The number of new rows
//   is returned in numRows (range [0,o_tiled_rows] ).
void Scheduler_SpMM_7::FetchAORows(int & numRows,
		int i_idx, int j_idx_start, int j_idx_stop, int k_idx){

	int a_reuse = params->getAReuse();
	uint64_t a_row_size, o_row_size;
	// This should normally free only one row; The reason I have used while is that
	//   in many cases the row size is zero (empty row), which we should skip!
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
		//FIXME: added for static tiling support
		if(params->getTilingMechanism() == tiling::t_static){
			remaining_row_slots++;
		}

		// If the size was greater than zero, then kudos! Get out of while loop
		if((a_row_size + o_row_size) > 0){
			// Remove the rows from LLB memory; A is flushed and O is written back
			llb->RemoveFromLLB('A', Req::read, a_row_size, DONT_UPDATE_TRAFFIC);
			llb->RemoveFromLLB('O', Req::write, o_row_size, UPDATE_TRAFFIC);
			SyncPETimesWithMinCycle(ending_cycle);

			break;
		}
	}

	/* Static tiling case */
	if(params->getTilingMechanism() == tiling::t_static){
		if(remaining_row_slots < 0){
			printf("Something has gone terribly wrong in static tiling\n");
			exit(1);
		}
		//int available_row_slots = total_row_slots - remaining_row_slots;
		int end_idx = first_row_notFetched + remaining_row_slots;
		for(int idx = first_row_notFetched; idx< std::min(end_idx, o_tiled_rows); idx++){
			remaining_row_slots--;
			a_row_size = vecCommittedRows_ASize[idx];
			o_row_size = vecCommittedRows_OSize[idx];
			llb->AddToLLB('A', Req::read, a_row_size, UPDATE_TRAFFIC);
			llb->AddToLLB('O', Req::read, o_row_size, UPDATE_TRAFFIC);
			first_row_notFetched = idx+1;

			uint64_t curr_pe = *std::min_element(pe_time, pe_time+params->getPECount());
			vec_A_ExtractOverhead_top[idx] += std::max(std::max(curr_pe, tbuild_top_B_done),
					*std::min_element(tbuild_A_time_top, tbuild_A_time_top+par_tbuild_A_top));
			vec_O_ExtractOverhead_top[idx] += std::max(curr_pe,
					*std::min_element(tbuild_O_time_top, tbuild_O_time_top+par_tbuild_O_top));

			*std::min_element(tbuild_A_time_top, tbuild_A_time_top+par_tbuild_A_top) =
				vec_A_ExtractOverhead_top[idx];
			*std::min_element(tbuild_O_time_top, tbuild_O_time_top+par_tbuild_O_top) =
				vec_O_ExtractOverhead_top[idx];

			if(params->getIntersectModelTop() == intersect::skipModel)
				extractor_intersect_done_top[idx] = findFinishTimeTopLevel(
						vec_A_ExtractOverhead_top[idx], vec_O_ExtractOverhead_top[idx],
						intersect_overhead_top[idx]);
			else{
				extractor_intersect_done_top[idx] = std::max(
						vec_A_ExtractOverhead_top[idx], vec_O_ExtractOverhead_top[idx]);
			}
		}
		// The number of A, O rows we have in LLB
		numRows = first_row_notFetched - i_idx;
	}

	else if(params->getTilingMechanism() == tiling::t_dynamic){
		numRows = 0;
		for (int idx = first_row_notFetched; idx < o_tiled_rows; idx++){
			// I modified the accumulate functions to take into account smart PE tile fetch
			//   They will only be brought into LLB if they will be used in any computation
			a_row_size = vecCommittedRows_ASize[idx];
			o_row_size = vecCommittedRows_OSize[idx];

			/*
			// DEBUGGING: Check if we are falling into any bug
			if(!llb->DoesFitInLLB('A', a_row_size))
				printf("A is full! %lu %lu %lu\n", llb->GetASize(), llb->GetBSize(), llb->GetOSize());
			if(!llb->DoesFitInLLB('O', o_row_size))
				printf("O is full! %lu %lu %lu\n", llb->GetASize(), llb->GetBSize(), llb->GetOSize());
			*/
			// If both the A and O rows fit in the LLB, then bring them in
			if ((llb->DoesFitInLLB('A', a_row_size)) & (llb->DoesFitInLLB('O', o_row_size))){
				llb->AddToLLB('A', Req::read, a_row_size, UPDATE_TRAFFIC);
				llb->AddToLLB('O', Req::read, o_row_size, UPDATE_TRAFFIC);

				first_row_notFetched = idx+1;
			}
			// We can not fit any more rows, return
			else{break;}

			// Find the first available operator for LLB row A & O to extract
			//  (tbuild, search, mbuild)
			//	Add them up with the delay it takes to build current rows
			//	Now, vec_A_ExtractOverhead_top[idx] and vec_O_ExtractOverhead_top
			uint64_t curr_pe = *std::min_element(pe_time, pe_time+params->getPECount());
			// It should be at least the time of current PE!
			//   Since we are allowed to fetch data at curr_PE time
			// Therefore, find out which operator is free from curr_time now on

			// Important: since A can only start after B is done (to have corret o_reuse)
			// std::max(curr_pe, tbuild_top_B_done) shows when it is possible to start
			//   building tiles according to PE time and building B elements
			// std::min_elemets(...,...) finds the first available tile builder to
			//   work on the row.
			// Whicever comes first should be added up to the actual overhead to give us
			//   the time
			// Please note this only happens once per each row section
			vec_A_ExtractOverhead_top[idx] += std::max(std::max(curr_pe, tbuild_top_B_done),
					*std::min_element(tbuild_A_time_top, tbuild_A_time_top+par_tbuild_A_top));
			vec_O_ExtractOverhead_top[idx] += std::max(curr_pe,
					*std::min_element(tbuild_O_time_top, tbuild_O_time_top+par_tbuild_O_top));

			// Update when each of the operators will be available for the next tile extracts
			*std::min_element(tbuild_A_time_top, tbuild_A_time_top+par_tbuild_A_top) =
				vec_A_ExtractOverhead_top[idx];
			*std::min_element(tbuild_O_time_top, tbuild_O_time_top+par_tbuild_O_top) =
				vec_O_ExtractOverhead_top[idx];

			// Produces the time that the top level modules are done
			//   DRAM->LLB extractor + intersection unit
			if(params->getIntersectModelTop() == intersect::skipModel)
				extractor_intersect_done_top[idx] = findFinishTimeTopLevel(
						vec_A_ExtractOverhead_top[idx], vec_O_ExtractOverhead_top[idx],
						intersect_overhead_top[idx]);
			else{
				extractor_intersect_done_top[idx] = std::max(
						vec_A_ExtractOverhead_top[idx], vec_O_ExtractOverhead_top[idx]);
			}
		}
		// The number of A, O rows we have in LLB
		numRows = first_row_notFetched - i_idx;
		//printf("numRows :%d ,%d, %lu\n",numRows, num_vecCommittedRows, llb->GetSize());
	}
	else{
		printf("Tiling choice is not available!\n");
		exit(1);
	}

	return;
}

// Uses a logger array for top level intersection unit. When
//	an LLB row of A & O are ready they are passed to the intersection unit
//	to calculate the ready time.
// This functions gets the time when extracting A & O rows are over and
//  the #cycles intersection would take. Then, the function checks the
//  array and finds the closest empty cycles.
uint64_t Scheduler_SpMM_7::findFinishTimeTopLevel(uint64_t vec_A_time,
		uint64_t vec_O_time, uint8_t intersect_overhead){

	uint64_t max_vec_times = std::max(vec_A_time, vec_O_time);
	uint8_t count_empty_cycles = 0;
	while(1){
		// if the intersect unit is not available, skip the current cycle
		if(intersect_logger_top[max_vec_times])
			max_vec_times++;
		else{
			// Check if all the cycles are available
			for(auto start= max_vec_times; start<max_vec_times+intersect_overhead; start++){
				// Not enough number of cycles is available
				if(intersect_logger_top[start]){
					max_vec_times = start+1;
					count_empty_cycles = 0;
					break;
				}
				else{count_empty_cycles++;}
			}
		}
		// Intersection unit was available during all the cycles
		if(count_empty_cycles == intersect_overhead){
			// Mark cycles as used
			std::fill(&intersect_logger_top[max_vec_times],
					&intersect_logger_top[max_vec_times+intersect_overhead], 1);
			break;
		}
	}
	return (max_vec_times + intersect_overhead);
}


// Find horizonatal sum of B tiles in LLB; this is used in
//   deciding whether to bring A PE tiles to LLB or not
// If horizontalsum of the row is zero then do not bring the
//   corresponding PE tile of A
void Scheduler_SpMM_7::CalcBLLBHorizontalSum(int j_start, int j_end,
		int k_start, int k_end, int * b_llb_horizontalSum){

	b_inter_count_top = 0;
	for(int j_idx = j_start; j_idx<j_end; j_idx++){
		if (AccumulateSize('B', j_idx, j_idx+1, k_start, k_end, params->getBFormat()) > 0){
			b_llb_horizontalSum[j_idx] = 1;
			if(params->getIntersectModelTop() == intersect::skipModel)
				b_intersect_vector_top[b_inter_count_top++] = j_idx;
		}
		else
			b_llb_horizontalSum[j_idx] = 0;
	}

	return;
}


// Finds the size of A PE tiles wrt B tiles
//  looks at the B tiles in the LLB to see whether
//  they should be loaded or not
/*
 * // Dense apparoach code! too slow
 * for(int j_idx=j_start; j_idx<j_end; j_idx++){
 *			if(b_llb_horizontalSum[j_idx]){
 *				size += matrix->getCSFSize('A', i_idx, j_idx);}}
 */
uint64_t Scheduler_SpMM_7::AccumulateSize_AwrtB(int i_idx,
		int j_start, int j_end, int *b_llb_horizontalSum){

	uint64_t size = 0;
	// Iterate over the column idxs of a specific row (e.g., i_idx)
	for(int t_idx = matrix->a_csr_outdims->pos[i_idx];
		 	t_idx < matrix->a_csr_outdims->pos[i_idx+1]; t_idx++){

		int j_idx = matrix->a_csr_outdims->idx[t_idx];
		// If the col idx is smaller than start skip
		if(j_idx<j_start) continue;
		// If the col idx matches the range, then add the size
		else if(j_idx<j_end){
			if(b_llb_horizontalSum[j_idx]){
				size += matrix->getCSFSize('A', i_idx, j_idx);}
		}
		// If the col idx is larger than the max, get out. You are done soldier Svejk
		else break;
	}
	return size;
}

// Finds the size of O PE tiles wrt A & B tiles
//  looks at the A&B intersection in the LLB to see whether
//  they should be loaded or not
uint64_t Scheduler_SpMM_7::AccumulateSize_OwrtAB(int i_idx,
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
uint32_t Scheduler_SpMM_7::ShouldIFetchThisOTile(int i_idx,
		int j_idx_start, int j_idx_end, int k_idx){

	// Small hack. If we are beinging from the begining of an A row, then
	//   size is zero and it does not matter
	if (j_idx_start == 0)
		return 0;
	int a_pos_start = matrix->a_csr_outdims->pos[i_idx];
	int a_pos_end = matrix->a_csr_outdims->pos[i_idx+1];
	int b_pos_start = matrix->b_csc_outdims->pos[k_idx];
	int b_pos_end = matrix->b_csc_outdims->pos[k_idx+1];
	if((a_pos_end>a_pos_start)&(b_pos_end>b_pos_start)){
		int match = 0;
		// find whether the intersection has any output
		match = intersectTwoVectors(&(matrix->a_csr_outdims->idx[a_pos_start]),
			&(matrix->a_csr_outdims->idx[a_pos_end-1]),
			&(matrix->b_csc_outdims->idx[b_pos_start]),
			&(matrix->b_csc_outdims->idx[b_pos_end-1]),
			j_idx_start, j_idx_end);
		if(match)
			return 1;
	}
	return 0;
}

// Gets two vectors and finds whether the intersection is empty or not
//   empty: 0
//   not empty: 1
int Scheduler_SpMM_7::getOverheadIntersectTwoVectors(int * vec1_begin, int * vec1_end,
		int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound){

	// In ideal case it takes 1 cycle
	if(params->getIntersectModelTop()==intersect::idealModel)
		return 1;

	// A flag that can be changed later
	//   It assumes once you found the intersect is not null report cycles
	//   and exit
	int EarlyIntersectTermination = 1;

	int max_val = 0; int cycle = 0;
	// Point to the start of the first vector where it is
	//   bigger or equal to the lower bound
	int *p1 = vec1_begin;
	// Point to the start of the second vector where it is
	//   bigger or equal to the lower bound
	int *p2 = vec2_begin;
	// Compare the two vectors. If they have a match return immediately.
	//   Otherwise, compare until reaching the end of both vectors
	while((p1 <= vec1_end) & (*p1<upper_bound) &
			(p2 <= vec2_end) & (*p2<upper_bound)){
		cycle++;
		max_val = std::max(*p1,*p2);
		if(*p1<max_val){
			while ((*p1<max_val) & (p1<=vec1_end) & (*p1<upper_bound)) {p1++;}
		}
		else if(*p2<max_val){
			while ((*p2<max_val) & (p2<=vec2_end) & (*p2<upper_bound)) {p2++;}
		}
		else{
			if(EarlyIntersectTermination)
				return cycle;
			p1++; p2++;
		}
	}
	return cycle;

}


// Gets two vectors and finds whether the intersection is empty or not
//   empty: 0
//   not empty: 1
int Scheduler_SpMM_7::intersectTwoVectors(int * vec1_begin, int * vec1_end,
		int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound){

	// Point to the start of the first vector where it is
	//   bigger or equal to the lower bound
	int *p1 = vec1_begin;
	//while ((*p1<lower_bound) & (p1<=vec1_end)) p1++;
	// Point to the start of the second vector where it is
	//   bigger or equal to the lower bound
	int *p2 = vec2_begin;
	//while ((*p2<lower_bound) & (p2<=vec2_end)) p2++;

	// Compare the two vectors. If they have a match return immediately.
	//   Otherwise, compare until reaching the end of both vectors
	while((p1 <= vec1_end) & (*p1<upper_bound) &
			(p2 <= vec2_end) & (*p2<upper_bound)){
		if(*p1 == *p2)
			return 1;
		if(*p1 < *p2)
			p1++;
		else
			p2++;
	}
	return 0;
}

// Gets two vectors and finds whether the intersection is empty or not
//   empty: 0
//   not empty: 1
void Scheduler_SpMM_7::intersectTwoVectors(int * vec1_begin, int * vec1_end,
		int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound,
		std::vector<int> & intersect_vector, std::vector<int> & overhead_vector,
		std::vector<int> & a_idx, std::vector<int> & b_idx,
		int & len_B_fiber){

	int overhead = 1; int max_val = 0; int cycle = 0;
	intersect intersect_middle = params->getIntersectModelMiddle();
	// Point to the start of the first vector where it is
	//   bigger or equal to the lower bound
	int *p1 = vec1_begin;
	// Point to the start of the second vector where it is
	//   bigger or equal to the lower bound
	int *p2 = vec2_begin;
	// Compare the two vectors. If they have a match return immediately.
	//   Otherwise, compare until reaching the end of both vectors
	while((p1 <= vec1_end) & (*p1<upper_bound) &
			(p2 <= vec2_end) & (*p2<upper_bound)){
		cycle++;
		max_val = std::max(*p1,*p2);
		if(*p1<max_val){
			while ((*p1<max_val) & (p1<=vec1_end) & (*p1<upper_bound)) {p1++;}
		}
		else if(*p2<max_val){
			while ((*p2<max_val) & (p2<=vec2_end) & (*p2<upper_bound)) {p2++;}
		}
		else{
			intersect_vector.push_back(*p1);
			a_idx.push_back(p1-vec1_begin);
			b_idx.push_back(p2-vec2_begin);
			p1++; p2++;
			if(intersect_middle == intersect::instantModel)
				overhead_vector.push_back(0);
			else if(intersect_middle == intersect::idealModel)
				overhead_vector.push_back(overhead++);
			else if(intersect_middle == intersect::skipModel)
				overhead_vector.push_back(cycle);
			else{
				printf("The intersection model is not available\n");
				exit(1);
			}
		}
	}

	if((params->getSearchModelMiddle() != search_tiles::instant) &
			(params->getMetadataBuildModelMiddle() != metadata_build::instant)){
		while ((p2<=vec2_end) & (*p2<upper_bound)) {p2++;}
		len_B_fiber = p2-vec2_begin;
	}
	return;
}

// Returns the cycles taken for Extract in the middle layer
//   via overhead_middle_extract variable
// NOTE: Please note that the mechanism highly depends on how the routing
//   to middle layer takes place too.
//   Currently it is assumed that the same middle (LLB->PE) DOT unit is responsible
//   for extracting the A row and corresponding B columns
//   (only entries between lower_bound and upper_bound for both vec A and vec B)
void Scheduler_SpMM_7::CalculateMiddleLevelExtarctOverhead(int a_vec_count, int b_vec_count,
		int num_effectaul_intersects, uint32_t & overhead_middle_extract){
	if((params->getSearchModelMiddle() != search_tiles::instant)
			& (params->getMetadataBuildModelMiddle()!= metadata_build::instant)){
		// Search should happen once and only for matrix A since it is the
		//   startionary one and B is streamed
		/*uint32_t overhead_search = (overhead_middle_extract) ? 0 : a_vec_count; */
		// If this is the first column of B then overhead is zero so far
		//   and find which vector takes longer to build
		/*uint32_t overhead_mbuild = (overhead_middle_extract) ? b_vec_count :
			std::max(a_vec_count, b_vec_count);
		uint32_t overhead = std::max(overhead_search, overhead_mbuild);
		*/
		// A is already build, we only need to build B and add the overhead
		uint32_t overhead_mbuild = b_vec_count;
		overhead_middle_extract = overhead_mbuild;
		/*overhead_middle_extract = overhead;*/
	}
	return;
}

// Returns the cycles taken for intersection in the middle layer
//   via overhead_middle_intersect variable
// NOTE: Please note that the mechanism highly depends on how the routing
//   to middle layer takes place too.
//   Currently it is assumed that the same middle (LLB->PE) DOT unit is responsible
//   for extracting the A row and corresponding B columns
//   (only entries between lower_bound and upper_bound for both vec A and vec B)
void Scheduler_SpMM_7::CalculateMiddleLevelIntersectOverhead(
		std::vector<int> intersect_overhead, uint32_t & overhead_middle_intersect){
	//if(params->getIntersectModelMiddle() != intersect::instantModel){
		// Take the last element! Shows when the intersection between two
		//   vectors is finished
		//overhead_middle_intersect = intersect_overhead.at(0);
	overhead_middle_intersect = intersect_overhead.at(intersect_overhead.size()-1);
	//}
	return;
}



// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the size that block of tiles would occupy in LLB
uint64_t Scheduler_SpMM_7::AccumulateSize(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end, CSX inp_format){

	uint64_t size_tot = 0;

	if(inp_format == CSX::Dense){
		size_tot = (d2_end - d2_start) * (d1_end - d1_start) * params->getDenseTileSize();
		return size_tot;
	}

	switch (mat_name){
		case 'A':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				size_tot += matrix->accumulateCSFSize_sparse('A',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		case 'B':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					size_tot += matrix->accumulateCSFSize_sparse('B',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					size_tot += matrix->accumulateCSFSize_sparse('B',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					size_tot += matrix->accumulateCSFSize_sparse('B',i_idx, d2_start, d2_end, CSX::CSR);
			}
			break;}
		case 'O':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				size_tot += matrix->accumulateCOOSize('O',i_idx, d2_start, d2_end);
			break;}
		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}

	return size_tot;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the total number of non-zero elements in that range
uint64_t Scheduler_SpMM_7::AccumulateNNZ(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t nnz_tot = 0;
	switch (mat_name){
		case 'A':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				nnz_tot += matrix->accumulateNNZ('A',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		case 'B':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					nnz_tot += matrix->accumulateNNZ('B',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					nnz_tot += matrix->accumulateNNZ('B',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					nnz_tot += matrix->accumulateNNZ('B',i_idx, d2_start, d2_end, CSX::CSR);
			}
			break;}
		case 'O':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				nnz_tot += matrix->accumulateNNZ('O',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}
	return nnz_tot;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the total number of non zero rows among all tiles of that range
uint64_t Scheduler_SpMM_7::AccumulateNNR(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t nnr_tot = 0;
	switch (mat_name){
		case 'A':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				nnr_tot += matrix->accumulateNNR('A',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		case 'B':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					nnr_tot += matrix->accumulateNNR('B',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					nnr_tot += matrix->accumulateNNR('B',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					nnr_tot += matrix->accumulateNNR('B',i_idx, d2_start, d2_end, CSX::CSR);
			}
			break;}
		case 'O':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				nnr_tot += matrix->accumulateNNR('O',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}
	return nnr_tot;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the total number of non zero tiles in that range
uint64_t Scheduler_SpMM_7::AccumulateNNZTiles(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t nnz_tot = 0;
	switch (mat_name){
		case 'A':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				nnz_tot += matrix->accumulateNNZTiles('A',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		case 'B':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					nnz_tot += matrix->accumulateNNZTiles('B',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					nnz_tot += matrix->accumulateNNZTiles('B',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					nnz_tot += matrix->accumulateNNZTiles('B',i_idx, d2_start, d2_end, CSX::CSR);
			}
			break;}
		case 'O':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
				nnz_tot += matrix->accumulateNNZTiles('O',i_idx, d2_start, d2_end, CSX::CSR);
			break;}
		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}
	return nnz_tot;
}



template<typename T>
void Scheduler_SpMM_7::Swap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

void Scheduler_SpMM_7::printPEs(){
	for(int i=0; i<params->getPECount(); i++)
		printf("%lu ", pe_time[i]);
	printf("\n");
	return;
}

void Scheduler_SpMM_7::PrintBWUsage(){
	//uint64_t size = 0;
	double size = 0;
	for (uint64_t i=0; i<= stats->Get_cycles(); i++)
		size += bw_logger[i];
	for(auto i=stats->Get_cycles()+1; i<stats->Get_cycles()+1000; i++){
		if(bw_logger[i] !=0) { printf("Shiiiit!\n"); break;}

	}
	printf("BW logger shows : %lu bytes, %f GBs\n", (uint64_t)size, (double)size/(1024.0*1024.0*1024.0));

	printf("BW logger a: %lu, b: %lu, o_r: %lu, o_w: %lu\n", a_bwl_traffic, b_bwl_traffic, o_bwl_traffic_read, o_bwl_traffic_write);

	printf("total_traffic %lu, a_read %lu, b read %lu, o_read %lu, o_write %lu\n",
			total_traffic, a_traffic, b_traffic, o_traffic_read, o_traffic_write);

	return;
}

void Scheduler_SpMM_7::PrintBWLog(){
	FILE * pFile;
	pFile = fopen ("bw_log.txt","w");
	for (uint64_t i=0; i< stats->Get_cycles(); i++)
		fprintf(pFile, "%f\n", bw_logger[i]);
	fclose(pFile);
	return;
}

#endif
