#ifndef SCHEDULER_8_CPP
#define SCHEDULER_8_CPP

#include "scheduler_8.h"

/* There are three DOT levels that instead of calling them DRAM, LLB, PE, Registers
 *   I have called them TOP (DRAM->LLB), MIDDLE (LLB->PE), BOTTOM (PE->Registers)
 *
 * Each level has its own modules
 *	Top level: Extract (TBuild, Search, Mbuild, Scanners), Intersection
 *	Middle Level: Extract (Search, MBuild, Scanners), Intersection
 *	Bottom Lebel: Extract (Scanners), Intersection unit
 */


// constructor -> intializer of the scheduler
Scheduler_8::Scheduler_8(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
	this->matrix = mtx;
	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Dimensions of inputs and the output tensors
	o_tiled_rows = matrix->getTiledORowSize();
	o_tiled_cols = matrix->getTiledOColSize();
	b_tiled_rows = matrix->getTiledBRowSize();

	// #Bytes that can be transferred every cycle
	top_bytes_per_ns = (float)params->getTopBandwidth() / params->getFrequency();
	middle_bytes_per_ns = (float)params->getMiddleBandwidth() / params->getFrequency();
	printf("Sanity check, middle bandwidth is now %f, %f\n", middle_bytes_per_ns, (float) params->getMiddleBandwidth());

	// BW logger to have bw usage in cycle level accuracy
	//	This is a cool approach.
	// The top variable is for the top level DRAM->LLB
	// The middle variable is for the middle level LLB->PE
	top_bw_logger = new float[MAX_TIME];
	middle_bw_logger = new float[MAX_TIME];

	// PE times
	pe_time = new uint64_t[params->getPECount()];
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
		nnz_counts = new uint64_t[params->getPECount()];
		std::fill(nnz_counts, nnz_counts+params->getPECount(), 0);
	}

	// Initialize PE arrays and BW_logger to all zeros
	std::fill(top_bw_logger, top_bw_logger+MAX_TIME, 0.0);
	std::fill(middle_bw_logger, middle_bw_logger+MAX_TIME, 0.0);

	// Each row that commit is recorded in these arrays
	//	They are mainly committed, but waiting to be retired
	//	to be evicted from the LLB memory finally
	vecCommittedRows_ASize = new uint64_t[o_tiled_rows];

	/* Top Level (DRAM -> LLB)
	 *	Extract and Intersect variables
	 */
	// Get the parallelism for builing basic tile in top level for tensors A, B, & O
	//	DRAM->LLB
	par_tbuild_A_top = params->getParallelismBasicTileBuildTop_A();
	par_tbuild_B_top = params->getParallelismBasicTileBuildTop_B();

	/*
	// A time-keeper for each basic tile builder in LLB of tensors A & O
	tbuild_A_time_top = new uint64_t[par_tbuild_A_top];
	// Set all the entries to 0
	std::fill(tbuild_A_time_top, tbuild_A_time_top +par_tbuild_A_top, 0);
	*/

	// Get the parallelism for search in LLB of tensor B
	//   Also its log2 value!
	par_search_top = params->getParallelismSearchTop();
	log2_search_top = 0;
	while (par_search_top >>= 1) ++log2_search_top;
	par_search_top = params->getParallelismSearchTop();

	int row_size = params->getTileSize();
	log2_mbuild_top = 0;
	while (row_size >>= 1) ++log2_mbuild_top;

	if(params->getSearchModelMiddle() != search_tiles::instant){
		par_middle = params->getParallelismMiddle();
	}
	else
		par_middle = 1;

	t_extract_middle_A = new uint64_t[par_middle];
	std::fill(t_extract_middle_A, t_extract_middle_A + par_middle, 0);

	// Used only for the round-robin scheduler
	round_robin_slot = 0;
	// DEBUGGING variables. DON'T REMOVE
	a_traffic = 0; b_traffic = 0; o_traffic_read = 0; o_traffic_write = 0; total_traffic = 0;
	a_bwl_traffic =0; b_bwl_traffic = 0; o_bwl_traffic_read = 0; o_bwl_traffic_write = 0;

	pe_utilization_logger = new uint64_t[params->getPECount()];
	std::fill(pe_utilization_logger, pe_utilization_logger+params->getPECount(), 0);

	return;
}

// Destructor -> delete [] bunch of dynamically allocated arrays
Scheduler_8::~Scheduler_8(){
	delete [] middle_bw_logger;
	delete [] top_bw_logger;
	delete [] pe_time;
	delete [] vecCommittedRows_ASize;
	delete [] t_extract_middle_A;
	delete [] pe_utilization_logger;
	//delete [] tbuild_A_time_top;
	if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
		delete [] nnz_counts;
	}

	return;
}

// Reset all the internal stats; Used when there are multiple runs in one main file
//	Usually used for bandwidth scaling sweep
void Scheduler_8::Reset(){

	llb_reset_count = 0;
	// Reset PE_time and BW_logger
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
		std::fill(nnz_counts, nnz_counts+params->getPECount(), 0);
	}
	std::fill(top_bw_logger, top_bw_logger+MAX_TIME, 0.0);
	std::fill(middle_bw_logger, middle_bw_logger+MAX_TIME, 0.0);
	// Sometimes bandwidth changes before reset, so update it
	top_bytes_per_ns = (float)params->getTopBandwidth() / params->getFrequency();
	middle_bytes_per_ns = (float)params->getMiddleBandwidth() / params->getFrequency();
	// Reset tp DOT basic tile build scheduler arrays
	std::fill(t_extract_middle_A, t_extract_middle_A +par_middle, 0);
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
 *	        there are some changes. However, they follow the same logic and order
 *
 * For row based synchronization we need two vectors
 * 1) A and O rows in LLB (every time we will add some to fill up the LLB portion)
 * 2) A and O row ID and finish time, so if it is evicted then the rest of the PEs
 *			added later should start not earlier than that time.
 *
 */
int Scheduler_8::Run(){

	// B_reuse -> I dimensioin
	// O_reuse -> J_dimension
	// a_reuse -> K_dimension
	//
	// If we have a static tiling mechanism (ExTensori style )
	//  then do some initialization
	if(params->getTilingMechanism() == tiling::t_static){
		// I, J, and K of the LLB (Top DOT) tiles
		I_top_tile = params->getITopTile();
		J_top_tile = params->getJTopTile();
		K_top_tile = params->getKTopTile();
		// O_reuse and a_reuse are constant values in static tiling
		// NOTE: AReuse should be interpreted as BReuse
		params->setAReuse(I_top_tile);
		params->setOReuse(J_top_tile);
		printf("I_top: %d, J_top: %d, K_top: %d\n", I_top_tile, J_top_tile, K_top_tile);
	}

	llb_reset_count = 0; round_robin_slot = 0;
	int a_reuse = params->getAReuse();
	a_reuse_const = a_reuse;

	// Iterations over B cols / O cols / a_reuse
	int k_end_top = 0;
	for(int k_start_top = 0; k_start_top < o_tiled_cols; /* EMPTY */){
        printf("Debug: top k %d of %d\n", k_start_top, o_tiled_cols); fflush(stdout);
		// Range for the a_reuse = [j_idx_start, j_idx_stop)
		int j_start_top = 0, j_end_top  = 0;

		totalSize_outputCOOLog = 0;
		// Iteration over B rows (in LLB tile granularity) / A cols / o_reuse
		while(j_start_top < b_tiled_rows){
			// Calculate O_reuse and fetch B tiles
			//  Calculate B_extract top level overhead as well
			ExtractBTopTile(k_start_top, j_start_top);
			// Load the latest updated a_reuse (it might be bigger than user defined)
			a_reuse = params->getAReuse();
			k_end_top = std::min(k_start_top+a_reuse, o_tiled_cols);
			j_end_top = std::min(j_start_top + params->getOReuse(), b_tiled_rows);

			// Find out what is the A row sizes we need to fetch
			//	after LLB intersection
			//	Also, calculates extract overhead for A rows
			PreCalculateARowsSize(j_start_top, j_end_top, k_start_top, k_end_top);

			//	Fetch B basic tiles in tile extraction
			EarlyFetchBBasicTiles(j_start_top, j_end_top, k_start_top, k_end_top);
			// iteration over A rows / O rows
			//   there is no i_idx++ since it goes as far as LLB assists
			int i_end_top = 0;
			for(int i_start_top = 0; i_start_top < o_tiled_rows;/* EMPTY */){

				// j_end_top and k end_top are already calculated,
				//	now need to figure out i_end
				ExtractATopTile(i_start_top, i_end_top, j_start_top, j_end_top, k_start_top, k_end_top);

				// Fetch A basic tiles! in basic tile build of tile extraction
				EarlyFetchABasicTiles(i_start_top, i_end_top, j_start_top, j_end_top);

				// Schedule the multiplication of basic tiles of an A top tile (LLB)
				//	by basic tiles of a top tile of B
				ScheduleMiddleDOT(i_start_top, i_end_top,
						j_start_top, j_end_top, k_start_top, k_end_top);

				// In static tiling the sizes should be accurate and we should
				//	never face oversized
				if(llb->GetSize() > llb->GetCapacity()){
					printf("LLB Size is not Enough!\
							(This message should be shown only in static tiling)\n");
					exit(1);
				}

				i_start_top = i_end_top;

				llb->EvictMatrixFromLLB('A', DONT_UPDATE_TRAFFIC);
				std::fill(matrix->a_csc_outdims->vals,
					matrix->a_csc_outdims->vals + matrix->a_csc_outdims->nnz, 0.0);
			}

			// Update B LLB col value
			j_start_top = j_end_top;
			// Flush everything inside LLB since we are going to bring new data
			// None of them going to write-back data except for O!
			//	However, it is not updating the traffic since it is being
			//	updated when the last row of B is reached!
			llb->EvictMatrixFromLLB('B', DONT_UPDATE_TRAFFIC);
			llb->EvictMatrixFromLLB('O', DONT_UPDATE_TRAFFIC);
			// Mark all A and B tiles as not present in LLB
			//	When 0, the tile is not in LLB
			//	When 1, the tile is present in LLB
			std::fill(matrix->b_csr_outdims->vals,
					matrix->b_csr_outdims->vals + matrix->b_csr_outdims->nnz, 0.0);

		}

		uint64_t topLevel_output_bw = AccumulateSize('O', 0, o_tiled_rows,
				k_start_top, k_end_top, CSX::CSF);

		// Update output write bytes!
		llb->AddToLLB('O', Req::write, topLevel_output_bw, UPDATE_TRAFFIC);
		llb->EvictMatrixFromLLB('O', DONT_UPDATE_TRAFFIC);
		// Top DOT NoC traffic stats update
		//stats->Accumulate_o_write(topLevel_output_bw);

		// This means we have to take into account COO partial product logs
		topLevel_output_bw += totalSize_outputCOOLog;
		// Update output (log file) read bytes
		llb->AddToLLB('O', Req::read, totalSize_outputCOOLog, UPDATE_TRAFFIC);
		llb->EvictMatrixFromLLB('O', DONT_UPDATE_TRAFFIC);
		// Top DOT NoC traffic stats update
		//stats->Accumulate_o_read(totalSize_outputCOOLog);

		uint64_t starting_cycle = *std::min_element(pe_time, pe_time + params->getPECount());
		uint64_t ending_cycle = *std::max_element(pe_time, pe_time + params->getPECount());
		uint64_t endingCycle_memory = updateBWLog(starting_cycle,
				topLevel_output_bw,	top_bw_logger, top_bytes_per_ns);

		stats->Set_cycles(std::max(ending_cycle, endingCycle_memory));
		stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());

		std::fill(pe_time, pe_time+params->getPECount(), stats->Get_cycles());

		// Outermost for loop counter
		k_start_top = k_end_top;
		// B LLB column is changed, load back the user defined a_reuse val
		params->setAReuse(a_reuse_const);
	}

	return 0;
}


// Gets a fiber of A and O and a rectangle LLB tile of B and schedules them
//   The ending cycles of each row is recorded in vecCommittedRows for
//   synnchronization, fetching, and committing rows
void Scheduler_8::ScheduleMiddleDOT(int i_start_top, int i_end_top,
		int j_start_top, int j_end_top, int k_start_top, int k_end_top){

	uint64_t max_time_accessed_in_batch = 0;
	uint64_t min_time_pe_progressed = 0;
	// Only used for oracle_relaxed static distributor
	uint32_t total_top_traffic = 0,	total_middle_traffic = 0;

	uint64_t LogWriteBackSize= 0;
	//printf("i_top:[%d:%d), j_top: [%d:%d), k_top: [%d:%d)\n",
	//	i_start_top, i_end_top, j_start_top, j_end_top, k_start_top, k_end_top);
	uint64_t batch_starting_cycle =
			*std::min_element(pe_time, pe_time+params->getPECount());

	int output_nnz = 0;

	// Initilizes the log matrix for output -> All empty tiles
	matrix->initOutputLogMatrix(i_start_top, i_end_top, k_start_top, k_end_top);

	for(int j_index_middle = j_start_top; j_index_middle < j_end_top; j_index_middle++){
		//printf("j_index: %d\n", j_index_middle);
		std::vector<int> a_indices_middle, a_indices_count_pos,
			a_offset_idx, a_search_overhead;
		std::vector<int> b_indices_middle, b_offset_idx;
		std::vector<uint64_t> top_middle_done_time;
		std::vector<int> pe_indices;

		// Extract A Middle Tiles
		ExtractAMiddleTiles(i_start_top, i_end_top, j_index_middle,
				a_indices_middle, a_indices_count_pos, a_offset_idx, a_search_overhead);

		// Extract B middle Tiles
		ExtractBMiddleTiles(j_index_middle, k_start_top, k_end_top,
					b_indices_middle, b_offset_idx);

		// Calculates the earliest start time after taking into account
		//	current PE time, top overhead, and middle overhead
		CalcTopMiddleDoneTime(std::max(batch_starting_cycle, extractor_top_A_done)
				, a_search_overhead, top_middle_done_time);

		int a_groups = a_indices_count_pos.size() - 1;
		int fib_b_len = b_indices_middle.size();

		// Early termination
		if((a_groups == 0) | (fib_b_len == 0))
			continue;

		// Debugged! working correctly
		PickPEsAccordingToPolicy(a_groups, pe_indices);

		// Data structures for handling top, middle, and bottom bandwidth calculations
		uint32_t ** comp_times = new uint32_t*[a_groups];
		// DRAM->LLB traffic
		uint32_t ** top_level_traffics = new uint32_t*[a_groups];
		// LLB->PEs traffic; currently, top and middle traffics shdould look alike
		//	However, I am generalizing the solution to be able to derive more variations later
		uint32_t ** middle_level_traffics = new uint32_t*[a_groups];

		// It is shared among all the PE units
		uint32_t * top_level_traffics_outputlog = new uint32_t[fib_b_len];

		for(int i = 0; i< a_groups; i++){
			comp_times[i] = new uint32_t[fib_b_len];
			top_level_traffics[i] = new uint32_t[fib_b_len];
			middle_level_traffics[i] = new uint32_t[fib_b_len];
			std::fill(comp_times[i], comp_times[i] + fib_b_len, 0);
			std::fill(top_level_traffics[i], top_level_traffics[i] + fib_b_len, 0);
			std::fill(middle_level_traffics[i], middle_level_traffics[i] + fib_b_len, 0);
		}
		std::fill(top_level_traffics_outputlog, top_level_traffics_outputlog + fib_b_len, 0);

		uint64_t ** curr_times = new uint64_t*[a_groups];
		for(int i = 0; i< a_groups; i++){
			curr_times[i] = new uint64_t[fib_b_len];
			std::fill(curr_times[i], curr_times[i] + fib_b_len, 0);
		}
		uint64_t *max_times = new uint64_t[fib_b_len];
		// max_times initialization
		std::fill(max_times, max_times + fib_b_len, 0);


		std::vector<uint32_t> b_ranges_start;
		std::vector<uint32_t> b_ranges_end;

		FindBBasicTileRanges(j_index_middle, k_start_top, k_end_top,
				b_ranges_start, b_ranges_end);

		for(int pe_idx = 0; pe_idx < a_groups; pe_idx++){
			for(int a_idx = a_indices_count_pos.at(pe_idx);
					a_idx < a_indices_count_pos.at(pe_idx+1); a_idx++){
				for(int b_idx = 0; b_idx < fib_b_len; b_idx++){
					int i_index_middle = a_indices_middle.at(a_idx);
					int k_index_middle = b_indices_middle.at(b_idx);

					uint32_t pproduct_size = 0; int cycles_comp = 0;
					uint32_t top_traffic = 0, middle_traffic = 0;
					uint64_t LogWriteBackSize_ct= 0;

					if(matrix->a_csc_outdims->vals[a_offset_idx.at(a_idx)] == 0.0){
						matrix->a_csc_outdims->vals[a_offset_idx.at(a_idx)] = 1.0;
						// Add A tiled size to the traffic computation
						uint32_t a_tile_size = matrix->getCSFSize('A', i_index_middle, j_index_middle);
						top_traffic += a_tile_size; middle_traffic += a_tile_size;
						// Stats update for top and middle NoC traffic
						stats->Accumulate_a_read_middle((uint64_t)a_tile_size);
					}

					// Bring the B tile if it is not in the LLB already
					if(matrix->b_csr_outdims->vals[b_offset_idx.at(b_idx)] == 0.0){
						matrix->b_csr_outdims->vals[b_offset_idx.at(b_idx)] = 1.0;
						// If need to load B tiles add them to the memory traffic usage
						uint32_t b_tile_size = matrix->getCSFSize('B', j_index_middle, k_index_middle);
						top_traffic += b_tile_size;
						//stats->Accumulate_b_read((uint64_t)b_tile_size);
					}

					// Middle traffic for B basic tiles
					// The unoptimized outer product(what we expect in inner product) will fetch
					//   B basic tiles separately for each computation
					// For the optimized outer product it will need to fetch it only once for
					//   each A and B fiber of the same LLB batch (i.e., fetch it again for the next LLB
					//   multiplication)
					if(params->getMiddleDOTDataflow() == arch::innerProdMiddle){
						auto b_tile_size = matrix->getCSFSize('B', j_index_middle, k_index_middle);
						middle_traffic += (uint32_t) b_tile_size;
						stats->Accumulate_b_read_middle((uint64_t)b_tile_size);
					}
					else if(params->getMiddleDOTDataflow() == arch::outerProdMiddle){
						if((pe_idx==0) & (a_idx==0)){
							auto b_tile_size = matrix->getCSFSize('B', j_index_middle, k_index_middle);
							middle_traffic += (uint32_t) b_tile_size;
							stats->Accumulate_b_read_middle((uint64_t)b_tile_size);
						}
					}

					// Do the actual calculation
					matrix->CSRTimesCSR(i_index_middle, j_index_middle,
							k_index_middle,	&cycles_comp, pproduct_size);
					//printf("Comp-> (%d, %d, %d): %d\n",
					//		i_index_middle, j_index_middle, k_index_middle, cycles_comp);

					// Add up the busy cycles; Used for starts and also sanity check
					stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);
					// Get the size of output partial products in the LLB memory
					//	If they are more than the limit then write-back logs to main memory
					uint64_t output_size = matrix->getOutputLogNNZCOOSize();
					// Partial products that are produced in the bottom DOT
					middle_traffic += pproduct_size;
					stats->Accumulate_o_write_middle((uint64_t)pproduct_size);

					if(!llb->DoesFitInLLB('O', output_size)){
						/*
						 * Code for partial eviction! Nice idea but did not work out honestly and
						 * full eviction result looked more promising
						 * Maybe the algorithm was not just good! who knows?*/
						/*
						if(j_end_top == b_tiled_rows){
							uint64_t evict_size =
								matrix->partiallyEvictOutputLogMatrix(0.5, llb->GetMaxOSize());
							llb->AddToLLB('O', Req::write, evict_size, DONT_UPDATE_TRAFFIC);
							llb->EvictMatrixFromLLB('O', UPDATE_TRAFFIC);
							LogWriteBackSize += evict_size;

							top_level_traffics_outputlog[b_idx] += evict_size;
						}
						else{
						*/
						// first write back ouput to the DRAM then add the outsanding memory write
						llb->AddToLLB('O', Req::write, output_size, DONT_UPDATE_TRAFFIC);
						llb->EvictMatrixFromLLB('O', UPDATE_TRAFFIC);
						// Top DOT NoC traffic stats update
						//stats->Accumulate_o_write(output_size);
						// update the total # of non-zeros for adaptive LLB partitioning
						output_nnz += matrix->getOutputLogNNZCount();
						// Evict all the tiles available in top memory (LLB)
						//  Let's do an experiment! I will evict it partially (up to a threshold)
						matrix->evictOutputLogMatrix();
						// Updating the size of write-back for pproducts
						//	This value should be read-back into top memory (LLB) later for merge
						LogWriteBackSize += output_size;
						LogWriteBackSize_ct += output_size;
						top_level_traffics_outputlog[b_idx] += output_size;
						//}
					}
					if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
						nnz_counts[pe_indices.at(pe_idx)] +=
							(matrix->getNNZOfATile('A', i_index_middle, j_index_middle) +
							 matrix->getNNZOfATile('B', j_index_middle, k_index_middle));
					}
					else if(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed){
						// PE utilization logger update
						int pe_id = std::distance(pe_time, std::min_element(pe_time, pe_time+params->getPECount()));
						pe_utilization_logger[pe_id] += (uint64_t) cycles_comp;
						// Add cycles to the PE_time and update top and middle traffic
						*std::min_element(pe_time, pe_time+params->getPECount()) += (uint64_t)cycles_comp;
						total_top_traffic += top_traffic;
						total_middle_traffic += middle_traffic;
					}
					else if(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed_ct){
						uint64_t * chosen_pe = std::min_element(pe_time, pe_time+params->getPECount());
						// The starting time should be batch_starting_cycle because
						//  pre-fetching can happen
						uint64_t endingCycle_top = updateBWLog(batch_starting_cycle,
								top_traffic, top_bw_logger, top_bytes_per_ns);
						// For the middle DOT we should start when computation takes place
						uint64_t endingCycle_middle =0;
						if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
							endingCycle_middle = updateBWLog(*chosen_pe,
								middle_traffic, middle_bw_logger, middle_bytes_per_ns);
						}
						*chosen_pe = std::max( std::max(endingCycle_top, endingCycle_middle)
									,*chosen_pe + (uint64_t) cycles_comp);

						// Take into account prematurely writing back the output log
						if(LogWriteBackSize_ct){
							endingCycle_top = updateBWLog(*chosen_pe,
								LogWriteBackSize_ct, top_bw_logger, top_bytes_per_ns);
						}
						uint64_t max_time_accessed_in_batch = std::max(max_time_accessed_in_batch,
								std::max(*chosen_pe, endingCycle_top));
						// PE utilization logger update
						int pe_id = std::distance(pe_time, chosen_pe);
						pe_utilization_logger[pe_id] += (uint64_t) cycles_comp;
					}

					comp_times[pe_idx][b_idx] += (uint32_t) cycles_comp;
					top_level_traffics[pe_idx][b_idx] += (uint32_t) top_traffic;
					middle_level_traffics[pe_idx][b_idx] += (uint32_t) middle_traffic;

					// PE utilization logger update
					if((params->getStaticDistributorModelMiddle() != static_distributor::oracle_relaxed_ct)
						& (params->getStaticDistributorModelMiddle() != static_distributor::oracle_relaxed))
						pe_utilization_logger[pe_indices.at(pe_idx)] += (uint64_t) cycles_comp;

				} // for b_idx
			} // for a_idx
		} // for pe_idx

		if((params->getStaticDistributorModelMiddle() != static_distributor::oracle_relaxed)
				& (params->getStaticDistributorModelMiddle() != static_distributor::oracle_relaxed_ct)){
			max_times[0] = 0;
			for(int i = 0; i< a_groups; i++){
				max_times[0] = std::max(max_times[0], pe_time[pe_indices.at(i)]);
			}
			for(int i = 0; i< a_groups; i++){
				curr_times[i][0] = max_times[0];
			}
			max_time_accessed_in_batch = std::max(max_time_accessed_in_batch, max_times[0]);

			int curr_b_range_group = 0, b_group_changed = 0;
			for(int b_index_bottom = 0; b_index_bottom < fib_b_len; b_index_bottom++){
				for(int pe_index_bottom = 0; pe_index_bottom < a_groups; pe_index_bottom++){
					//uint64_t starting_cycle = batch_starting_cycle;
					//	Add it to comp_time since computation cannot start before they are settled
					if(b_index_bottom == 0){
						curr_times[pe_index_bottom][b_index_bottom] = std::max(
							curr_times[pe_index_bottom][b_index_bottom], top_middle_done_time.at(pe_index_bottom));
					}
					else{
						curr_times[pe_index_bottom][b_index_bottom]= std::max(
							curr_times[pe_index_bottom][b_index_bottom-1], top_middle_done_time.at(pe_index_bottom));
					}

					uint64_t starting_cycle_middle = curr_times[pe_index_bottom][b_index_bottom];
					uint64_t starting_cycle_top = batch_starting_cycle;

					if(b_group_changed){
						curr_times[pe_index_bottom][b_index_bottom] =
							std::max(curr_times[pe_index_bottom][b_index_bottom], max_times[curr_b_range_group-1])
							+ (uint64_t)comp_times[pe_index_bottom][b_index_bottom];
					}
					else{
						curr_times[pe_index_bottom][b_index_bottom] +=
							(uint64_t)comp_times[pe_index_bottom][b_index_bottom];
					}

					uint64_t endingCycle_top = updateBWLog(//starting_cycle,
							starting_cycle_top,
							top_level_traffics[pe_index_bottom][b_index_bottom],
							top_bw_logger, top_bytes_per_ns);
					uint64_t endingCycle_middle =0;
					if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
						endingCycle_middle = updateBWLog(//starting_cycle,
							starting_cycle_middle,
							middle_level_traffics[pe_index_bottom][b_index_bottom],
							middle_bw_logger, middle_bytes_per_ns);
					}

					curr_times[pe_index_bottom][b_index_bottom]= std::max(
							curr_times[pe_index_bottom][b_index_bottom],
							std::max(endingCycle_top, endingCycle_middle));

					max_time_accessed_in_batch = std::max(
							max_time_accessed_in_batch, curr_times[pe_index_bottom][b_index_bottom]);

				} // for pe_index_bottom

				// Update max_times of the curr_b_range_group
				b_group_changed = 0;
				// Next b_index is bigger than the last basic tile exists in this batch
				while((b_index_bottom + 1) > ((int)b_ranges_end[curr_b_range_group]-1)){
					uint64_t max_val = 0;
					for(int t_idx = 0; t_idx< a_groups; t_idx++)
						max_val = std::max(max_val, curr_times[t_idx][curr_b_range_group]);
					max_times[curr_b_range_group] = max_val;

					// Update the next b_range_group and raise the flag for change
					curr_b_range_group++;
					b_group_changed = 1;
				}

				// Output log partial write-back
				if(top_level_traffics_outputlog[b_index_bottom]){
					uint64_t outputlog_flush_time = 0;
					for(int t_idx = 0; t_idx< a_groups; t_idx++)
						outputlog_flush_time = std::max(outputlog_flush_time,
								curr_times[t_idx][b_index_bottom]);
					// Ignore the output since this is not going to affect PE execution times
					//   It is just affecting the top level (LLB->DRAM) traffic.
					uint64_t endingCycle_top = updateBWLog(outputlog_flush_time,
							top_level_traffics_outputlog[b_index_bottom],
							top_bw_logger, top_bytes_per_ns);

					max_time_accessed_in_batch = std::max(
							max_time_accessed_in_batch, endingCycle_top);
				}// If
			} // for b_index_bottom

			// update pe_time when everything is done!
			uint64_t * temp_pe_time = new uint64_t[a_groups];
			for(int i = 0; i < a_groups; i++){
				temp_pe_time[i] = curr_times[i][fib_b_len-1];
			}
			std::sort(temp_pe_time, temp_pe_time + a_groups);
			min_time_pe_progressed = temp_pe_time[0];
			for(int i = 0; i < a_groups; i++){
				//pe_time[pe_indices.at(i)] = temp_pe_time[a_groups-i-1];
				pe_time[pe_indices.at(i)] = temp_pe_time[a_groups-i-1];
			}
			delete [] temp_pe_time;
		}// end if

		// De-allocate all the dynamically allocated datastructure for the
		//	specific column of A top tile and row of B top tile
		delete [] max_times;
		for(int i = 0; i< a_groups; i++){
			delete [] curr_times[i];
			delete [] middle_level_traffics[i];
			delete [] top_level_traffics[i];
			delete [] comp_times[i];
		}
		delete [] curr_times;
		delete [] middle_level_traffics;
		delete [] top_level_traffics;
		delete [] comp_times;

		delete [] top_level_traffics_outputlog;

		//printPEs();

	} // for j_index_middle
	/****** Loop back to the next column of A and row of B *******/

	if(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed){
		uint64_t endingCycle_top = updateBWLog(batch_starting_cycle,
				total_top_traffic, top_bw_logger, top_bytes_per_ns);

		uint64_t endingCycle_middle =0;
		if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
			endingCycle_middle = updateBWLog(batch_starting_cycle,
				total_middle_traffic,	middle_bw_logger, middle_bytes_per_ns);
		}
		uint64_t max_traffic_time = std::max(endingCycle_top, endingCycle_middle);
		for(int pe_idx = 0; pe_idx < params->getPECount(); pe_idx++){
			if(pe_time[pe_idx] < max_traffic_time)
				pe_time[pe_idx] = max_traffic_time;
		}

		max_time_accessed_in_batch =
			std::max(max_time_accessed_in_batch, max_traffic_time);
	}

	// In case we have not reached the last row of B (j_end)
	//	we need to write back the last batch of log to main memory
	if(j_end_top != b_tiled_rows){
		uint64_t writeback_log_size_last = matrix->getOutputLogNNZCOOSize();
		if(writeback_log_size_last){
			LogWriteBackSize += writeback_log_size_last;
			// Update the stats
			llb->AddToLLB('O', Req::write, writeback_log_size_last, DONT_UPDATE_TRAFFIC);
			llb->EvictMatrixFromLLB('O', UPDATE_TRAFFIC);

			uint64_t max_pe_time = *std::max_element(pe_time, pe_time+params->getPECount());
			uint64_t endingCycle_memory = updateBWLog(max_pe_time,
				writeback_log_size_last, top_bw_logger, top_bytes_per_ns);

			max_time_accessed_in_batch = std::max(
					max_time_accessed_in_batch, endingCycle_memory);
		}

	}

	stats->Set_cycles(std::max(stats->Get_cycles(), max_time_accessed_in_batch));
	stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());

	for(int pe_idx = 0; pe_idx<params->getPECount(); pe_idx++){
		if(pe_time[pe_idx] < min_time_pe_progressed)
			pe_time[pe_idx] = min_time_pe_progressed;
	}

	// update the total # of non-zeros for adaptive LLB partitioning
	// !It could had been calculated more accurately but this is good enough!
	output_nnz += matrix->getOutputLogNNZCount();
	//printf("Final size: %lu\n",matrix->getOutputLogNNZCOOSize());

	// Delete the Log matrix created in top memory
	matrix->deleteOutputLogMatrix();

	// The amount of data that will be read-back to top memory later
	totalSize_outputCOOLog += LogWriteBackSize;

	// Adaptively update the top memory (LLB) partitioning A to O ratio
//	if(params->getLLBPartitionPolicy() != llbPartitionPolicy::constant_initial){
	if((params->getLLBPartitionPolicy() == llbPartitionPolicy::adaptive_prev) |
			(params->getLLBPartitionPolicy() == llbPartitionPolicy::adaptive_min) |
			(params->getLLBPartitionPolicy() == llbPartitionPolicy::adaptive_avg)){

		// The ratio that was used for the current LLB partitioning
		float prev_a_ratio = llb->GetARatio() *10.0/8.0;

		// Size of A LLB tile in LLB
		uint64_t a_used_size_acc = llb->GetASize();

		int coo_entry_size =	params->getDataSize() + 2*params->getIdxSize();
		// A to O data ratio in the current batch
		float a_to_o_curr_ratio = (float)a_used_size_acc / (float)(output_nnz*coo_entry_size);
		// The percentage of LLB that B is occupying
		float b_perc = llb->GetBRatio();
		// The ideal percentage of LLB that we wanted for A
		float a_perc = a_to_o_curr_ratio * (1.0-b_perc);

		if(params->getLLBPartitionPolicy() == llbPartitionPolicy::adaptive_min){
			if(llb_reset_count > 0)
				a_perc = std::min(a_perc, prev_a_ratio);
		}
		else if(params->getLLBPartitionPolicy() == llbPartitionPolicy::adaptive_avg){
			a_perc = (a_perc + (prev_a_ratio * (float)llb_reset_count)) / ((float)llb_reset_count+1);
		}
		else if(params->getLLBPartitionPolicy() == llbPartitionPolicy::adaptive_prev){
			// Do nothing! we already have the numbers
		}
		else{
			printf("No such adaptive llb partitioning policy is available!\n");
			exit(1);
		}
		llb_reset_count++;

		// Do not allow it to go less than 1% for A
		a_perc = (a_perc < 0.01) ? 0.01 : a_perc;
		// Do not allow it to go beyond 20% for A
		a_perc = (a_perc > 0.2) ? 0.2 : a_perc;
		a_perc = a_perc * 80.0/100.0;
		// find o_perc; b_perc is constant, a_perc has change
		float o_perc = 1.0 - (a_perc + b_perc);
		// Set the new percentages for LLB
		llb->SetRatios(a_perc, b_perc, o_perc);

		//printf("Old a_perc: %4.2f, new a_perc: %4.2f\n", prev_a_ratio, a_perc);
		//printf("LLB Partitioning Changed To (A:%4.2f, B:%4.2f, O:%4.2f)\n", a_perc, b_perc, o_perc);
		//printf("input nnz_1: %d, a_nnz: %lu, b_nnz: %lu, output nnz: %d, a+b/o: %4.2f\n",
				//input_nnz, a_nnz, b_nnz, output_nnz, (float)(a_nnz+b_nnz)/(float)output_nnz);
		//printf("a_size: %lu, o_size: %d, ratio: %4.2f\n",
		//		a_used_size_acc, output_nnz * 16, (float)(a_used_size_acc)/((float)output_nnz*16));

	}
	return;
}


// Iterates over the A LLB tile columns and divides the consecutive basic tiles
//   in each column in a way that each group fits into the middle DOT buffer (PE buffer)
//   For example: tile 0..5 -> Buffer 0, tile 6..8-> Buffer 1, ...
// The functions gets the id of the column, the start and end I idx,
//   Produces all the i_idxs (starts of the groups)
//   and number of tiles in each group in aggregated manner
// Example:
//		i_indices_middle    = - 1 4 6 - 8 - 12 15 18 -
//		i_offset_idx        = - 12 13 14 - 15 - 16 17 18 -
//		i_entries_count_pos = 0 3 4 7 (3 groups 1:{1,4,6}, 2:{8}, 3:{12, 15, 18})
void Scheduler_8::ExtractAMiddleTiles(int i_start_top, int i_end_top, int j_start_middle,
		std::vector<int> & i_indices_middle, std::vector<int> & i_entries_count_pos,
		std::vector<int> & i_offset_idx, std::vector<int> & a_search_overhead){

	/********************************************/
	/* FWA: Find start and end of the A column vector! */
	// lower and upper bound of A_CSC
	//	find the entries between [lower_bound, upper_bound]
	int lower_bound = i_start_top; int upper_bound = i_end_top;
	// Find start and end of the A_CSC fiber positions
	int a_pos_start = matrix->a_csc_outdims->pos[j_start_middle];
	int a_pos_end   = matrix->a_csc_outdims->pos[j_start_middle+1];
	// Indexes (I_idx values of LLB column A)
	int * vecA_begin = &(matrix->a_csc_outdims->idx[a_pos_start]);
	int * vecA_end   = &(matrix->a_csc_outdims->idx[a_pos_end]);
	// Move to the strating point of row A according to lower_bound
	int offset_idx = a_pos_start;
	while ((*vecA_begin<lower_bound) & (vecA_begin<=vecA_end)) {
		vecA_begin++; offset_idx++;
	}
	/* FWA: Until here!  ************************/

/*
	// In the static tiling mechanism, there is no tile extractor
	//		to maximize the buffer occupancy
	if(params->getTilingMechanism() == tiling::t_static){
		int count = 0;
		// push the init POS entry, i.e., 0!
		i_entries_count_pos.push_back(count++);
		for(int * it = vecA_begin; it<vecA_end; it++, offset_idx++){
			// i index of the macro tile
			int i_idx_middle = *it;
			// push the coordinate index
			i_indices_middle.push_back(i_idx_middle);
			// offset is used for mapping pos to coordinate
			i_offset_idx.push_back(offset_idx);
			// each PE buffer is going to have only one A micro tile at a time
			i_entries_count_pos.push_back(count++);
		}
	}
	else if(params->getTilingMechanism() == tiling::t_dynamic){
*/
		// Middle buffer (PE buffer) size allocated to tensor A (config. value)
		int a_size_lim = (int)((float)params->getPEBufferSize() *
				params->getATensorPercPEBuffer());
		// Helper value to keep track of middle buffer already filled
		int size_buff_a = 0;
		// Helper value to keep track of number of A basic tiles already
		//   in the current middle unit (PE DOT unit)
		int count = 0;

		i_entries_count_pos.push_back(count);
		for(int * it = vecA_begin; it<vecA_end; it++, offset_idx++){
			int i_idx_middle = *it;

			// Early termination: All the tiles are included, get out of the for loop
			if( i_idx_middle >=upper_bound ){
				break;
			}

			//printf("j_idx: %d - [%d, %d]\n", j_idx, *vecA_begin, *vecA_end);
			i_indices_middle.push_back(i_idx_middle);
			i_offset_idx.push_back(offset_idx);
			count++;

			// Next tile size
			int tile_size = matrix->getCSFSize('A', i_idx_middle, j_start_middle);
			// The tile does not fit, thus, the tile size is bigger than PE buffer size
			if(tile_size > a_size_lim){
				printf("Fatal Error: PE Buffer is smaller than the basic tile size! %d %d\n", tile_size, a_size_lim);
				exit(1);
			}

			// if fits in the buffer continue accumulating
			if((size_buff_a + tile_size) < a_size_lim){
				size_buff_a += tile_size;
				//printf("j_idx: %d added, count: %d, size: %d\n", j_idx, count, size_buff_a);
			}
			// If it does not fit then record j and count values
			else{
				i_entries_count_pos.push_back(count);
				// It does not fit! now fetch the did not fit ones
				size_buff_a = tile_size;
			}
		}
		if(count > i_entries_count_pos[i_entries_count_pos.size()-1]){
			i_entries_count_pos.push_back(count);
		}
/*
}
	else{
		printf("No Such Tiling Mechanism is Available!\n"); exit(1);
	}
*/
	// Placeholder, in case the strategy changes. Otherwise these search
	//	numbers can be derived from i_entries_count_pos too
	for(int pe_idx = 1; pe_idx < (int)i_entries_count_pos.size(); pe_idx++){
		if(params->getSearchModelMiddle() == search_tiles::instant)
			a_search_overhead.push_back(0);
		else
			a_search_overhead.push_back(i_entries_count_pos.at(pe_idx));
	}

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

void Scheduler_8::PickPEsAccordingToPolicy(
		int count_needed, std::vector<int> & pe_indices){

	static_distributor dist_policy = params->getStaticDistributorModelMiddle();
	if(dist_policy == static_distributor::round_robin){
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices.push_back((pe_idx+round_robin_slot)%params->getPECount());
		}
		round_robin_slot = (round_robin_slot+count_needed)%params->getPECount();
	}
	else if(dist_policy == static_distributor::oracle){
		QuickSort2Desc(pe_time, pe_utilization_logger, 0, params->getPECount()-1);
		//SortPEs();
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices.push_back(pe_idx%params->getPECount());
		}
	}
	else if(dist_policy == static_distributor::nnz_based){
		QuickSort3Desc(nnz_counts, pe_time, pe_utilization_logger, 0, params->getPECount()-1);
		//QuickSort2Desc(nnz_counts, pe_time, 0, params->getPECount()-1);
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices.push_back(pe_idx%params->getPECount());
		}
	}
	else if(dist_policy == static_distributor::oracle_relaxed){}
	else if(dist_policy == static_distributor::oracle_relaxed_ct){}
	else{
		printf("Static Scheduler Policy Is Not Available!\n");
		exit(1);
	}

	return;
}

// Calculates the earliest start time after taking into account
//	current PE time, top overhead, and middle overhead
void Scheduler_8::CalcTopMiddleDoneTime(int batch_starting_cycle,
		std::vector<int> a_search_overhead,
		std::vector<uint64_t> & top_middle_done_time){

	uint64_t last_time = 0;
	uint64_t * min_operator =
		std::min_element(t_extract_middle_A, t_extract_middle_A + par_middle);

	// Taking into account current PE time and the middle search operator
	uint64_t start_time_extractor = std::max((uint64_t)batch_starting_cycle, *min_operator);
	// Taking into account the top level extractor and intersection unit
	start_time_extractor = std::max(start_time_extractor, extractor_top_A_done);
	for(int pe_idx = 0; pe_idx< (int)a_search_overhead.size(); pe_idx++){
		last_time = start_time_extractor + a_search_overhead.at(pe_idx);
		top_middle_done_time.push_back(last_time);
	}
	*min_operator = last_time;

	return;
}


// Iterates over the B LLB tile columns and divides the consecutive basic tiles
//   in each column in a way that each group fits into the middle DOT buffer (PE buffer)
//   For example: tile 0..5 -> Buffer 0, tile 6..8-> Buffer 1, ...
// The functions gets the id of the column, the start and end I idx,
//   Produces all the i_idxs (starts of the groups)
//   and number of tiles in each group
void Scheduler_8::ExtractBMiddleTiles(int j_index_middle, int k_start_top, int k_end_top,
		std::vector<int> & b_indices_middle, std::vector<int> & b_offset_idx){

	/********************************************/
	/* FWB: Find start and end of the B row vector! */
	// lower and upper bound of B_CSR
	//	find the entries between [lower_bound, upper_bound]
	int lower_bound = k_start_top; int upper_bound = k_end_top;
	// Find start and end of the B_CSR fiber positions
	int b_pos_start = matrix->b_csr_outdims->pos[j_index_middle];
	int b_pos_end   = matrix->b_csr_outdims->pos[j_index_middle+1];
	// Indices (I_idx values of LLB column A)
	int * vecB_begin = &(matrix->b_csr_outdims->idx[b_pos_start]);
	int * vecB_end   = &(matrix->b_csr_outdims->idx[b_pos_end]);
	// Move to the strating point of row A according to lower_bound
	int offset_idx = b_pos_start;
	while ((*vecB_begin<lower_bound) & (vecB_begin<=vecB_end)) {
		vecB_begin++; offset_idx++;
	}
	/* FWB: Until here!  ************************/

	// Starting idx
	for(int * it = vecB_begin; it<vecB_end; it++, offset_idx++){
		int k_idx_middle = *it;
		// Early termination: All the tiles are included, get out of the for loop
		if( k_idx_middle >=upper_bound ){
			break;
		}
		b_indices_middle.push_back(k_idx_middle);
		b_offset_idx.push_back(offset_idx);
	}

	return;
}

// FindBBaiscTileRanges function find the consecutive group of B basic tiles
//  that can be fit into the bottom DOT memory at the same time.
// This grouping is necessary to figure out when we need to wait
//	or we can run computation without obstruction
//	If only one B basic tile fits in PE then b_range_end will look like:
//		B_range_end: 1 2 3 4 5 6 ...
//		which should be translated into: only basic tile 0, 1, 2, 3, 4, ...
void Scheduler_8::FindBBasicTileRanges(int j_index_middle, int k_start_top, int k_end_top,
		std::vector<uint32_t> & b_ranges_start, std::vector<uint32_t> & b_ranges_end){

	// Find the start of the B fiber to iterate over
	// lower and upper bounds are the coordinate index limits
	int lower_bound = k_start_top; int upper_bound = k_end_top;
	int b_pos_start = matrix->b_csr_outdims->pos[j_index_middle];
	int b_pos_end   = matrix->b_csr_outdims->pos[j_index_middle+1];
	// Indexes (I_idx values of LLB column A)
	int * vecB_begin = &(matrix->b_csr_outdims->idx[b_pos_start]);
	int * vecB_end   = &(matrix->b_csr_outdims->idx[b_pos_end]);
	// Move to the strating point of row A according to lower_bound
	while ((*vecB_begin<lower_bound) & (vecB_begin<=vecB_end)) vecB_begin++;
	//printf("find B:");

	// Maximum size that can be allocated to B basic tiles
	int b_size_lim = (int)((float) params->getPEBufferSize() *
			(1.0-params->getATensorPercPEBuffer()));

	// it_idx_start is the start index of B Basic tiles (starts from 0)
	// it_idx_end is the end index of B basic tile that can be fit into
	//	the same bottom buffer all together at the same time
	int it_idx_start = 0, it_idx_end = 0;
	for(int * it = vecB_begin; it<vecB_end; it++, it_idx_start++){
		int size_buffer_b = 0;
		int * curr_batch = it;
		it_idx_end = it_idx_start+1;

		// Start iterating from the starting basic tile and see how many
		//	B basic tiles can be fit into the bottom memory
		while((curr_batch < vecB_end) & (*curr_batch < upper_bound)){
			//printf("%d,",it_idx_end);
			int k_index_middle = *curr_batch;
			int tile_size = matrix->getCSFSize('B', j_index_middle, k_index_middle);
			// Surpassed the size limit; record the end and return to for loop
			if((size_buffer_b + tile_size) > b_size_lim){
				b_ranges_end.push_back(it_idx_end);
				break;
			}
			// Add the tile to the batch and check the next one
			else{
				curr_batch++; it_idx_end++;
				size_buffer_b += tile_size;
			}
			// We have covered all the basic tiles in the B fiber
			//	record the end_idx and jump out
			if((curr_batch == vecB_end) | (*curr_batch >= upper_bound)){
				b_ranges_end.push_back(it_idx_end);
				break;
			}

		} // While loop

		// We have covered all the basic tiles in the B row fiber
		if((curr_batch == vecB_end) | (*curr_batch >= upper_bound)){
			break;
		} // if statement
	}	// for *it loop
	return;
}

// Sort the time for PEs. The N smallest ones will be used for
//	the current middle column of A
void Scheduler_8::SortPEs(){
	std::sort(pe_time, pe_time + params->getPECount());
	return;
}

// This is a beautiful bw logger that gets the start cycle
//   of each tile multiplication and in a cycle accurate way says
//   when the data transfer is finished
// Its role is to keep track of bandwidth either for the top level
//  or middle level, depending on the bw_logger and bytes_per_ns assignment
uint64_t Scheduler_8::updateBWLog(uint64_t starting_cycle, uint64_t action_bytes,
		float *bw_logger, float bytes_per_ns){

	total_traffic += action_bytes;

	float action_bytes_f = float(action_bytes);
	for(uint64_t i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		float rem_cap = bytes_per_ns - bw_logger[i_idx];
		// Move on until finding the first DRAM bw available cycle
		if((action_bytes_f > 0) & (rem_cap == 0))
			continue;
		// Use the available BW, but this is not the end
		if(action_bytes_f > rem_cap){
			bw_logger[i_idx] = bytes_per_ns;
			action_bytes_f -= rem_cap;
		}
		// Last cycle needed to transfer the specified data
		else{
			bw_logger[i_idx] += action_bytes_f;
			return i_idx;
		}
	}
	printf("Starting cycle: %lu, Bytes: %lu, bandwidth: %f bytes/s\n",
			starting_cycle, action_bytes, bytes_per_ns);
	printf("%d bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}

// All of the PEs should finish their work before fetching the next	B tiles
void Scheduler_8::SyncPETimes(){

	int max_val = *std::max_element(pe_time, pe_time+params->getPECount());
	std::fill(pe_time, pe_time+params->getPECount(), max_val);

	return;
}

// Gets the Column boundary and start address of matrix B row,
//   returns the start and stop address or the matrix B row, i.e., o_reuse
// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
void Scheduler_8::ExtractBTopTile(int k_idx, int j_idx_start){

	int a_reuse = params->getAReuse();
	int j_idx_stop = j_idx_start;
	int k_idx_stop = std::min(k_idx+a_reuse, o_tiled_cols);

	// In static tiling case we already now what the j_idx_end
	//	should be and have set the OReuse value in params
	if(params->getTilingMechanism() == tiling::t_static){
		int j_idx_stop = std::min(j_idx_start + params->getOReuse(),
				b_tiled_rows);
		uint64_t extra_size = AccumulateSize('B', j_idx_start, j_idx_stop,
				k_idx, k_idx_stop, params->getBFormat());
		llb->AddToLLB('B', Req::read, extra_size, UPDATE_TRAFFIC);
	}
	else if(params->getTilingMechanism() == tiling::t_dynamic){
		// Add rows until it either runs out of memory or reaches the last row
		for(int idx = j_idx_start; idx<b_tiled_rows; idx++){
			// Find the size of the new row
			uint64_t extra_size = AccumulateSize('B', idx, idx+1,
					k_idx, k_idx_stop, CSX::CSF);
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
						k_idx+a_reuse, k_idx+a_reuse+1, CSX::CSF);
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
		printf("No Such Tiling Mechanism is Available!\n");
	}

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
			uint64_t temp = (serial_overhead_search%par_search_top == 0)?
				serial_overhead_search/ par_search_top:
				serial_overhead_search/ par_search_top + 1;
			overhead_search = temp + log2_search_top;
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
		overhead_posbuild = uint64_t(log2_mbuild_top + 1);
	}

	// overhead = max(tbuild,search,mbuilt) + overhead_posbuild;
	overhead_tbuild_search_top_B =
		std::max(overhead_basic_tile_build,	overhead_search);
	overhead_extractor_top_B = std::max(overhead_tbuild_search_top_B, overhead_mbuild);
	overhead_extractor_top_B += overhead_posbuild;
	// This variable is used to know when extracting tensor B for LLB is over
	extractor_top_B_done = overhead_extractor_top_B +
		*std::min_element(pe_time, pe_time+params->getPECount());
	// This variable is used for scheduling tensor A tbuild phase
	//	(building the basic tiles for LLB)
	tbuild_search_top_B_done = overhead_tbuild_search_top_B +
		*std::min_element(pe_time, pe_time+params->getPECount());
	/*
	printf("B Overheads\n\tBasic tile build: %lu\n",overhead_basic_tile_build);
	printf("\tSearch: %lu\n",overhead_search);
	printf("\tMbuild: %lu\n",overhead_mbuild);
	printf("\tPosbuild: %lu\n",overhead_posbuild);
	printf("\tExtract %lu\n",overhead_extractor_top_B);
	printf("B extract finish time: %lu\n",extractor_top_B_done);
	*/
	return;
}

// Find out what is the A row sizes we need to fetch
//	after LLB intersection
void Scheduler_8::PreCalculateARowsSize(int j_start, int j_end,
		int k_start, int k_end){

	/************************************/
	/* Pre-calculating A and O row size */
	int * b_llb_horizontalSum = new int[b_tiled_rows];
	// Calculate the horizontal sum of B PE tiles in the LLB tile
	//   This sum is used to decide whether to bring PE tiles of A
	//   It basically says whether there is any nnz PE tile in each row
	CalcBLLBHorizontalSum(j_start, j_end,
		k_start, k_end, b_llb_horizontalSum);

	#pragma omp parallel for
	for(int i_idx = 0; i_idx < o_tiled_rows; i_idx++){
		// Find the size of the A row size needs to be fetched
		//   taking into account whether B row is empty or not
		if((params->getBasicTileBuildModelTop() ==  basic_tile_build::parallel) |
				(params->getBasicTileBuildModelTop() ==  basic_tile_build::serial)){
			vecCommittedRows_ASize[i_idx] =
				AccumulateSize('A', i_idx, i_idx+1, j_start, j_end, CSX::CSF);
		}
		else{
			vecCommittedRows_ASize[i_idx] = AccumulateSize_AwrtB(i_idx,
				j_start, j_end, b_llb_horizontalSum);
		}

			//printf("--> %lu , %lu\n",vecCommittedRows_ASize[i_idx], vecCommittedRows_OSize[i_idx]);
	}
	delete [] b_llb_horizontalSum;

	return;
}

// Early fetched the A basic tiles when using not pre-tiled matrix
//	Therefore, we are having eith a parallel or serial (unlikely) basic tile builder
void Scheduler_8::EarlyFetchABasicTiles(int i_start_top, int i_end_top,
		int j_start_top, int j_end_top){

	uint32_t a_traffic = 0;
	// Start fetching data/metadata of basic tiles of A if and only if
	//	We are going to use the basic tile builder in the top extractor
	if((params->getBasicTileBuildModelTop() ==  basic_tile_build::parallel) |
			(params->getBasicTileBuildModelTop() ==  basic_tile_build::serial)){
		// iterate over all columns of top tile of A
		for(int j_idx = j_start_top; j_idx < j_end_top; j_idx++){
			// iterate over the rows that are between the start and end
			for(int t_idx = matrix->a_csc_outdims->pos[j_idx];
					t_idx < matrix->a_csc_outdims->pos[j_idx+1];
					t_idx++){
				int i_idx = matrix->a_csc_outdims->idx[t_idx];
				if(i_idx < i_start_top) continue;
				if(i_idx > i_end_top) break;
				// Set the fetch flag to 1 -> so, it will not be fetched again
				matrix->a_csc_outdims->vals[t_idx] = 1.0;
				// Add A tiled size to the traffic computation
				uint32_t a_tile_size = matrix->getCSFSize('A', i_idx, j_idx);
				a_traffic += a_tile_size;
				// Stats update for top and middle NoC traffic
				//stats->Accumulate_a_read((uint64_t)a_tile_size);
				stats->Accumulate_a_read_middle((uint64_t)a_tile_size);
			} // for t_idx
		} // for j_idx

		uint64_t starting_cycle_top =
			*std::min_element(pe_time, pe_time+params->getPECount());
		// Now do the actual fetching of all A data and metadata
		uint64_t endingCycle_top = updateBWLog(starting_cycle_top, a_traffic,
			top_bw_logger, top_bytes_per_ns);
		// No process can start before loading A basic tile data/metadata
		for(int pe_idx = 0; pe_idx < params->getPECount(); pe_idx++){
			if(pe_time[pe_idx] < endingCycle_top)
				pe_time[pe_idx] = endingCycle_top;
		}
		stats->Set_cycles(std::max(stats->Get_cycles(), endingCycle_top));
		stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());

	}
	return;
}

// Early fetched the B basic tiles when using not pre-tiled matrix
//	Therefore, we are having eith a parallel or serial (unlikely) basic tile builder
void Scheduler_8::EarlyFetchBBasicTiles(int j_start_top, int j_end_top,
		int k_start_top, int k_end_top){

	uint32_t b_traffic = 0;
	// Start fetching data/metadata of basic tiles of B if and only if
	//	We are going to use the basic tile builder in the top extractor
	if((params->getBasicTileBuildModelTop() ==  basic_tile_build::parallel) |
			(params->getBasicTileBuildModelTop() ==  basic_tile_build::serial)){
		// iterate over all rows of top tile of B
		for(int j_idx = j_start_top; j_idx < j_end_top; j_idx++){
			// iterate over the rows that are between the start and end
			for(int t_idx = matrix->b_csr_outdims->pos[j_idx];
					t_idx < matrix->b_csr_outdims->pos[j_idx+1];
					t_idx++){
				int k_idx = matrix->b_csr_outdims->idx[t_idx];
				if(k_idx < k_start_top) continue;
				if(k_idx > k_end_top) break;
				// Set the fetch flag to 1 -> so, it will not be fetched again
				matrix->b_csr_outdims->vals[t_idx] = 1.0;
				// Add A tiled size to the traffic computation
				uint32_t b_tile_size = matrix->getCSFSize('B', j_idx, k_idx);
				b_traffic += b_tile_size;
				// Stats update for top and middle NoC traffic
				//stats->Accumulate_a_read((uint64_t)a_tile_size);
				stats->Accumulate_b_read_middle((uint64_t)b_tile_size);
			} // for t_idx
		} // for j_idx

		uint64_t starting_cycle_top =
			*std::min_element(pe_time, pe_time+params->getPECount());
		// Now do the actual fetching of all B data and metadata
		uint64_t endingCycle_top = updateBWLog(starting_cycle_top, b_traffic,
			top_bw_logger, top_bytes_per_ns);
		// No process can start before loading B basic tile data/metadata
		for(int pe_idx = 0; pe_idx < params->getPECount(); pe_idx++){
			if(pe_time[pe_idx] < endingCycle_top)
				pe_time[pe_idx] = endingCycle_top;
		}
		stats->Set_cycles(std::max(stats->Get_cycles(), endingCycle_top));
		stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());

	}
	return;
}

// Gets two vectors and finds the intersection coordinates between them
// This function is used in ideal llb partitioning SoL pocily only
void Scheduler_8::intersectTwoVectors(int * vec1_begin, int * vec1_end,
		int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound,
		std::vector<int> & intersect_vector){

	int max_val = 0;
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
		max_val = std::max(*p1,*p2);
		if(*p1<max_val){
			while ((*p1<max_val) & (p1<=vec1_end) & (*p1<upper_bound)) {p1++;}
		}
		else if(*p2<max_val){
			while ((*p2<max_val) & (p2<=vec2_end) & (*p2<upper_bound)) {p2++;}
		}
		else{
			intersect_vector.push_back(*p1);
			p1++; p2++;
		}
	}
	return;
}

// Multiplies one LLB row of A to a B LLB tile and reports what the iutput size will be
// Please note that this is just for the ideal llb partition policy and meant to
//	produce SoL variant result.
// This function is used straight out of ExTensor scheduler code (we needed the same
//	computation dataflow);Thus, it has been tested and works correctly.
//	There has been just some clean ups to keep the essential parts
uint64_t Scheduler_8::multiplyOneARowInLogOutput(int i_idx,
		int j_start_top, int j_end_top, int k_start_top, int k_end_top){

	// Becausse the log matrix needs to have the address of the starting row
	//   the i_idx needs to be adjusted everytime to avoid seg fault
	matrix->SetLogMatrixStart(i_idx);
	// Remove all the output produced in the previous i_idx
	matrix->evictOutputLogMatrix();

	/********************************************/
	/* FWA: Find start and end of the A vector! */
	// lower and upper bound of A_CSR and B_CSC fiber to intersect
	//	find the intersect between [lower_bound, upper_bound]
	int lower_bound = j_start_top; int upper_bound = j_end_top;
	// Find start and end of the A_CSR fiber positions
	int a_pos_start = matrix->a_csr_outdims->pos[i_idx];
	int a_pos_end = matrix->a_csr_outdims->pos[i_idx+1];
	// Indexes (J values of row A)
	int * vecA_begin = &(matrix->a_csr_outdims->idx[a_pos_start]);
	int * vecA_end = &(matrix->a_csr_outdims->idx[a_pos_end-1]);
	// Move to the strating point of row A according to lower_bound
	while ((*vecA_begin<lower_bound) & (vecA_begin<=vecA_end)) vecA_begin++;
	// distance between the start of the idx to where vecA_begin is after fastforward
	//int vecA_offset = vecA_begin -  &(matrix->a_csr_outdims->idx[a_pos_start]);
	/* FWA: Until here!  ************************/

	// Iterate over all B column fibers (or O coulmns)
	for(int k_index_middle = k_start_top; k_index_middle < k_end_top; k_index_middle++){
			/********************************************/
			/* FWB: Find start and end of the B vector! */
			// Find start and end of the B_CSC fiber
			int b_pos_start = matrix->b_csc_outdims->pos[k_index_middle];
			int b_pos_end = matrix->b_csc_outdims->pos[k_index_middle+1];
			int * vecB_begin = &(matrix->b_csc_outdims->idx[b_pos_start]);
			int * vecB_end = &(matrix->b_csc_outdims->idx[b_pos_end-1]);
			while ((*vecB_begin<lower_bound) & (vecB_begin<=vecB_end)) vecB_begin++;
			//int vecB_offset = vecB_begin -  &(matrix->b_csc_outdims->idx[b_pos_start]);
			/* FWB: Until here!  ************************/

			// Find the intersection of an A_CSR and a B_CSC fiber
			std::vector<int> intersect_vector, intersect_overhead_vector, a_idx, b_idx;
			//int len_B_fiber = 0;
			intersectTwoVectors(vecA_begin, vecA_end, vecB_begin, vecB_end,
					lower_bound, upper_bound, intersect_vector);

			// There is no intersection, so no computation. Skip!
			if(intersect_vector.size() == 0){
				continue;
			}
			for (std::vector<int>::iterator it = intersect_vector.begin();
					it != intersect_vector.end(); ++it){
				int j_index_middle = *it;

				// Do the actual calculation
				matrix->CSRTimesCSROnlyLogUpdated(i_idx, j_index_middle, k_index_middle);
			}
	}
	// Report the COO size that the output produces
	return matrix->getOutputLogNNZCOOSize();

}

// ExtractAOTopTiles gets the dimensions of the B LLB tile and grows number of rows
// until filling up the top buffer (LLB) for A and O tensors
//	Tasks: 1) Report i_end_top 2) Fetch A and O top tiles into LLB buffer
void Scheduler_8::ExtractATopTile(int i_start_top, int &i_end_top,
		int j_start_top, int j_end_top, int k_start_top, int k_end_top){

	// In the static tiling mechasnism we already know what i_idx_end
	//	should be since it is constant!
	if(params->getTilingMechanism() == tiling::t_static){
        // not o_tiled_rows 
		i_end_top = std::min(i_start_top + params->getITopTile(), o_tiled_rows);
		uint64_t a_row_size = 0;
		for(int i_idx_top = i_start_top; i_idx_top < i_end_top; i_idx_top++){
			a_row_size += vecCommittedRows_ASize[i_idx_top];
		}
		llb->AddToLLB('A', Req::read, a_row_size, UPDATE_TRAFFIC);

	}
	else if(params->getTilingMechanism() == tiling::t_dynamic){
		// SoL variant of llb partitioning policy! It assumes a constant B percentage
		//	then looks at after computation output size to determine the ideal
		//	LLB partitioning between A and O
		// THIS IS AN SOL VARIANT! don't use it for TACTile non-SoL variants!
		if(params->getLLBPartitionPolicy() ==  llbPartitionPolicy::ideal){
			uint64_t a_llb_size = 0, o_llb_size = 0;
			// B LLB Size
			uint64_t b_llb_size = llb->GetBSize();

			// Please note that only one output log matrix is produced for the whole process and
			//   after each row computation it is cleaned up (NOT deleted) for the next row
			// Create a temporary log matrix to see if the LLB tile sizing works
			//   It consists of only one row
			matrix->initOutputLogMatrix(i_start_top, i_start_top+1, k_start_top, k_end_top);

			for(int i_idx_top = i_start_top; i_idx_top<o_tiled_rows;i_idx_top++){
				// This is a A row multiplication out of ExTensor style!
				// Multiplies a row of A to a B LLB tile
				uint64_t output_size = multiplyOneARowInLogOutput(i_idx_top,
						j_start_top, j_end_top, k_start_top, k_end_top);
				// If the newly produved output size allows then ncrement the number of LLB tile rows
				if((a_llb_size + vecCommittedRows_ASize[i_idx_top] +
							o_llb_size + output_size + b_llb_size) < llb->GetCapacity()){
					a_llb_size += vecCommittedRows_ASize[i_idx_top];
					o_llb_size += output_size;
					i_end_top++;
				}
				else{ break; }
			}

			// Remove the temporary log matrix created
			matrix->deleteOutputLogMatrix();

			// Add the finalized a_llb_size to the llb memory
			llb->AddToLLB('A', Req::read, a_llb_size, UPDATE_TRAFFIC);

			// Get the capacity of the llb and determine how much of it is left out for output!
			uint64_t llbCapacity = llb->GetCapacity();
			o_llb_size = llbCapacity - a_llb_size - b_llb_size;
			// First set the ratios
			llb->SetRatios((float)a_llb_size / (float)llbCapacity,
					(float)b_llb_size/ (float)llbCapacity,
					(float)o_llb_size/ (float)llbCapacity);
			// Then set the new sizes
			// PLEASE don't change the order between ratios and sizes function calls
			llb->SetSizes(a_llb_size, b_llb_size, o_llb_size);

			printf("New ratios: %0.2f, %0.2f, %0.2f for i [%d-%d] \n",
					llb->GetARatio(), llb->GetBRatio(), llb->GetORatio(), i_start_top, i_end_top );
			//printf("New sizes: %lu, %lu, %lu for %d rows\n",
			//	a_llb_size, b_llb_size, o_llb_size, i_end_top-i_start_top);
			fflush(stdout);
		}
		// If not using the ideal llb partitioning policy(an SoL variant) then take the easy way
		else{
			i_end_top = i_start_top;
			// Go over every row until reaching the maximum partition allowance
			for(int i_idx_top = i_start_top; i_idx_top < o_tiled_rows; i_idx_top++ ){
				uint64_t a_row_size = vecCommittedRows_ASize[i_idx_top];
				if (llb->DoesFitInLLB('A', a_row_size)){
					llb->AddToLLB('A', Req::read, a_row_size, UPDATE_TRAFFIC);
					i_end_top++;
				}
				else{break;}
			}
		}

	}
	else{
		printf("No such Tiling Mechanism is Available!\n"); exit(1);
	}

	// Please note that Extract Unit overheads for A are calculated
	//	similar to calculations for matrix B
	// / *************************************************************** /
	// Calculate the basic tile build overhead top level (DRAM->LLB))
	// Important output : overhead_basic_tile_build
	uint64_t overhead_basic_tile_build = 0;
	// Instant and not tile build will take 0 cycles
	if((params->getBasicTileBuildModelTop() == basic_tile_build::instant)
			|(params->getBasicTileBuildModelTop() == basic_tile_build::noTilebuild))
		overhead_basic_tile_build = 0;
	// Parallel and serial cases
	else{
		int length_array = j_end_top - j_start_top;
		uint64_t * per_col_runtime =	new uint64_t[length_array];
		// Get the overhead of building basic tiles in each column
		for(int j_idx_middle = j_start_top;
				j_idx_middle< j_start_top + length_array;
				j_idx_middle++){
			uint64_t nnz =
				AccumulateNNZ('A', i_start_top, i_end_top, j_idx_middle, j_idx_middle+1);
			uint64_t nnr =
				AccumulateNNR('A', i_start_top, i_end_top, j_idx_middle, j_idx_middle+1);
			per_col_runtime[j_idx_middle-j_start_top] = nnz+nnr;
			overhead_basic_tile_build += (nnz+nnr);
		}
		// if we are doing it parallel then schedule it with
		//	column granularity
		if(params->getBasicTileBuildModelTop() == basic_tile_build::parallel){
			int par_factor = par_tbuild_A_top;
			int *sched_arr = new int[par_factor];
			std::fill(sched_arr, sched_arr+par_factor, 0);
			for (int p_idx = 0; p_idx < length_array; p_idx++){
				*std::min_element(sched_arr, sched_arr+par_factor) += per_col_runtime[p_idx];
			}
			overhead_basic_tile_build = *std::max_element(sched_arr, sched_arr+par_factor);
			delete [] sched_arr;
		}
		delete [] per_col_runtime;
	}

	// / *  Calculate the search overhead top level (DRAM->LLB) * /
	// Important output : overhead_search
	uint64_t overhead_search = 0;
	uint64_t serial_overhead_search = 0;
	uint64_t max_overhead_column = 0;
	// Instant and not tile build will take 0 cycles
	if(params->getSearchModelTop() == search_tiles::instant)
		overhead_search = 0;
	else{
		for(int j_idx_middle = j_start_top; j_idx_middle< j_end_top; j_idx_middle++){
			uint64_t overhead_column =
				AccumulateNNZTiles('A', i_start_top, i_end_top, j_idx_middle, j_idx_middle+1);
			serial_overhead_search += overhead_column;
			max_overhead_column = std::max(max_overhead_column, overhead_column);
		}
		// Parallel case
		if(params->getSearchModelTop() == search_tiles::parallel){
			uint64_t temp = (serial_overhead_search%par_search_top == 0)?
				serial_overhead_search/ par_search_top:
				serial_overhead_search/ par_search_top + 1;
			overhead_search = temp + log2_search_top;
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
		overhead_posbuild = uint64_t(log2_mbuild_top + 1);
	}

	// overhead = max(tbuild,search,mbuilt) + overhead_posbuild;
	overhead_extractor_top_A = std::max(overhead_basic_tile_build,
			overhead_search);
	overhead_extractor_top_A = std::max(overhead_extractor_top_A, overhead_mbuild);
	overhead_extractor_top_A += overhead_posbuild;
	// This variable is used to know when extracting tensor B for LLB is over
	extractor_top_A_done = overhead_extractor_top_A +
		std::max(
				tbuild_search_top_B_done,
				*std::min_element(pe_time, pe_time+params->getPECount()));
	return;
}

// Find horizonatal sum of B tiles in LLB; this is used in
//   deciding whether to bring A PE tiles to LLB or not
// If horizontalsum of the row is zero then do not bring the
//   corresponding PE tile of A
void Scheduler_8::CalcBLLBHorizontalSum(int j_start, int j_end,
		int k_start, int k_end, int * b_llb_horizontalSum){

	for(int j_idx = j_start; j_idx<j_end; j_idx++){
		if (AccumulateSize('B', j_idx, j_idx+1, k_start, k_end, CSX::CSF) > 0){
			b_llb_horizontalSum[j_idx] = 1;
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
uint64_t Scheduler_8::AccumulateSize_AwrtB(int i_idx,
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


// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the size that block of tiles would occupy in LLB
uint64_t Scheduler_8::AccumulateSize(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end, CSX inp_format){

	uint64_t size_tot = 0;
	switch (mat_name){
		case 'A':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					size_tot += matrix->accumulateCSFSize_sparse('A',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					size_tot += matrix->accumulateCSFSize_sparse('A',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					size_tot += matrix->accumulateCSFSize_sparse('A',i_idx, d2_start, d2_end, CSX::CSR);
			}
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
				if(inp_format == CSX::COO)
					size_tot += matrix->accumulateCOOSize('O',i_idx, d2_start, d2_end);
				else if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize('O',i_idx, d2_start, d2_end);
				else{
					printf("Format not supported!\n"); exit(1);
				}

			break;}
		// Output log file
		case 'L':{
			for(int i_idx= 0; i_idx < d1_end- d1_start; i_idx++)
				if(inp_format == CSX::COO){
					size_tot += matrix->accumulateCOOSize('L',i_idx, 0, d2_end-d2_start);
				}
				else if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize('L',i_idx, 0, d2_end-d2_start);
				else{
					printf("Format not supported!\n"); exit(1);
				}

			break;}

		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}

	return size_tot;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the total number of non-zero elements in that range
uint64_t Scheduler_8::AccumulateNNZ(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t nnz_tot = 0;
	switch (mat_name){
		case 'A':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					nnz_tot += matrix->accumulateNNZ('A',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					nnz_tot += matrix->accumulateNNZ('A',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					nnz_tot += matrix->accumulateNNZ('A',i_idx, d2_start, d2_end, CSX::CSR);
			}
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
uint64_t Scheduler_8::AccumulateNNR(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t nnr_tot = 0;
	switch (mat_name){
		case 'A':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					nnr_tot += matrix->accumulateNNR('A',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					nnr_tot += matrix->accumulateNNR('A',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation as default
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					nnr_tot += matrix->accumulateNNR('A',i_idx, d2_start, d2_end, CSX::CSR);
			}
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
			// Use CSR representation as default
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
uint64_t Scheduler_8::AccumulateNNZTiles(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end){

	uint64_t nnz_tot = 0;
	switch (mat_name){
		case 'A':{
			// Use CSR representation
			if(d1_end == (d1_start+1)){
					nnz_tot += matrix->accumulateNNZTiles('A',d1_start, d2_start, d2_end, CSX::CSR);
			}
			// Use CSC representation
			else if(d2_end == (d2_start+1)){
					nnz_tot += matrix->accumulateNNZTiles('A',d2_start, d1_start, d1_end, CSX::CSC);
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++)
					nnz_tot += matrix->accumulateNNZTiles('A',i_idx, d2_start, d2_end, CSX::CSR);
			}
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

// Sorting a, b, and c according to a (b and c are dependent)
// This function is used for finding to corresponding PE units, and pe_utilization_logger
//	a is the nnz count for each pe, and pe is the pe progress time
template<class A, class B, class C> void Scheduler_8::QuickSort3Desc(A a[], B b[], C c[], int l, int r)
{
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

		if (i <= j)
		{
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
template<class A, class B> void Scheduler_8::QuickSort2Desc(A a[], B b[], int l, int r)
{
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

		if (i <= j)
		{
			Swap(a[i], a[j]);
			Swap(b[i], b[j]);
			i++;
			j--;
		};
	} while (i <= j);
	if (l < j)QuickSort2Desc(a, b, l, j);
	if (i < r)QuickSort2Desc(a, b, i, r);
}



template<typename T>
void Scheduler_8::Swap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

void Scheduler_8::printPEs(){
	for(int i=0; i<params->getPECount(); i++)
		printf("%lu ", pe_time[i]);
	printf("\n");
	return;
}

void Scheduler_8::PrintBWUsage(){
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

	printf("BW logger a: %lu, b: %lu, o_r: %lu, o_w: %lu\n", a_bwl_traffic, b_bwl_traffic, o_bwl_traffic_read, o_bwl_traffic_write);

	printf("total_traffic %lu, a_read %lu, b read %lu, o_read %lu, o_write %lu\n",
			total_traffic, a_traffic, b_traffic, o_traffic_read, o_traffic_write);
    PrintBWLog();
	return;
}

void Scheduler_8::PrintBWLog(){
	FILE * pFile;
	pFile = fopen ("top_bw_log.txt","w");
	for (uint64_t i=0; i< stats->Get_cycles(); i++)
		fprintf(pFile, "%f\n", top_bw_logger[i]);
	fclose(pFile);
	return;
}

// Reports the standard deviation and mean of the PE utilization
void Scheduler_8::ReportPEUtilization(){
	printf("Sorted PE intersection cycles:\n");
	std::sort(pe_utilization_logger, pe_utilization_logger + params->getPECount());
/*
	for(int i=0;i<params->getPECount(); i++){
		printf("%lu ,",pe_utilization_logger[i]);
	}
	printf("\n");
*/
	// Calculate STDEV and MEAN
	double sum = 0.0, mean, standardDeviation = 0.0;
	for(int i=0;i<params->getPECount(); i++){
		sum += (double)pe_utilization_logger[i];
	}
	mean = sum/params->getPECount();
	for(int i=0; i<params->getPECount(); i++){
		standardDeviation += pow((double)pe_utilization_logger[i] - mean, 2);
	}
  standardDeviation = sqrt(standardDeviation / (double)params->getPECount());

	printf("MEAN: %.0f, STDEV %.0f\n", mean, standardDeviation);

	return;
}

#endif
