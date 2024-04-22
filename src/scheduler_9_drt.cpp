#ifndef SCHEDULER_9_DRT_CPP
#define SCHEDULER_9_DRT_CPP

// Two level OuterSpace + Dynamic Reflexive Tiling
// The levels are Top and Bottom SDOTs, where Top SDOT brings "Micro Tiles" in
//	A-stationary fashion for both top and bottom SDOTs
//	B-stationary fashion and the bottom SDOT receives scalaras in A-stationary fashion.

#include "scheduler_9_drt.h"

// constructor -> intializer of the scheduler
Scheduler_9_drt::Scheduler_9_drt(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
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
	if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
		nnz_counts = new uint64_t[params->getPECount()];
		std::fill(nnz_counts, nnz_counts+params->getPECount(), 0);
	}

	pe_utilization_logger = new uint64_t[params->getPECount()];
	std::fill(pe_utilization_logger, pe_utilization_logger+params->getPECount(), 0);

	// Pre-calculate A row sizes corresponding to B LLB tile
	//	that will be used in different parts of code
	//vecCommittedRows_ASize = new uint64_t[o_tiled_rows];
	preCalculated_BColSize = new uint64_t[o_tiled_cols];

	// DEBUGGING variables. DON'T REMOVE
	deb_a_traffic = 0; deb_b_traffic = 0; deb_o_traffic_read = 0; deb_o_traffic_write = 0; deb_total_traffic = 0;


	return;
}

// Destructor -> delete [] bunch of dynamically allocated arrays
Scheduler_9_drt::~Scheduler_9_drt(){
	delete [] middle_bw_logger;
	delete [] top_bw_logger;
	delete [] pe_time;
	delete [] preCalculated_BColSize;
	delete [] pe_utilization_logger;
	if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
		delete [] nnz_counts;
	}

	return;
}

// Reset all the internal stats; Used when there are multiple runs in one main file
//	Usually used for bandwidth scaling sweep
void Scheduler_9_drt::Reset(){

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
	// Reset round-robin scheduler slot holder
	round_robin_slot = 0;

	return;
}

/* Dataflow in pseudo code is :
 *
 * for(J_top = 0; J_top < Jt; J_top++)
 *	for(I_top = 0; I_top < It; I_top++)
 *		for(K_top = 0; K_top < Kt; K_top++)
 *			for(j_bottom=0; j_bottom<a_micro_cols; j_bottom++)
 *				for(i_bottom=0; i_bottom<a_micro_rows; i_bottom++)
 *					for(k_bottom=0; k_bottom<b_micro_cols; k_bottom++)
 *					// The next three loopnests are in the ScheduleBasicTileMult finction
 *						for(j_inner=0; j_inner<tile_dim (32); j_inner++)
 *							parallel for(i_inner=0; i_bottom<; i_++) -- Parallelization ---
 *								for(k_inner=0; k_bottom<; k_++)
 *									i = i_top*a_micro_row + i_bottom
 *									j = j_top*a_micro_col + j_bottom
 *									k = k_top*b_micro_col + k_bottom
 *									a[i][k] += a[i][j]*b[j][k];
 *
 */
int Scheduler_9_drt::Run(){

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

	int j_end_top =0, i_end_top =0, k_end_top=0;
	llb_reset_count = 0; round_robin_slot = 0;
	int b_reuse = params->getAReuse();
	b_reuse_const = b_reuse;

	j_end_top = 0; 		totalSize_outputCOOLog = 0;
	for(int j_start_top = 0; j_start_top < b_tiled_rows; /* EMPTY */ ){
		for(int i_start_top = 0; i_start_top < o_tiled_rows; /* EMPTY */){
			// Calculate O_reuse and fetch A tiles
			ExtractATopTile(i_start_top, j_start_top);

			// Load the latest updated a_reuse (it might be bigger than user defined)
			b_reuse = params->getAReuse();
			// Now, find the end of i and j iterators based ExtractATopTile result!
			i_end_top = std::min(i_start_top+b_reuse, o_tiled_rows);
			j_end_top = std::min(j_start_top + params->getOReuse(), b_tiled_rows);

			// Find out what is the B col sizes we need to fetch
			//	after LLB intersection
			PreCalculateBColsSize(i_start_top, i_end_top, j_start_top, j_end_top);
			for(int k_start_top = 0; k_start_top < o_tiled_cols; /* EMPTY */){
				// i_end_top and j end_top are already calculated,
				//	now need to figure out k_end_top
				ExtractBTopTile(i_start_top,i_end_top, j_start_top, j_end_top, k_start_top, k_end_top);

				// Schedule the multiplication of basic tiles of an A top tile (LLB)
				//	by basic tiles of a top tile of B
				ScheduleBottomSDOT(i_start_top, i_end_top,
						j_start_top, j_end_top, k_start_top, k_end_top);

				// In static tiling the sizes should be accurate and we should
				//	never face oversized
				if(llb->GetSize() > llb->GetCapacity()){
					printf("LLB Size is not Enough!\
							(This message should be shown only in static tiling)\n");
					exit(1);
				}
				// Evict the B LLB tile to bring the new LLB tile (A-stationary dataflow)
				llb->EvictMatrixFromLLB('B', DONT_UPDATE_TRAFFIC);
				llb->EvictMatrixFromLLB('O', DONT_UPDATE_TRAFFIC);
				std::fill(matrix->b_csr_outdims->vals,
					matrix->b_csr_outdims->vals + matrix->b_csr_outdims->nnz, 0.0);

				// Update k_start_top for the next for loop
				k_start_top = k_end_top;
			}// k_start_top

			llb->EvictMatrixFromLLB('A', DONT_UPDATE_TRAFFIC);
			std::fill(matrix->a_csc_outdims->vals,
					matrix->a_csc_outdims->vals + matrix->a_csc_outdims->nnz, 0.0);

			// Update i_start_top for the next for loop
			i_start_top = i_end_top;
			// A LLB row is changed, load back the user defined b_reuse val
			params->setAReuse(b_reuse_const);
		}//i_start_top

		// Update j_start_top for the next for loop
		j_start_top = j_end_top;
	}//j_start_top
	// The outoput file that is going to be written to output at the last phase after reduced!
	uint64_t topLevel_output_bw = AccumulateSize('O', 0, o_tiled_rows,
			0, o_tiled_cols, params->getOFormat());
	deb_o_traffic_write += topLevel_output_bw;
	deb_o_traffic_read += totalSize_outputCOOLog;
	// Update output write bytes!
	llb->AddToLLB('O', Req::write, topLevel_output_bw, UPDATE_TRAFFIC);
	llb->EvictMatrixFromLLB('O', DONT_UPDATE_TRAFFIC);

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

	return 0;
}

// Gets A and B LLB tile and schedules them
// Each processing element will get one basic tile of A and B
//	in an A-stationary fashion
void Scheduler_9_drt::ScheduleBottomSDOT(int i_start_top, int i_end_top,
		int j_start_top, int j_end_top, int k_start_top, int k_end_top){

	uint64_t max_time_accessed_in_batch = 0, min_time_pe_progressed = 0, LogWriteBackSize= 0;
	int output_nnz = 0;
	// Keeps the output size that all O micro tiles in a macro tile have before reduce
	//	One per column of output to parallelize it
	//	(finer granularity than this will not have necessarily any benefit and slower for simulation)
	uint64_t * macro_tile_middle_write = new uint64_t[k_end_top-k_start_top];
	uint64_t * macro_tile_middle_nnz = new uint64_t[k_end_top-k_start_top];
	std::fill(macro_tile_middle_write, macro_tile_middle_write + (k_end_top-k_start_top), 0);
	std::fill(macro_tile_middle_nnz, macro_tile_middle_nnz + (k_end_top-k_start_top), 0);
	//printf("i_top:[%d:%d), j_top: [%d:%d), k_top: [%d:%d)\n",
	//	i_start_top, i_end_top, j_start_top, j_end_top, k_start_top, k_end_top);

	// Vector of traffic/comp for one micro tile of A times a micro tile of B
	int max_cont = params->getTileSize();
	uint64_t * a_middle_traffic_vec = new uint64_t[max_cont];
	uint64_t * b_middle_traffic_vec = new uint64_t[max_cont];
	uint64_t * o_middle_traffic_vec = new uint64_t[max_cont];
	uint64_t * a_elements_in_col = new uint64_t[max_cont];
	uint64_t * macc_count_vec = new uint64_t[max_cont];
	int * pe_indices_micro = new int[std::max(max_cont,k_end_top - k_start_top)];

	// The earliest time that the macro tile computation starts
	uint64_t batch_starting_cycle = *std::min_element(pe_time, pe_time+params->getPECount());

	// Initilizes the log matrix for output -> All empty tiles
	matrix->initOutputLogMatrix(i_start_top, i_end_top, k_start_top, k_end_top);

	// Loop over columns of A (outer product dataflow)
	for(int j_index_middle = j_start_top; j_index_middle < j_end_top; j_index_middle++){
		//printf("j_top: %d\n",j_index_middle);

		// vectors to keep track of A column and B row nnz micro tiles
		std::vector<int> a_indices, a_offset_idx;
		std::vector<int> b_indices, b_offset_idx;

		// Extract A Basic Tiles
		// Returns the nnz micro tiles of the j column of A in [i_start,i_end] bound
		ExtractABasicTiles(i_start_top, i_end_top, j_index_middle,
				a_indices, a_offset_idx);

		// Extract B Basic Tiles
		// Returns the nnz micro tiles of the j row of B in [k_start,k_end] bound
		ExtractBBasicTiles(j_index_middle, k_start_top, k_end_top,
					b_indices, b_offset_idx);

		// A and B fiber sizes
		int fib_a_len = a_indices.size();
		int fib_b_len = b_indices.size();

		// Early termination
		if((fib_a_len == 0) | (fib_b_len == 0))
			continue;

		// i iterations; Rows of A/O
		for(int a_idx = 0; a_idx < fib_a_len; a_idx++){
			// k iterations; Column of B/O
			for(int b_idx = 0; b_idx < fib_b_len; b_idx++){
				int i_index_middle = a_indices.at(a_idx);
				int k_index_middle = b_indices.at(b_idx);

				uint32_t top_traffic = 0, middle_traffic = 0, pproduct_size = 0;
				int cycles_comp = 0, vec_counter = 0;

				// These variables show how many nnz scalars and how much of data
				//	is used per solumn of A/row of B output log
				// At the end of the micro tile computation, output logs scalars are brought to
				//	PE to reduce/merge
				uint64_t micro_tile_middle_write = 0, micro_tile_middle_nnz = 0;

				if(matrix->a_csc_outdims->vals[a_offset_idx.at(a_idx)] == 0.0){
					matrix->a_csc_outdims->vals[a_offset_idx.at(a_idx)] = 1.0;
					// Add A tiled size to the traffic computation
					uint32_t a_tile_size;
					if(params->getAFormat() == CSX::COO)
						a_tile_size = matrix->getCOOSize('A', i_index_middle, j_index_middle);
					else if(params->getAFormat() == CSX::CSF)
						a_tile_size = matrix->getCSFSize('A', i_index_middle, j_index_middle);
					deb_a_traffic += a_tile_size;
					top_traffic += a_tile_size;
				}

				// Bring the B tile if it is not in the LLB already
				if(matrix->b_csr_outdims->vals[b_offset_idx.at(b_idx)] == 0.0){
					matrix->b_csr_outdims->vals[b_offset_idx.at(b_idx)] = 1.0;
					// If need to load B tiles add them to the memory traffic usage
					uint32_t b_tile_size;
					if(params->getAFormat() == CSX::COO)
						b_tile_size = matrix->getCOOSize('B', j_index_middle, k_index_middle);
					else if(params->getAFormat() == CSX::CSF)
						b_tile_size = matrix->getCSFSize('B', j_index_middle, k_index_middle);
					deb_b_traffic += b_tile_size;
					top_traffic += b_tile_size;
				}

				// Do the actual calculation
				matrix->CSRTimesCSR(i_index_middle, j_index_middle,
						k_index_middle,	&cycles_comp, pproduct_size);

				// Shows the amount of partial products that are written to LLB
				//	These pproducts should be brought to PE one more time to be reduced
				macro_tile_middle_write[k_index_middle-k_start_top] += pproduct_size;
				macro_tile_middle_nnz[k_index_middle-k_start_top] += (uint64_t)cycles_comp;

				// Multiply two basic tiles according to OuterSpace dataflow
				//	and report traffic and comp cycles
				// According to outerSpace, each scalar of A should be assigned to a PE.
				//	In other words, parallelization is over A scalars in a column of A
				//	micro tile

				ScheduleBasicTileMult(i_index_middle, j_index_middle, k_index_middle, pproduct_size,
					a_middle_traffic_vec, b_middle_traffic_vec, o_middle_traffic_vec, a_elements_in_col,
				 	macc_count_vec, vec_counter);

				// Add up the busy cycles; Used for starts and also sanity check
				stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);
				// Get the size of output partial products in the LLB memory
				//	If they are more than the limit then write-back logs to main memory
				uint64_t output_size = matrix->getOutputLogNNZCOOSize();
/*
				// convert COO to CSR
				uint64_t output_size = AccumulateSize('L', 0, i_end_top - i_start_top,
					0, k_end_topd - k_start_top, params->getOFormat());
*/
				if(!llb->DoesFitInLLB('O', output_size)){
					// first write back ouput to the DRAM then add the outsanding memory write
					llb->AddToLLB('O', Req::write, output_size, DONT_UPDATE_TRAFFIC);
					llb->EvictMatrixFromLLB('O', UPDATE_TRAFFIC);
					// update the total # of non-zeros for adaptive LLB partitioning
					output_nnz += matrix->getOutputLogNNZCount();
					// Evict all the tiles available in top memory (LLB)
					//  Let's do an experiment! I will evict it partially (up to a threshold)
					matrix->evictOutputLogMatrix();
					// Updating the size of write-back for pproducts
					//	This value should be read-back into top memory (LLB) later for merge
					LogWriteBackSize += output_size;
					top_traffic += output_size;
					deb_o_traffic_write += output_size;
				}

				// The starting time should be batch_starting_cycle because
				//  pre-fetching can happen
				//printf("start time (1): %lu\n", batch_starting_cycle);
				uint64_t endingCycle_top = updateBWLog(batch_starting_cycle,
							top_traffic, top_bw_logger, top_bytes_per_ns);

				// Update the Processing element time loggers and middle NoC bandwidth logger for
				//		each A scalar in the micro tile
				// Note -> The reason that I am going through all the torture for this useless fine-grained
				//	scheduling is to make it extremely similar to OuterSpace!
				// Iterate through all the columns of the micro tile


				for(int a_inner_col =0; a_inner_col < vec_counter; a_inner_col++ ){
					// Figure out what PE units are going to be assigned for the task
					PickPEsAccordingToPolicy(a_elements_in_col[a_inner_col], pe_indices_micro);

					// Calculate middle traffic used for one inner row of B micro tile
					// Please note all the scalars in the A column will produce the same traffic and MACC ops.
					//	However, they should be handled separately because of different times of the PE unit
					//	they are assigned to.
					middle_traffic = a_middle_traffic_vec[a_inner_col] +
						b_middle_traffic_vec[a_inner_col] + o_middle_traffic_vec[a_inner_col];

					// Iterate through all the A scalars in the inner column ID
					for(int a_inner_row =0; a_inner_row < (int)a_elements_in_col[a_inner_col]; a_inner_row++){

						// Log the output write size from pe to LLB! These nnz elements should be brought to
						//		PE one more time and be merged(correct term: reduced)
						micro_tile_middle_write += o_middle_traffic_vec[a_inner_col];
						// Difference in number of nnz in the pre-merged log and final PE res
						//	will show how many cycles will be spent for accumulation
						micro_tile_middle_nnz += macc_count_vec[a_inner_col];

						// Chose the PE that is going to be updated
						uint64_t * chosen_pe;
						if((params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed) |
							(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed_ct)){
							chosen_pe = std::min_element(pe_time, pe_time+params->getPECount());
						}
						// if not oracle_relaxed or oracle_relaxed_ct (normally the else condition is chosen)
						else{
							chosen_pe = &pe_time[pe_indices_micro[a_inner_row]];
						}
						// For the middle DOT we should start when computation takes place
						uint64_t endingCycle_middle = 0;
						// Update the middle DOT traffic for the A scalar
						if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
							endingCycle_middle = updateBWLog(*chosen_pe,
								middle_traffic, middle_bw_logger, middle_bytes_per_ns);
						}
						// Update the PE logger time
						*chosen_pe = std::max( std::max(endingCycle_top, endingCycle_middle)
									,*chosen_pe + macc_count_vec[a_inner_col]);

						// PE time becomes the maximum of pe_time, top_bw_logger, and middle_bw_logger
						uint64_t max_time_accessed_in_batch = std::max(max_time_accessed_in_batch,
								std::max(*chosen_pe, endingCycle_top));
						// PE utilization logger update
						int pe_id = std::distance(pe_time, chosen_pe);
						// Update PE utilization logger; Debugging purposes
						pe_utilization_logger[pe_id] += macc_count_vec[a_inner_col];

						if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
							nnz_counts[pe_indices_micro[a_inner_row]] +=	(1 + macc_count_vec[a_inner_col]	);
						} // if nnz_based

					} // for a_inner_row
				} // for a_inner_col

				/********************************************************/
				/****** First reduction -> per micro-tile reduction *****/
				// Now that the micro-tile computation is over, first bring the partial producuts
				//	one by one and merge (reduce) them

				// 1- Figure out what PE units are going to be assigned for the task
				PickPEsAccordingToPolicy(1, pe_indices_micro);
				// 2- Chose the PE that is going to be updated
				uint64_t * merge_pe;
				if((params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed) |
					(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed_ct)){
					merge_pe = std::min_element(pe_time, pe_time+params->getPECount());
				}
				// if not oracle_relaxed or oracle_relaxed_ct (normally the else condition is chosen)
				else{
					merge_pe = &pe_time[pe_indices_micro[0]];
				}
				// 3- Update Middle NoC traffic
				uint64_t endingCycle_middle = 0;
				// micro_tile_middle_write -> need to be read back into PE
				// pproducr_size -> The final size that is going to be written back to LLB
				uint64_t reduce_traffic = micro_tile_middle_write + pproduct_size;
				// Update middle NoC O read/write stats
				stats->Accumulate_o_write_middle(pproduct_size);
				stats->Accumulate_o_read_middle(micro_tile_middle_write);
				// Update the middle DOT traffic for merging O micro tile
				if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
					endingCycle_middle = updateBWLog(*merge_pe,	reduce_traffic,
							middle_bw_logger, middle_bytes_per_ns);
				}
				// 4- Update the PE logger time
				uint64_t reduce_accumulations = micro_tile_middle_nnz - (uint64_t)cycles_comp;
				*merge_pe = std::max( std::max(endingCycle_top, endingCycle_middle)
									,*merge_pe + reduce_accumulations);

				// PE time becomes the maximum of pe_time, top_bw_logger, and middle_bw_logger
				uint64_t max_time_accessed_in_batch = std::max(max_time_accessed_in_batch, *merge_pe);
				// PE utilization logger update
				int pe_id = std::distance(pe_time, merge_pe);
				// Update PE utilization logger; Debugging purposes
				pe_utilization_logger[pe_id] += reduce_accumulations;

			} // for b_idx
		} // for a_idx
	} // for j_index_middle
	/****** Loop back to the next column of A and row of B *******/

	/*************************************************************/
	/****** Second reduction -> per the macro-tile reduction *****/
	// Final reduce for the Macro level
	// Now that the macro-tile computation is over, first bring the partial producuts
	//	one by one and merge (reduce) them

	// 1- Figure out what PE units are going to be assigned for the task
	PickPEsAccordingToPolicy(k_end_top - k_start_top, pe_indices_micro);
	for(int k_index = 0; k_index < k_end_top - k_start_top; k_index++){
		// 2- Chose the PE that is going to be updated
		uint64_t * merge_pe;
		if((params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed) |
			(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed_ct)){
			merge_pe = std::min_element(pe_time, pe_time+params->getPECount());
		}
		// if not oracle_relaxed or oracle_relaxed_ct (normally the else condition is chosen)
		else{
			merge_pe = &pe_time[pe_indices_micro[k_index]];
		}
		// 3- Update Middle NoC traffic
		uint64_t endingCycle_middle = 0;
		// micro_tile_middle_write -> need to be read back into PE
		// write_to_LLB_traffic -> The final size that is going to be written back to LLB
		uint64_t macro_write_to_LLB_traffic =	AccumulateSize('L', i_start_top, i_end_top,
				k_start_top+k_index, k_start_top + k_index + 1, params->getOFormat());
		uint64_t reduce_traffic = macro_tile_middle_write[k_index] + macro_write_to_LLB_traffic;
		// Update middle NoC O read/write stats
		stats->Accumulate_o_write_middle(macro_write_to_LLB_traffic);
		stats->Accumulate_o_read_middle(macro_tile_middle_write[k_index]);
		// Update the middle DOT traffic for merging O micro tile
		if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
			endingCycle_middle = updateBWLog(*merge_pe,	reduce_traffic,
					middle_bw_logger, middle_bytes_per_ns);
		}
		// 4- Update the PE logger time
		uint64_t macro_nnz_count = AccumulateNNZ('L', i_start_top, i_end_top,
				k_start_top + k_index, k_start_top + k_index+ 1);
		uint64_t reduce_accumulations = macro_tile_middle_nnz[k_index] - macro_nnz_count;
		*merge_pe = std::max(endingCycle_middle	,*merge_pe + reduce_accumulations);

		// PE time becomes the maximum of pe_time, top_bw_logger, and middle_bw_logger
		uint64_t max_time_accessed_in_batch = std::max(max_time_accessed_in_batch, *merge_pe);
		// PE utilization logger update
		int pe_id = std::distance(pe_time, merge_pe);
		// Update PE utilization logger; Debugging purposes
		pe_utilization_logger[pe_id] += reduce_accumulations;
	}

	// In case we have not reached the last row of B (j_end)
	//	we need to write back the last batch of log to main memory
	// What is the else situation? Data will be written to main memory
	//	without fetch and reduce. This has been taken care of at the Run function
	//	through topLevel_output_bw variable!
	if(j_end_top != b_tiled_rows){
		uint64_t writeback_log_size_last = matrix->getOutputLogNNZCOOSize();
		if(writeback_log_size_last){
			LogWriteBackSize += writeback_log_size_last;
			deb_o_traffic_write += writeback_log_size_last;
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
	// Delete the Log matrix created in top memory
	matrix->deleteOutputLogMatrix();
	// The amount of data that will be read-back to top memory later
	totalSize_outputCOOLog += LogWriteBackSize;
	// Adaptively update the top memory (LLB) partitioning A to O ratio
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

	}

	delete [] macro_tile_middle_write;
	delete [] macro_tile_middle_nnz;
	delete [] a_middle_traffic_vec;
	delete [] b_middle_traffic_vec;
	delete [] o_middle_traffic_vec;
	delete [] a_elements_in_col;
	delete [] macc_count_vec;
	delete [] pe_indices_micro;

	return;
}

// Iterates over the A LLB tile columns and finds all the basic tiles
void Scheduler_9_drt::ExtractABasicTiles(int i_start_top, int i_end_top, int j_start_middle,
		std::vector<int> & a_indices_middle, std::vector<int> & a_offset_idx){

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

	for(int * it = vecA_begin; it<vecA_end; it++, offset_idx++){
		int i_idx_middle = *it;
		// Early termination: All the tiles are included, get out of the for loop
		if( i_idx_middle >=upper_bound ){
			break;
		}
		a_indices_middle.push_back(i_idx_middle);
		a_offset_idx.push_back(offset_idx);
	}
	return;
}

// Iterates over the B LLB tile rows and finds all the basic tiles
void Scheduler_9_drt::ExtractBBasicTiles(int j_index_middle, int k_start_top, int k_end_top,
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

// Multiplies one basic tile (micro tile) to another one and measures the bandwidth it needs
void Scheduler_9_drt::ScheduleBasicTileMult(
		int i_idx, int j_idx, int k_idx, uint64_t finalProductSize,
	 	uint64_t * a_traffic_vec, uint64_t * b_traffic_vec,
		uint64_t * o_traffic_vec, uint64_t * a_elements_in_col,
		uint64_t * macc_count_vec, int & vec_counter){


	uint64_t a_traffic_per_pe, b_traffic_per_pe, o_traffic_per_pe, macc_count_per_pe; int a_elements;
	uint64_t total_a_traffic = 0, total_b_traffic = 0, total_o_traffic = 0,	total_maccs = 0;

	vec_counter = 0;
	// iterate over rows
	for(int j_micro_tile_idx=0; j_micro_tile_idx< params->getTileSize(); j_micro_tile_idx++){
		// This function just looks at the numbers and does not do actual multiplication
		// The returned numbers say how much traffic is used per one A scalar and how many
		//		A scalars are available in the column
		matrix->OuterProduct(i_idx, j_idx, k_idx, j_micro_tile_idx,
				a_traffic_per_pe, b_traffic_per_pe, o_traffic_per_pe, macc_count_per_pe, a_elements);

		// it records how much traffic is used per PE unit
		// Please note that all the PEs that take different A scalars of the same column
		//		will have the same exact traffic numbers!
		a_traffic_vec[vec_counter] = a_traffic_per_pe;
		b_traffic_vec[vec_counter] = b_traffic_per_pe;
		o_traffic_vec[vec_counter] = o_traffic_per_pe;
		macc_count_vec[vec_counter]  = macc_count_per_pe;
		a_elements_in_col[vec_counter] = a_elements;
		// vec_counter shows how many elements of the vec is used
		// Please don't use stl datastructures (vectors, ...) since MKL messes up with them
		vec_counter++;

		total_a_traffic += (a_traffic_per_pe*a_elements);
		total_b_traffic += (b_traffic_per_pe*a_elements);
		total_o_traffic += (o_traffic_per_pe*a_elements);
		total_maccs += (macc_count_per_pe*a_elements);
	}

	// Calculate data read and write traffics for output
	// There is no reduction in this phase! It will happen when later
	//	in the ScheduleBottomSDOT function later on in two phases:
	//		1- Reduction per output micro tile
	//		2- Reduction per output macro tile
	uint64_t pe_o_read_traffic = 0;
	uint64_t pe_o_write_traffic = total_o_traffic;

	// Update middle NoC read/write stats
	stats->Accumulate_a_read_middle(total_a_traffic);
	stats->Accumulate_b_read_middle(total_b_traffic);
	stats->Accumulate_o_write_middle(pe_o_write_traffic);
	stats->Accumulate_o_read_middle(pe_o_read_traffic);
}
// Gets the row boundary and start address of matrix A rows,
//   returns the start and stop address of the matrix A cols, i.e., o_reuse
// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
void Scheduler_9_drt::ExtractATopTile(int i_idx_start, int j_idx_start){

	int b_reuse = params->getAReuse();
	int j_idx_stop = j_idx_start;
	int i_idx_end = std::min(i_idx_start+b_reuse, o_tiled_rows);

	// In static tiling case we already now what the j_idx_end
	//	should be and have set the OReuse value in params
	if(params->getTilingMechanism() == tiling::t_static){
		int j_idx_end = std::min(j_idx_start + params->getOReuse(),
				b_tiled_rows);
		uint64_t extra_size = AccumulateSize('A', i_idx_start, i_idx_end,
				j_idx_start, j_idx_end, params->getAFormat());
		llb->AddToLLB('A', Req::read, extra_size, UPDATE_TRAFFIC);
	}
	// Use DRT to figure out j_idx_end
	else if(params->getTilingMechanism() == tiling::t_dynamic){
		// Add rows until it either runs out of memory or reaches the last row
		for(int j_idx = j_idx_start; j_idx<b_tiled_rows; j_idx++){
			// Find the size of the new col
			uint64_t extra_size = AccumulateSize('A', i_idx_start, i_idx_end,
					j_idx, j_idx+1, params->getAFormat());
			// It means that it could not fit the new row in LLB and failed
			if(llb->DoesFitInLLB('A', extra_size) == 0) {break;}
			llb->AddToLLB('A', Req::read, extra_size, UPDATE_TRAFFIC);
			j_idx_stop++;
		}

		// if the if statements is correct, it means A partition of LLB still has room
		// Thus, let's increase the a_reuse value
		if((j_idx_start == 0) & (j_idx_stop == b_tiled_rows)){
			while(i_idx_start+b_reuse < o_tiled_rows){
				uint64_t extra_size = AccumulateSize('A', i_idx_start+b_reuse,
						i_idx_start+b_reuse+1, j_idx_start, j_idx_stop, params->getAFormat());
				// It means that it could not fit the new row in LLB and failed
				if(llb->DoesFitInLLB('A', extra_size) == 0) {break;}
				llb->AddToLLB('A', Req::read, extra_size, UPDATE_TRAFFIC);
				b_reuse++;
			}
			params->setAReuse(b_reuse);
		}
		params->setOReuse(j_idx_stop- j_idx_start);
	}
	else{
		printf("No Such Tiling Mechanism is Available!\n"); exit(1);
	}
	return;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the size that block of tiles would occupy in LLB
uint64_t Scheduler_9_drt::AccumulateSize(char mat_name, int d1_start, int d1_end,
	 	int d2_start, int d2_end, CSX inp_format){

	uint64_t size_tot = 0;
	switch (mat_name){
		case 'A':{
			// Use CSC representation
			if(d2_end == (d2_start+1)){
				if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize_sparse('A',d2_start, d1_start, d1_end, CSX::CSC);
				else if(inp_format == CSX::COO)
					size_tot += matrix->accumulateCOOSize_sparse('A',d2_start, d1_start, d1_end, CSX::CSC);
				else{printf("A format not supported!\n"); exit(1);}
			} // if
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++){
					if(inp_format == CSX::CSF)
						size_tot += matrix->accumulateCSFSize_sparse('A',i_idx, d2_start, d2_end, CSX::CSR);
					else if(inp_format == CSX::COO)
						size_tot += matrix->accumulateCOOSize_sparse('A',i_idx, d2_start, d2_end, CSX::CSR);
					else{printf("A format not supported!\n"); exit(1);}
				} // for i_idx
			} // else
			break;
		} // case 'A'

		case 'B':{
			// Use CSC representation
			if(d2_end == (d2_start+1)){
				if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize_sparse('B',d2_start, d1_start, d1_end, CSX::CSC);
				else if(inp_format == CSX::COO)
					size_tot += matrix->accumulateCOOSize_sparse('B',d2_start, d1_start, d1_end, CSX::CSC);
				else{printf("B format not supported!\n"); exit(1);}
			}
			// Use CSR representation
			else{
				for(int i_idx=d1_start; i_idx<d1_end; i_idx++){
					if(inp_format == CSX::CSF)
						size_tot += matrix->accumulateCSFSize_sparse('B',i_idx, d2_start, d2_end, CSX::CSR);
					else if(inp_format == CSX::COO)
						size_tot += matrix->accumulateCOOSize_sparse('B',i_idx, d2_start, d2_end, CSX::CSR);
					else{printf("B format not supported!\n"); exit(1);}
				} // for i_idx
			} // else
			break;
		} // case 'B'

		case 'O':{
			for(int i_idx=d1_start; i_idx<d1_end; i_idx++){
				if(inp_format == CSX::COO)
					size_tot += matrix->accumulateCOOSize('O',i_idx, d2_start, d2_end);
				else if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize('O',i_idx, d2_start, d2_end);
				else{printf("Format not supported!\n"); exit(1);}
			} // for
			break;
		} // case 'O'

		// Output log file
		case 'L':{
			for(int i_idx= 0; i_idx < d1_end- d1_start; i_idx++){
				if(inp_format == CSX::COO)
					size_tot += matrix->accumulateCOOSize('L',i_idx, 0, d2_end-d2_start);
				else if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize('L',i_idx, 0, d2_end-d2_start);
				else{printf("Format not supported!\n"); exit(1);}
			} // for i+idx
			break;
		} // case 'L'

		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}

	return size_tot;
}

// Gets the start and end address of both dimensions of either matrices (A, B, and O)
//   and returns the total number of non-zero elements in that range
uint64_t Scheduler_9_drt::AccumulateNNZ(char mat_name, int d1_start, int d1_end,
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
		case 'L':{
			for(int i_idx=0; i_idx < d1_end - d1_start; i_idx++)
				nnz_tot += matrix->accumulateNNZ('L',i_idx, d2_start, d2_end, CSX::CSR);
			break;}

		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}
	return nnz_tot;
}

// Find out what is the B col sizes we need to fetch
//	after LLB intersection
void Scheduler_9_drt::PreCalculateBColsSize(int i_start, int i_end,
		int j_start, int j_end){

	// Finding whether any col of A is empty or not
	int * a_llb_verticalSum = new int[o_tiled_rows];
	// Calculate the horizontal sum of B PE tiles in the LLB tile
	//   This sum is used to decide whether to bring PE tiles of A
	//   It basically says whether there is any nnz PE tile in each row
	CalcALLBVerticalSum(i_start, i_end,
		j_start, j_end, a_llb_verticalSum);

	#pragma omp parallel for
	// preCalculated_BColSize is a DRT related global variable!
	// It works for both MatRaptor and OuterSpace because they are A-stationary
	for(int k_idx = 0; k_idx < o_tiled_cols; k_idx++){
		preCalculated_BColSize[k_idx] = AccumulateSize_BwrtA(j_start, j_end,
				k_idx, a_llb_verticalSum);
	}
	delete [] a_llb_verticalSum;

	return;
}

// Find vertical sum of A tiles in LLB; this is used in
//   deciding whether to bring B PE tiles to LLB or not
// If vertical sum of the col is zero then do not bring the
//   corresponding PE tile of B
void Scheduler_9_drt::CalcALLBVerticalSum(int i_start, int i_end,
		int j_start, int j_end, int * a_llb_verticalSum){

	for(int j_idx = j_start; j_idx<j_end; j_idx++){
		a_llb_verticalSum[j_idx] = (AccumulateSize('A', i_start, i_end,
				j_idx, j_idx+1, params->getAFormat()) > 0) ? 1 : 0;
	} // for j_idx

	return;
}

// Finds the size of B Macro tile wrt A macro tile
//  looks at the A tiles in the LLB to see whether
//  each B micro tile should be loaded or not
uint64_t Scheduler_9_drt::AccumulateSize_BwrtA(int j_start, int j_end,
		int k_idx, int *a_llb_verticalSum){

	uint64_t size = 0;
	// Iterate over the row idxs of a specific column of B (e.g., k_idx)
	for(int t_idx = matrix->b_csc_outdims->pos[k_idx];
		 	t_idx < matrix->b_csc_outdims->pos[k_idx+1]; t_idx++){

		int j_idx = matrix->b_csc_outdims->idx[t_idx];
		// If the col idx is smaller than start skip
		if(j_idx<j_start) continue;
		// If the col idx matches the range, then add the size
		else if(j_idx<j_end){
			if(a_llb_verticalSum[j_idx]){
				if(params->getBFormat() == CSX::CSF)
					size += matrix->getCSFSize('B', j_idx, k_idx);
				else if(params->getBFormat() == CSX::COO)
					size += matrix->getCOOSize('B', j_idx, k_idx);
				else{printf("B format not supported!\n"); exit(1);}
			}
		}
		// If the col idx is larger than the max, get out. You are done soldier Svejk
		else break;
	}
	return size;
}


// ExtractBTopTiles gets the dimensions of the B LLB tile and grows number of cols
// until filling up the top buffer (LLB) for tensor B
//	Tasks: 1) Report k_end_top 2) Fetch B top tiles into LLB buffer
void Scheduler_9_drt::ExtractBTopTile(int i_start_top, int i_end_top,
		int j_start_top, int j_end_top, int k_start_top, int &k_end_top){

	// In the static tiling mechasnism we already know what k_idx_end
	//	should be since it is constant!
	if(params->getTilingMechanism() == tiling::t_static){
		k_end_top = std::min(o_tiled_cols, k_start_top + K_top_tile);
		uint64_t b_col_size = 0;
		for(int k_idx_top = k_start_top; k_idx_top < o_tiled_cols; k_idx_top++){
			b_col_size += preCalculated_BColSize[k_idx_top];
		}
		llb->AddToLLB('B', Req::read, b_col_size, UPDATE_TRAFFIC);
	}
	// Use DRT to figure out what k_idx_end should be
	else if(params->getTilingMechanism() == tiling::t_dynamic){
		k_end_top = k_start_top;
		// Go over every row until reaching the maximum partition allowance
		for(int k_idx_top = k_start_top; k_idx_top < o_tiled_cols; k_idx_top++ ){
			uint64_t b_col_size = preCalculated_BColSize[k_idx_top];
			if (llb->DoesFitInLLB('B', b_col_size)){
				llb->AddToLLB('B', Req::read, b_col_size, UPDATE_TRAFFIC);
				k_end_top++;
			}
			else{break;}
		}
	}
	else{
		printf("No Such Tiling Mechanism is Available!\n"); exit(1);
	}
	return;
}

void Scheduler_9_drt::PickPEsAccordingToPolicy(
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
		for(int pe_idx = 0; pe_idx < count_needed; pe_idx++){
			pe_indices.push_back(pe_idx%params->getPECount());
		}
	}
	else if(dist_policy == static_distributor::nnz_based){
		QuickSort3Desc(nnz_counts, pe_time, pe_utilization_logger, 0, params->getPECount()-1);
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

void Scheduler_9_drt::PickPEsAccordingToPolicy(
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
	else if(dist_policy == static_distributor::oracle_relaxed){}
	else if(dist_policy == static_distributor::oracle_relaxed_ct){}
	else{
		printf("Static Scheduler Policy Is Not Available!\n");
		exit(1);
	}

	return;
}

// Sorting a, b, and c according to a (b and c are dependent)
// This function is used for finding to corresponding PE units, and pe_utilization_logger
//	a is the nnz count for each pe, and pe is the pe progress time
template<class A, class B, class C> void Scheduler_9_drt::QuickSort3Desc(A a[], B b[], C c[], int l, int r)
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
template<class A, class B> void Scheduler_9_drt::QuickSort2Desc(A a[], B b[], int l, int r)
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
void Scheduler_9_drt::Swap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

// This is a beautiful bw logger that gets the start cycle
//   of each tile multiplication and in a cycle accurate way says
//   when the data transfer is finished
// Its role is to keep track of bandwidth either for the top level
//  or middle level, depending on the bw_logger and bytes_per_ns assignment
uint64_t Scheduler_9_drt::updateBWLog(uint64_t starting_cycle, uint64_t action_bytes,
		float *bw_logger, float bytes_per_ns){

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

void Scheduler_9_drt::PrintBWUsage(){
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

	deb_total_traffic = deb_a_traffic + deb_b_traffic + deb_o_traffic_read + deb_o_traffic_write;
	printf("total_traffic %lu, a_read %lu, b read %lu, o_read %lu, o_write %lu\n",
			deb_total_traffic, deb_a_traffic, deb_b_traffic, deb_o_traffic_read, deb_o_traffic_write);

	return;
}



#endif

