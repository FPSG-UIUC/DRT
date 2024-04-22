#ifndef SCHEDULER_10_DRT_CPP
#define SCHEDULER_10_DRT_CPP

// Two level OuterSpace + Dynamic Reflexive Tiling
// The levels are Top and Bottom SDOTs, where Top SDOT brings "Micro Tiles" in
//	A-stationary fashion for both top and bottom SDOTs
//	B-stationary fashion and the bottom SDOT receives scalaras in A-stationary fashion.

#include "scheduler_10_drt.h"

// constructor -> intializer of the scheduler
Scheduler_10_drt::Scheduler_10_drt(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
	this->matrix = mtx;
	this->params = params;
	this->stats = stats;
	this->llb = llb;

    //debugger = new Debugger();
    //debugger->setDebug(2);
	
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

	pe_utilization_logger = new uint64_t[params->getPECount()];
	std::fill(pe_utilization_logger, pe_utilization_logger+params->getPECount(), 0);

	// Pre-calculate A row sizes corresponding to B LLB tile
	//	that will be used in different parts of code
	//vecCommittedRows_ASize = new uint64_t[o_tiled_rows];
	preCalculated_BColSize = new uint64_t[o_tiled_cols];

	pe_utilization_logger = new uint64_t[params->getPECount()];
	std::fill(pe_utilization_logger, pe_utilization_logger+params->getPECount(), 0);

	// DEBUGGING variables. DON'T REMOVE
	deb_a_traffic = 0; deb_b_traffic = 0; deb_o_traffic_read = 0; deb_o_traffic_write = 0; deb_total_traffic = 0;


	return;
}

// Destructor -> delete [] bunch of dynamically allocated arrays
Scheduler_10_drt::~Scheduler_10_drt(){
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
void Scheduler_10_drt::Reset(){

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
 * This is the version where we're duplicating the baseline dataflow
 * to the outer dimensions
 * i_bot_size = num_I_microtiles_in_I_bot_tile
 * for (I_top = 0; I_top < It; I_top++)
 *  for (J_top = 0; J_top < Jt; J_top++)
 *   for (K_top = 0; K_top < Kt; K_top++)
 *
 *       for(I_bottom = 0; I_bottom<a_micro_rows; i_bottom++)
 *         for (j_bottom = 0; j_bottom<a_micro_cols; j_bottom++)
 *          for (k_bottom = 0; k_bottom<b_micro_cols; k_bottom++)
 *
 *            for (i_inner = 0; i_inner < 32; i_inner++)
 *             for (j_inner = 0; j_inner < 32; j_inner++)
 *              for (k_inner = 0; k_inner < 32; k_inner++)
 *                i = I_top*I_top_size + i_bottom*i_bot_size + i_inner
 *                j = J_top*J_top_size + j_bottom*j_bot_size + j_inner
 *                k = K_top*K_top_size + k_bottom*k_bot_size + k_inner
 *                a[i][k] += a[i][j] * b[j][k]
 *
 */
int Scheduler_10_drt::Run(){

	// If we have a static tiling mechanism (ExTensor style)
	//  then do some initialization
	if(params->getTilingMechanism() == tiling::t_static){
		// I, J, and K of the LLB (Top DOT) tiles
        // Coordinates are in terms of microtiles
		I_top_tile = params->getITopTile();
		J_top_tile = params->getJTopTile();
		K_top_tile = params->getKTopTile();

		// O_reuse and b_reuse are constant values in static tiling
        // TO: We're A-stationary so we want to set the reuse we want out of B
        // which depends on the number of rows of A included in the LLB
        // the larger I_top_tile is, the better the reuse on B 
        // Output reuse works the same way -- the larger J_top_tile is, the
        // lower the number of times we need to bring the O tile into LLB 
		params->setBReuse(I_top_tile);
		params->setOReuse(J_top_tile);
		printf("I_top: %d, J_top: %d, K_top: %d\n", I_top_tile, J_top_tile, K_top_tile);
        //printf("Toluwa DEBUG: In static tiling\n");
//	fflush(stdout);

	}

	int j_end_top =0, i_end_top =0, k_end_top=0;
	llb_reset_count = 0; round_robin_slot = 0;
	int b_reuse = params->getBReuse();
	b_reuse_const = b_reuse;
	//printf("b_reuse ddis %d\n", b_reuse);
    //debugger->print("B reuse is: ", 2);
    //debugger->print(std::to_string(b_reuse).c_str(), 2);
    

	totalSize_outputCOOLog = 0;
    for(int i_start_top = 0; i_start_top < o_tiled_rows;)
    {
        //j_end_top = 0;
        for(int j_start_top = 0; j_start_top < b_tiled_rows;) 
        {
            //Calculate O_reuse and fetch A tiles using DRT
            ExtractATopTile(i_start_top, j_start_top);
            
            //In case b_reuse has been updated by DRT, update it
            b_reuse = params->getBReuse();

            //Get the end of i and j iterators based on ExtractATopTile
            //b_reuse is updated by ExtractATopTile if there was still space
            // to grow in the I dimension (after J dimension is covered)
            //OReuse is also updated and it refers to the contracted dimension
            i_end_top = std::min(i_start_top+b_reuse, o_tiled_rows);
            j_end_top = std::min(j_start_top + params->getOReuse(), b_tiled_rows);  

            //After intersection in the LLB, what is the size of the B cols we
            //need to fetch?
            //This updates the array preCalculated_BColSize which has k elements. 
            //Each k element contains the size in bytes? of that column of B 
            PreCalculateBColsSize(i_start_top, i_end_top, j_start_top, j_end_top);
            for(int k_start_top=0; k_start_top < o_tiled_cols;)
            {
                //We've brought in the A top tile to the LLB
                //Time to bring in the B top tile
                //This function updates k_end_top
                ExtractBTopTile(i_start_top, i_end_top, 
                                j_start_top, j_end_top,
                                k_start_top, k_end_top);

                //Schedule multiplication of microtiles of the A top tile in LLB
                //to microtiles of the B top tile in LLB
                //This function contains the two lower level loop nests
                ScheduleBottomSDOT(i_start_top, i_end_top, 
                                   j_start_top, j_end_top,
                                   k_start_top, k_end_top);

                //In static tiling the sizes should be accurate and we should
                // never face oversized
                //TODO: Toluwa - I don't understand the point of this
				if(llb->GetSize() > llb->GetCapacity()){
					printf("LLB Size is not Enough!\
							(This message should be shown only in static tiling)\n");
					exit(1);
				}

                // Evict the B LLB tile to bring in another B LLB tile since 
                // we're A stationary                 
                //this array below lets us know which nnz microtiles of B have been processed
				llb->EvictMatrixFromLLB('B', DONT_UPDATE_TRAFFIC);
				llb->EvictMatrixFromLLB('O', DONT_UPDATE_TRAFFIC);
				std::fill(matrix->b_csr_outdims->vals,
					matrix->b_csr_outdims->vals + matrix->b_csr_outdims->nnz, 0.0);
                
                //update k_start_top!
                k_start_top = k_end_top;

            } // K
            
            //we've completed this set of B tiles for J
            //Time to bringin a new A tile
            llb->EvictMatrixFromLLB('A', DONT_UPDATE_TRAFFIC);
            std::fill(matrix->a_csr_outdims->vals,
                      matrix->a_csr_outdims->vals +
                        matrix->a_csr_outdims->nnz, 0.0);

            //Load back the b_reuse value
            params->setBReuse(b_reuse_const);

            //update j_start_top!
            j_start_top = j_end_top;
            
        }//j_start_top
    
        i_start_top = i_end_top;
    } //i_start_top


    //Same as outerspace
	// The output file that is going to be written to output at the last phase after reduced!
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
void Scheduler_10_drt::ScheduleBottomSDOT(int i_start_top, int i_end_top,
		int j_start_top, int j_end_top, int k_start_top, int k_end_top){

	uint64_t max_time_accessed_in_batch = 0; 
    uint64_t min_time_pe_progressed = 0;
    uint64_t LogWriteBackSize= 0;

	// Keeps the output size that all O micro tiles in a macro tile have before reduce
	//	One per column of output to parallelize it
	//	(finer granularity than this will not have necessarily any benefit and slower for simulation)
	uint64_t * macro_tile_middle_write = new uint64_t[k_end_top-k_start_top];
	uint64_t * macro_tile_middle_nnz = new uint64_t[k_end_top-k_start_top];
	std::fill(macro_tile_middle_write, macro_tile_middle_write + (k_end_top-k_start_top), 0);
	std::fill(macro_tile_middle_nnz, macro_tile_middle_nnz + (k_end_top-k_start_top), 0);
	
	//uint64_t * macc_count_vec = new uint64_t[max_cont];

	int max_cont = params->getTileSize();
	int * pe_indices_micro = new int[std::max(max_cont,k_end_top - k_start_top)];

	// Only used for oracle_relaxed static distributor
	uint32_t total_top_traffic = 0,	total_middle_traffic = 0;

    //Debug
	//printf("i_top:[%d:%d), j_top: [%d:%d), k_top: [%d:%d)\n",
	//	i_start_top, i_end_top, j_start_top, j_end_top, k_start_top, k_end_top);
	//fflush(stdout);

    //this batch of LLB macrotiles
    //Get the time of the PE with the least cycles
	uint64_t batch_starting_cycle = *std::min_element(pe_time, pe_time+params->getPECount());

	uint64_t pe_traffic, pe_macc_count;

	int output_nnz = 0;

	// Initilizes the log matrix for output -> All empty microtiles
	matrix->initOutputLogMatrix(i_start_top, i_end_top, k_start_top, k_end_top);

    for(int i_index_middle = i_start_top; i_index_middle < i_end_top; i_index_middle++) {
	//printf("i_index_middle: %d\n", i_index_middle);
        //fflush(stdout); 
    
        //a_indices stores the set of A col indices (j) that are in this
        //LLB row (i_index_middle row). a_offset_idx stores the index into
        //the CSR col idx array for the starting col  
        std::vector<int> a_indices, a_offset_idx;

       //Let's get the basic tiles in A for this LLB row of A
       //Returns the nnz microtiles of the i row of A in [j_start, j_end] bound
       ExtractABasicTiles(i_index_middle, j_start_top, j_end_top, a_indices, a_offset_idx);

        // A fiber sizes -- the nnz in this LLB row of A
        int fib_a_len = a_indices.size();

        //Early termination -- this row of A is empty!
        if ((fib_a_len == 0))
            continue;


        //TODO: Do I want to assign PEs here? 
        //Assign a PE to each microtile in this A LLB row
        //This stores which PEs are going to be used.
	std::vector<int> pe_indices;
	PickPEsAccordingToPolicy(fib_a_len, pe_indices);

        //j iterations (js that correspond to a non-zero)
		for(int a_idx = 0; a_idx < fib_a_len; a_idx++){
			//printf("Toluwa DEBUG: start of this a microtile iteation %d %d\n", a_idx, fib_a_len);
			//fflush(stdout);
            //We need to get j_index middle here so we can get the corresponding B microtile
            int j_index_middle = a_indices.at(a_idx);
        
	    //b_indices stores the set of B col indices (j) that are in this
            //LLB row of B (depends on current j of A)
	    std::vector<int> b_indices, b_offset_idx;
            
		    // Let's get the basic tiles in B for this LLB row of B
            // Returns the nnz microtiles of the j row of B in [k_start, k_end] bound
		    ExtractBBasicTiles(j_index_middle, k_start_top, k_end_top,
					           b_indices, b_offset_idx);

		    int fib_b_len = b_indices.size();
		    //printf("Toluwa DEBUG: fib_b_len is %d\n", fib_b_len);

		    // Early termination -- this row of B is empty! Nothing to do.
		    if((fib_b_len == 0))
			    continue;

            
            //k iterations (ks that correspond to a non-zero)
			for(int b_idx = 0; b_idx < fib_b_len; b_idx++){
				int k_index_middle = b_indices.at(b_idx);

				uint32_t pproduct_size = 0; int cycles_comp = 0;
				uint32_t top_traffic = 0, middle_traffic = 0;
				uint64_t LogWriteBackSize_ct= 0;
				//printf("Toluwa DEBUG: start of this b microtile iteration %d %d j, k %d %d\n", b_idx, fib_b_len, j_index_middle, k_index_middle);
				//fflush(stdout);

				if(matrix->a_csr_outdims->vals[a_offset_idx.at(a_idx)] == 0.0){
					matrix->a_csr_outdims->vals[a_offset_idx.at(a_idx)] = 1.0;
					// Add A tiled size to the traffic computation
					uint32_t a_tile_size = 0;
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
					uint32_t b_tile_size = 0 ;
					if(params->getBFormat() == CSX::COO) {
						b_tile_size = matrix->getCOOSize('B', j_index_middle, k_index_middle);
                    			}
					else if(params->getBFormat() == CSX::CSF) {
						b_tile_size = matrix->getCSFSize('B', j_index_middle, k_index_middle);
                    			}

					deb_b_traffic += b_tile_size;
					top_traffic += b_tile_size;
				}

				// Do the actual calculation
				matrix->CSRTimesCSR(i_index_middle, j_index_middle,
						k_index_middle,	&cycles_comp, pproduct_size);
				//printf("Toluwa DEBUG: done with this CSRTimesCSR\n");
				//fflush(stdout);

				// Shows the amount of partial products that are written to LLB
				//	These pproducts should be brought to PE one more time to be reduced
				macro_tile_middle_write[k_index_middle-k_start_top] += pproduct_size;
				macro_tile_middle_nnz[k_index_middle-k_start_top] += (uint64_t)cycles_comp;
				//printf("Toluwa DEBUG: done updating macro_tile_middle_write and nnz %d %d\n", pproduct_size, cycles_comp);
				//fflush(stdout);
				// Multiply two basic tiles according to MatRaptor dataflow
				//	and report traffic and comp cycles
				ScheduleBasicTileMult(i_index_middle, j_index_middle, k_index_middle, pproduct_size,
						pe_traffic, pe_macc_count);
				//printf("Just scheduled a basic tile\n");
				//fflush(stdout);

				// update middle NoC traffic
				middle_traffic += pe_traffic;

				// Add up the busy cycles; Used for stats and also sanity check
				stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);

				//printf("Toluwa DEBUG - after accumulate busy cycles %d\n", cycles_comp);
				//fflush(stdout);

				// Get the size of output partial products in the LLB memory
				//	If they are more than the limit then write-back logs to main memory
				uint64_t output_size = matrix->getOutputLogNNZCOOSize();
				
				//printf("Toluwa DEBUG - THE OUTPUT SIZE IS NOW %d\n", output_size);
				//fflush(stdout);

				//TODO: Toluwa check with Hadi on this, why is it using the entire LLB macrotile?
				//uint64_t output_size = AccumulateSize('L', 0, i_end_top - i_start_top, 
				//	         0, k_end_top - k_start_top, params->getOFormat());

                		//If the matrix does not fit into the LLB...
				if(!llb->DoesFitInLLB('O', output_size)){
					//printf("Toluwa DEBUG: It didn't fit in the LLB!\n");
					//fflush(stdout);
					/*
					// this case should not happen for static tiling
					if(params->getTilingMechanism() == tiling::t_static){
						printf("LLB Size is not Enough!\n");
						exit(1);
					}
					*/

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

				//printf("Toluwa DEBUG: start of updating chosen pe\n");
				//fflush(stdout);
				uint64_t * chosen_pe;
				if((params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed) |
					(params->getStaticDistributorModelMiddle() == static_distributor::oracle_relaxed_ct)){
					chosen_pe = std::min_element(pe_time, pe_time+params->getPECount());
				}
				else{
					chosen_pe = &pe_time[pe_indices.at(a_idx)];
				}

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

				uint64_t max_time_accessed_in_batch = std::max(max_time_accessed_in_batch,
						std::max(*chosen_pe, endingCycle_top));
				// PE utilization logger update
				int pe_id = std::distance(pe_time, chosen_pe);
				pe_utilization_logger[pe_id] += (uint64_t) cycles_comp;

				if(params->getStaticDistributorModelMiddle() == static_distributor::nnz_based){
					nnz_counts[pe_indices.at(a_idx)] +=
						(matrix->getNNZOfATile('A', i_index_middle, j_index_middle) +
						 matrix->getNNZOfATile('B', j_index_middle, k_index_middle));
				}
				//printf("Toluwa DEBUG: ready to bring in another b microtile for this microtile\n");
				//fflush(stdout);
 
			} // for b_idx (k_index_middle)
			//printf("Toluwa DEBUG: ready to bring in another a microtile END LOOP\n");
			//fflush(stdout);
		} // for a_idx (j_index_middle)

        //Ok, do a reduction in the LLB per microtile
        //TODO - should I be doing a per-microtile reduction instead or is macrotile enough?
        //Below is copied and modified from Hadi's version. Decided to use macrotile reduction instead of microtile
        //reduction
 	    /*************************************************************/
	    /****** Second reduction -> per the macro-tile reduction *****/
	    // Final reduce for the Macro level
	    // Now that the macro-tile computation is over, first bring the partial producuts
	    //	one by one and merge (reduce) them
	    //	TODO : Toluwa - what about if the output was too full for LLB in the previous loop nest (that contains scheduleBasicMultTile)?
	    //	       Shouldn't we be taking reduction into account at that point as well?
	    // 1- Figure out what PE units are going to be assigned for the task
	    /*PickPEsAccordingToPolicy(k_end_top - k_start_top, pe_indices_micro);
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
	    
        	// write_to_LLB_traffic -> The final size that is going to be written back to LLB
		    uint64_t macro_write_to_LLB_traffic =	AccumulateSize('L', i_index_middle, i_index_middle+1,
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
    		uint64_t macro_nnz_count = AccumulateNNZ('L', i_index_middle, i_index_middle+1,
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

*/
	} // for i_index_middle

	/****** Loop back to the next column of A and row of B *******/
	//printf("Toluwa DEBUG: We finally reached reduction\n");
	//fflush(stdout);
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

                // write_to_LLB_traffic -> The final size that is going to be written back to LLB
                    uint64_t macro_write_to_LLB_traffic =       AccumulateSize('L', 0, i_end_top-i_start_top,
                                k_index, k_index + 1, params->getOFormat());
                    uint64_t reduce_traffic = macro_tile_middle_write[k_index] + macro_write_to_LLB_traffic;

                    // Update middle NoC O read/write stats
                stats->Accumulate_o_write_middle(macro_write_to_LLB_traffic);
                stats->Accumulate_o_read_middle(macro_tile_middle_write[k_index]);

                // Update the middle DOT traffic for merging O micro tile
                if(params->doesMiddleDOTTrafficCount() == middleDOTTrafficStatus::yes){
                        endingCycle_middle = updateBWLog(*merge_pe,     reduce_traffic,
                                        middle_bw_logger, middle_bytes_per_ns);
                    }

                // 4- Update the PE logger time
                uint64_t macro_nnz_count = AccumulateNNZ('L',0 , i_end_top-i_start_top,
                                k_index, k_index+ 1);
                uint64_t reduce_accumulations = macro_tile_middle_nnz[k_index] - macro_nnz_count;
                *merge_pe = std::max(endingCycle_middle ,*merge_pe + reduce_accumulations);

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
	//	without fetch and reduce. This is been taken care of at Run function
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

    //Update the PE times... 
    //TODO: Toluwa -- is this for synchronization purposes?
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
        	//printf("Toluwa DEBUG: prev_a_ratio %f %f\n", llb->GetARatio(), prev_a_ratio);

		// Size of A LLB tile in LLB
		uint64_t a_used_size_acc = llb->GetASize();

        //An entry in COO is the data (8 bytes) + the x and y coordinates (4 bytes each) 
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


        //TODO: Toluwa -- why do we have these limits on A?
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
	return;
}

// Iterates over the A LLB tile rows and finds all the basic tiles
void Scheduler_10_drt::ExtractABasicTiles(int i_start_middle, int j_start_top, int j_end_top,
		std::vector<int> & a_indices_middle, std::vector<int> & a_offset_idx){

	/********************************************/
	/* FWA: Find start and end of the A row vector! */
	// lower and upper bound of A_CSR
	//	find the entries between [lower_bound, upper_bound]
	int lower_bound = j_start_top; int upper_bound = j_end_top;
	// Find start and end of the A_CSR fiber positions
	int a_pos_start = matrix->a_csr_outdims->pos[i_start_middle];
	int a_pos_end   = matrix->a_csr_outdims->pos[i_start_middle+1];

	// Indexes (I_idx values of LLB column A)
	int * vecA_begin = &(matrix->a_csr_outdims->idx[a_pos_start]);
	int * vecA_end   = &(matrix->a_csr_outdims->idx[a_pos_end]);

	// Move to the starting point of row A according to lower_bound
	int offset_idx = a_pos_start;
	while ((*vecA_begin<lower_bound) & (vecA_begin<=vecA_end)) {
		vecA_begin++; offset_idx++;
	}
	/* FWA: Until here!  ************************/

	for(int * it = vecA_begin; it<vecA_end; it++, offset_idx++){
		int j_idx_middle = *it;
		// Early termination: All the tiles are included, get out of the for loop
		if( j_idx_middle >=upper_bound ){
			break;
		}
		a_indices_middle.push_back(j_idx_middle);
		a_offset_idx.push_back(offset_idx);
	}
	return;
}

// Iterates over the B LLB tile rows and finds all the basic tiles
void Scheduler_10_drt::ExtractBBasicTiles(int j_index_middle, int k_start_top, int k_end_top,
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
	// Move to the starting point of row A according to lower_bound
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
		//printf("Toluwa DEBUG: getting this macrotile (%d, %d) %d\n", j_index_middle, k_idx_middle, offset_idx);
		b_indices_middle.push_back(k_idx_middle);
		b_offset_idx.push_back(offset_idx);
	}
	return;
}

// Multiplies one basic tile (micro tile) to another one and measures the bandwidth it needs
void Scheduler_10_drt::ScheduleBasicTileMult(int i_idx, int j_idx, int k_idx, uint64_t finalProductSize,
	 	uint64_t &pe_traffic, uint64_t &pe_macc_count){

	uint64_t a_traffic, b_traffic, o_traffic, macc_count;
	uint64_t total_a_traffic = 0, total_b_traffic = 0,
					 total_o_traffic = 0,	total_traffic = 0, total_maccs = 0;
	// iterate over rows
	for(int i_micro_tile_idx=0; i_micro_tile_idx< params->getTileSize(); i_micro_tile_idx++){
		// This function just looks at the numbers and does not do actual multiplication
		matrix->RowWiseProduct(i_idx, j_idx, k_idx, i_micro_tile_idx,
				a_traffic, b_traffic, o_traffic, macc_count);

		total_o_traffic += o_traffic;
		total_a_traffic += a_traffic;
		total_b_traffic += b_traffic;
		total_maccs += macc_count;
	}

	//If we assume each microtile has local registers to merge rows per scalar, then 
	//total_o_traffic is equal to finalProductSize for this microtile computation
	total_o_traffic = finalProductSize;

	// Calculate data read and write traffics
    //TODO: does this account for the merge phase for this output microtile?
	uint64_t pe_read_traffic = total_a_traffic + total_b_traffic + total_o_traffic;

    //TODO: Toluwa -- why finalProductSize???
	uint64_t pe_write_traffic = total_o_traffic + finalProductSize;

	// Update returning variables
	pe_traffic = pe_read_traffic + pe_write_traffic;
	pe_macc_count = total_maccs;

	// Update middle NoC read/write stats
	stats->Accumulate_o_write_middle(pe_write_traffic);
	stats->Accumulate_o_read_middle(total_o_traffic);
	stats->Accumulate_a_read_middle(total_a_traffic);
	stats->Accumulate_b_read_middle(total_b_traffic);
}

// Gets the row boundary and start address of matrix A rows,
//   returns the start and stop address of the matrix A cols, i.e., o_reuse
// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
void Scheduler_10_drt::ExtractATopTile(int i_idx_start, int j_idx_start){

	int b_reuse = params->getBReuse();
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
        //printf("TOLUWA DEBUG: In Static tiling\n");
	}
	// Use DRT to figure out j_idx_end
	else if(params->getTilingMechanism() == tiling::t_dynamic){
        //printf("TOLUWA DEBUG: In dynamic tiling\n");
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
		// Thus, let's increase the b_reuse value
		if((j_idx_start == 0) & (j_idx_stop == b_tiled_rows)){
			while(i_idx_start+b_reuse < o_tiled_rows){
				uint64_t extra_size = AccumulateSize('A', i_idx_start+b_reuse,
						i_idx_start+b_reuse+1, j_idx_start, j_idx_stop, params->getAFormat());
				// It means that it could not fit the new row in LLB and failed
				if(llb->DoesFitInLLB('A', extra_size) == 0) {break;}
				llb->AddToLLB('A', Req::read, extra_size, UPDATE_TRAFFIC);
				b_reuse++;
			}
			params->setBReuse(b_reuse);
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
uint64_t Scheduler_10_drt::AccumulateSize(char mat_name, int d1_start, int d1_end,
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
			for(int i_idx= d1_start; i_idx < d1_end; i_idx++){
				if(inp_format == CSX::COO)
					size_tot += matrix->accumulateCOOSize('L',i_idx, d2_start, d2_end);
				else if(inp_format == CSX::CSF)
					size_tot += matrix->accumulateCSFSize('L',i_idx, d2_start, d2_end);
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
uint64_t Scheduler_10_drt::AccumulateNNZ(char mat_name, int d1_start, int d1_end,
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
			for(int i_idx=d1_start; i_idx < d1_end; i_idx++)
				nnz_tot += matrix->accumulateNNZ('L',i_idx, d2_start, d2_end, CSX::CSR);
			break;}

		default:{ printf("Unknown variable is requested!\n"); exit(1);}
	}
	return nnz_tot;
}



// Find out what is the B col sizes we need to fetch
//	after LLB intersection
void Scheduler_10_drt::PreCalculateBColsSize(int i_start, int i_end,
		int j_start, int j_end){

	// Finding whether any col of A is empty or not
	int * a_llb_verticalSum = new int[o_tiled_rows];
	// Calculate the horizontal sum of B PE tiles in the LLB tile
	//   This sum is used to decide whether to bring PE tiles of A
	//   It basically says whether there is any nnz PE tile in each row
	CalcALLBVerticalSum(i_start, i_end,
		j_start, j_end, a_llb_verticalSum);

	#pragma omp parallel for
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
void Scheduler_10_drt::CalcALLBVerticalSum(int i_start, int i_end,
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
uint64_t Scheduler_10_drt::AccumulateSize_BwrtA(int j_start, int j_end,
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
void Scheduler_10_drt::ExtractBTopTile(int i_start_top, int i_end_top,
		int j_start_top, int j_end_top, int k_start_top, int &k_end_top){

	// In the static tiling mechasnism we already know what k_idx_end
	//	should be since it is constant!
	if(params->getTilingMechanism() == tiling::t_static){
		k_end_top = std::min(o_tiled_cols, k_start_top + K_top_tile);
		uint64_t b_col_size = 0;
		for(int k_idx_top = k_start_top; k_idx_top < k_end_top; k_idx_top++){
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

void Scheduler_10_drt::PickPEsAccordingToPolicy(
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

void Scheduler_10_drt::PickPEsAccordingToPolicy(
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
template<class A, class B, class C> void Scheduler_10_drt::QuickSort3Desc(A a[], B b[], C c[], int l, int r)
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
template<class A, class B> void Scheduler_10_drt::QuickSort2Desc(A a[], B b[], int l, int r)
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
void Scheduler_10_drt::Swap(T &a, T &b)
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
uint64_t Scheduler_10_drt::updateBWLog(uint64_t starting_cycle, uint64_t action_bytes,
		float *bw_logger, float bytes_per_ns){

	double action_bytes_f = double(action_bytes);
	for(uint64_t i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		float rem_cap = bytes_per_ns - bw_logger[i_idx];
		// Move on until finding the first DRAM bw available cycle
		if((action_bytes_f > 0) & (rem_cap == 0))
			continue;
		// Use the available BW, but this is not the end
		if(action_bytes_f > (double) rem_cap){
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
	printf("%ld bandwidth logger: Max size is not enough - increase const value\n", MAX_TIME);
	exit(1);

	return 0;
}

// Multiplies one LLB row of A to a B LLB tile and reports what the iutput size will be
// Please note that this is just for the ideal llb partition policy and meant to
//	produce SoL variant result.
// This function is used straight out of ExTensor scheduler code (we needed the same
//	computation dataflow);Thus, it has been tested and works correctly.
//	There has been just some clean ups to keep the essential parts
uint64_t Scheduler_10_drt::multiplyOneARowInLogOutput(int i_idx,
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
	int vecA_offset = vecA_begin -  &(matrix->a_csr_outdims->idx[a_pos_start]);
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
			int vecB_offset = vecB_begin -  &(matrix->b_csc_outdims->idx[b_pos_start]);
			/* FWB: Until here!  ************************/

			// Find the intersection of an A_CSR and a B_CSC fiber
			std::vector<int> intersect_vector, intersect_overhead_vector, a_idx, b_idx;
			int len_B_fiber = 0;
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

// Gets two vectors and finds the intersection coordinates between them
// This function is used in ideal llb partitioning SoL pocily only
void Scheduler_10_drt::intersectTwoVectors(int * vec1_begin, int * vec1_end,
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

void Scheduler_10_drt::PrintBWUsage(){
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

