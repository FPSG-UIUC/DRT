#ifndef SCHEDULER_SpMM_7_H
#define SCHEDULER_SpMM_7_H

#include <queue>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <linux/unistd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "matrix.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"

#define DONT_UPDATE_TRAFFIC 0
#define UPDATE_TRAFFIC 1

// 2 seconds in 1GHz, 2 * 10^9 ns
#define MAX_TIME 2000000000
class Scheduler_SpMM_7{

	public:
		/* CONSTRUCTOR AND DESTRUCTOR*/
		// constructor -> intializer of the scheduler
		Scheduler_SpMM_7(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb);
		// Destructor -> delete [] bunch of dynamically allocated arrays
		~Scheduler_SpMM_7();

		/* MAIN EXECUTION INTERFACE FUNCTIONS */
		// If returns 0 it means that everything has been successful
		// if returns 1 (in static tiling) it means sth has gone wrong,
		//   currently sized of tiles have surpassed the LLB size
		int Run();
		// Reset all the internal stats; Used when there are multiple runs in one main file
		//	Usually used for bandwidth scaling sweep
		void Reset();

		/* DEBUG FUCTIONS */
		// Sanity check for bandwidth and traffic used for each matrix (A, B, and O)
		// This is the heart of scheduler debugger; If numbers are correct, then
		//		intersects, conditions, and schedules are working fine.
		void PrintBWUsage();
		// Sanity check for bw logger. Will output cycle accurate bw usage to a log file
		void PrintBWLog();
	private:
		/* Static tiling (ExTensor) related variables*/
		int I_top_tile, J_top_tile, K_top_tile;
		// Shows how many row slots we have in total
		int total_row_slots;
		// Shows how many of the rows are available to load data
		int remaining_row_slots;

		/* Round-Robin Scheduler related variable */
		int round_robin_slot;

		/* Extractor related arrays to keep track of overheads */

		/* TOP Lvele (LLB)
		 *   related Extract and Intersection variables */

		// Vectors to record the extract overhead (t_build, search_mbuild)
		//	to take into account overheads in high-fidelity simulation
		uint64_t * vec_O_ExtractOverhead_top;
		uint64_t * vec_A_ExtractOverhead_top;
		// A time-keeper for each basic tile builder in LLB of tensors A & O
		uint64_t * tbuild_A_time_top;
		uint64_t * tbuild_O_time_top;
		// Overhead of the tile extractor in tensor B DRAM->LLB
		//	(This shows the absolute value of overhead, e.g., 10 cycles)
		uint64_t overhead_extractor_top_B;
		// Overhead of the basic tile builder of the extractor
		//   in tensor B for DRAM->LLB
		uint64_t overhead_tbuild_top_B;
		// Indicates the cycle where the extractor is done with
		//	extracting tensor B for LLB
		uint64_t extractor_top_B_done;
		// Indicates the cycle where the basic tile builder is done with
		//	tensor B for LLB
		uint64_t tbuild_top_B_done;
		// Parallelism factor for basic tile builing in LLB for tensors A & O
		int par_tbuild_A_top;
		int par_tbuild_B_top;
		int par_tbuild_O_top;

		int par_search_B_top;
		int log2_search_B_top;
		int log2_mbuild_B_top;

		// A time-keeper for the intersection unit for the top level (DRAM->LLB)
		//	As of the discussion with Chris, this just decides whether
		//	the specific row should be fetched or not!
		uint8_t * intersect_logger_top;
		// Keeps the CSR format of b_horizontal_sum to intersect
		//  with everru A row
		//  --> gives the overhead for top level intersect (DRAM->LLB)
		int * b_intersect_vector_top;
		// b_intersect_vector_top vector size
		int b_inter_count_top;

		// Keeps the intersect overhead of the top level for each row
		uint8_t * intersect_overhead_top;
		// This is going to be a pointer to vec_A_ExtractOverhead_top, which
		//  will keep the cycle when all the top level operations for the
		//  current row is over
		uint64_t * extractor_intersect_done_top;

		int par_middle;
		/* Until here*/




		// Constant value for B_cols in LLB tile (in reflexive dynamic tiling)
		//	This variable will have the constant user entered value since
		//	the params value can change according	to the tile LLB chunk
		//	(might go further than a_reuse to fill the LLB)
		int a_reuse_const;

		// The traffic variable are for DEBUGGING purposes
		uint64_t a_traffic;
		uint64_t b_traffic;
		uint64_t o_traffic_read;
		uint64_t o_traffic_write;

		uint64_t a_bwl_traffic;
		uint64_t b_bwl_traffic;
		uint64_t o_bwl_traffic_read;
		uint64_t o_bwl_traffic_write;

		uint64_t total_traffic;
		uint64_t total_traffic_middle;

		int first_row_notFetched;
		// Keeps per cycle information of bandwidth to calculate
		//   accurate timing information +  excess cycles
		float * bw_logger;
		float * middle_bw_logger;

		float bytes_per_ns;
		float middle_bytes_per_ns;

		int load_b_tiles;

		uint64_t * pe_time;

		int * vecCommittedRows_iidx;
		uint64_t * vecCommittedRows_cycle;
		uint64_t * vecCommittedRows_ASize;
		uint64_t * vecCommittedRows_OSize;
		int num_vecCommittedRows;

		int o_tiled_rows;
		int o_tiled_cols;
		int b_tiled_rows;

		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem *llb;

		// Gets a fiber of A and O and a rectangle LLB tile of B and schedules them
		//   The ending cycles of each row is recorded in vecCommittedRows for
		//   synnchronization, fetching, and committing rows
		void Schedule(int i_idx, int j_idx_start, int j_idx_stop, int k_idx);

		// This function gets the cycles and bytes that a tile multiplication takes
		//  Then updates the bw logger, and adds the extra cycles to the computation
		//  and sschedules it
		void scheduleAndUpdateBW(int cycles_comp, uint64_t action_bytes,
				uint64_t &batch_ending_cycle);

		// This is a hard synchronization. Only happens when the last row of the
		//   matrix A & O are done and we need to fetch a new LLB tile of B
		void SyncPETimes();

		// This is for the fine synchronization after evicting a row and bringing
		//   new rows to LLB. All PEs should be at least at the same cycle as
		//   the evicted row.
		void SyncPETimesWithMinCycle(uint64_t min_val);

		// This is a beautiful bw logger that gets the start cycle and end cycle
		//   of each tile multiplication and in a cycle accurate way says how
		//   many extra cycles it is going to take.
		// Its role is to keep track of bandwidth
		uint64_t updateBWLogAndReturnExcess(uint64_t starting_cycle,
				uint64_t ending_cycle, uint64_t action_bytes, uint64_t & finish_cycle);
		// Same with debugging purposes (adds up all details to make sure
		//  every single byte is calculated correctly)
		uint64_t updateBWLogAndReturnExcess(uint64_t starting_cycle,
				uint64_t ending_cycle, uint64_t action_bytes, uint64_t & finish_cycle,
			 	uint64_t a_update, uint64_t b_update, uint64_t o_r_update, uint64_t o_w_update);

		// Bandwidth logger for the middle DOT
		uint64_t updateBWLogAndReturnExcessMiddleDOT(uint64_t starting_cycle,
				uint64_t ending_cycle, uint64_t action_bytes,uint64_t & finish_cycle);

		// Uses a logger array for top level intersection unit. When
		//	an LLB row of A & O are ready they are passed to the intersection unit
		//	to calculate the ready time.
		// This functions gets the time when extracting A & O rows are over and
		//  the #cycles intersection would take. Then, the function checks the
		//  array and finds the closest empty cycles.
		uint64_t findFinishTimeTopLevel(uint64_t vec_A_time,
			uint64_t vec_O_time, uint8_t intersect_overhead);

		// Returns the cycles taken for Extract in the middle layer
		//   via overhead_middle_extract variable
		void CalculateMiddleLevelExtarctOverhead(int a_vec_count, int b_vec_count,
				int num_effectaul_intersects, uint32_t & overhead_middle_extract);
		// Returns the cycles taken for intersection in the middle layer
		//   via overhead_middle_intersect variable
 		void CalculateMiddleLevelIntersectOverhead(
				std::vector<int> intersect_overhead, uint32_t & overhead_middle_intersect);

		// Calculate O_reuse based on a_reuse and LLB partitioning constraints
		// Gets the Column boundary and start address of matrix B row,
		//   returns the start and stop address or the matrix B row, i.e., o_reuse
		// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
		void CalcOReuse(int k_idx, int j_idx_start, int &j_idx_stop);

		// fill up LLB with A and O rows!
		// It fetches A and O rows as much as LLB allows. It first evicts the row with
		//   the smallest end cycle time and gives its space to new rows. The number of new rows
		//   is returned in numRows (range [0,o_tiled_rows] ).
		void FetchAORows(int & numRows, int i_idx,
				int j_idx_start, int j_idx_stop, int k_idx);

		// Find out what is the A and O row size we need to fetch
		//	after LLB intersection
		void PreCalculateAORowsSize(int j_start, int j_end,
				int k_start, int k_end);

		// Gets the start and end address of both dimensions of either matrices (A, B, and O)
		//   and returns the total number of non-zero elements in that range
		uint64_t AccumulateNNZ(char mat_name, int d1_start, int d1_end,
			 	int d2_start, int d2_end);

		// Gets the start and end address of both dimensions of either matrices (A, B, and O)
		//   and returns the total number of non zero rows among all tiles of that range
		uint64_t AccumulateNNR(char mat_name, int d1_start, int d1_end,
				int d2_start, int d2_end);

		// Gets the start and end address of both dimensions of either matrices (A, B, and O)
		//   and returns the total number of non zero tiles in that range
		uint64_t AccumulateNNZTiles(char mat_name, int d1_start, int d1_end,
				int d2_start, int d2_end);

		// Gets the start and end address of both dimensions of either matrices (A, B, and O)
		//   and returns the size that block of tiles would occupy in LLB
		uint64_t AccumulateSize(char mat_name, int d1_start, int d1_end,
				int d2_start, int d2_end, CSX inp_format);

		// Finds the size of A PE tiles wrt B tiles
		//  looks at the B tiles in the LLB to see whether
		//  they should be loaded or not
		uint64_t AccumulateSize_AwrtB(int i_idx, int j_start, int j_end, int *b_llb_horizontalSum);

		// Finds the size of O PE tiles wrt A & B tiles
		//  looks at the A&B intersection in the LLB to see whether
		//  they should be loaded or not
		uint64_t AccumulateSize_OwrtAB(int i_idx, int j_start, int j_end, int k_start, int k_end);

		// Gets the i, k and j range for matrix A and B
		//   then says whther the specific O tile should be fetched or
		//   not based on the intersection
		uint32_t ShouldIFetchThisOTile(int i_idx, int j_idx_start, int j_idx_end, int k_idx);

		// Find horizonatal sum of B tiles in LLB; this is used in
		//   deciding whether to bring A PE tiles to LLB or not
		// If horizontalsum of the row is zero then do not bring the
		//   corresponding PE tile of A
		void CalcBLLBHorizontalSum(int j_start, int j_end,
			int k_start, int k_end, int * b_llb_horizontalSum);

		// Gets two vectors and finds whether the intersection is empty or not
		//		empty: 0
		//		not empty: 1
		int intersectTwoVectors(int * vec1_begin, int * vec1_end, int * vec2_begin,
				int * vec2_end, int lower_bound, int upper_bound);
		void intersectTwoVectors(int * vec1_begin, int * vec1_end,
				int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound,
				std::vector<int> & intersect_vector, std::vector<int> & overhead_vector,
				std::vector<int> & a_idx, std::vector<int> & b_idx,
				int & len_B_fiber);

		// Gets two vectors and says how many cycles it takes to do the intersection
		int getOverheadIntersectTwoVectors(int * vec1_begin, int * vec1_end,
				int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound);

		void findJIndexesInTheRange(int i_idx, int j_idx_start, int j_idx_end,
				int * vecA_begin, int * vecA_end,
				std::vector<int> & j_indexes, std::vector<int> & a_entries_count);

		// Swap two registers
		template<typename T>
			void Swap(T &a, T &b);

		void printPEs();
};

#endif
