#ifndef SCHEDULER_SPMM_8_H
#define SCHEDULER_SPMM_8_H

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
class Scheduler_SpMM_8{

	public:
		/* CONSTRUCTOR AND DESTRUCTOR*/
		// constructor -> intializer of the scheduler
		Scheduler_SpMM_8(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb);
		// Destructor -> delete [] bunch of dynamically allocated arrays
		~Scheduler_SpMM_8();

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
		int llb_reset_count;
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

		// Overhead of the tile extractor in tensors A & B DRAM->LLB
		//	(This shows the absolute value of overhead, e.g., 10 cycles)
		uint64_t overhead_extractor_top_A;
		uint64_t overhead_extractor_top_B;
		// Overhead of the basic tile builder & search of the extractor
		//   in tensor B for DRAM->LLB
		uint64_t overhead_tbuild_search_top_B;
		// Indicates the cycle where the extractor is done with
		//	extracting tensors A & B for top level (DRAM->LLB)
		uint64_t extractor_top_A_done;
		uint64_t extractor_top_B_done;
		// Indicates the cycle where the basic tile builder & search is done with
		//	tensor B for top (DRAM->LLB):
		//	 A can start tbuild and search (needs number of rows in B)
		uint64_t tbuild_search_top_B_done;
		// Parallelism factor for basic tile builing in LLB for tensors A & O
		int par_tbuild_A_top;
		int par_tbuild_B_top;

		uint64_t * t_extract_middle_A;

		int par_search_top;
		int log2_search_top;
		int log2_mbuild_top;

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

		/* Bandwidth related variables ************/

		// The amount of COO log that need to be read to top memory
		//	to do the merge part
		uint64_t totalSize_outputCOOLog;
		uint64_t totalNNZ_outputLog;
		// Keeps per cycle information of top and middle bandwidth to calculate
		//   accurate timing information +  excess cycles
		float * top_bw_logger;
		float * middle_bw_logger;

		// Bandwidth for top and middle layers in bytes/second
		float top_bytes_per_ns;
		float middle_bytes_per_ns;

		/* Bandwidth related variables until here */
		uint64_t * pe_time;
		uint64_t * nnz_counts;

		uint64_t * vecCommittedRows_ASize;

		int o_tiled_rows;
		int o_tiled_cols;
		int b_tiled_rows;

		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem *llb;

		// Gets a top layer (LLB) tile of A and B (both extracted from matrices)
		//	runs outer product on basic tile granularity and schedules each
		//	two basic tiles to the bottom DOT
		void ScheduleMiddleDOT(int i_start_top, int i_end_top,
				int j_start_top, int j_end_top, int k_start_top, int k_end_top);

		// Calculate O_reuse based on a_reuse and LLB partitioning constraints
		// Gets the Column boundary and start address of matrix B row,
		//   returns the start and stop address or the matrix B row, i.e., o_reuse
		// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
		void ExtractBTopTile(int k_idx, int j_idx_start);

		// Go through the rows of A matrix and figure out
		//	what a top tile (LLB tile) of A should look like
		// It also calculates the extract overhead for matrix A
		void ExtractATopTile(int i_start_top, int & i_end_top,
				int j_start_top, int j_end_top, int k_start_top, int k_end_top);

		// Early fetched the B basic tiles when using not pre-tiled matrix
		//	Therefore, we are having eith a parallel or serial (unlikely) basic tile builder
		void EarlyFetchBBasicTiles(int j_start_top, int j_end_top,
				int k_start_top, int k_end_top);

		// Early fetched the A basic tiles when using not pre-tiled matrix
		//	Therefore, we are having eith a parallel or serial (unlikely) basic tile builder
		void EarlyFetchABasicTiles(int i_start_top, int i_end_top,
				int j_start_top, int j_end_top);

		// This is a beautiful bw logger that gets the start cycle of
		//   each tile multiplication and in a cycle accurate way says how
		//   many extra cycles it is going to take.
		// Its role is to keep track of bandwidth
		// This works for both top and middle level according to the float *
		//   logger and bytes_per_ns (they can be for top or middle levels)
		uint64_t updateBWLog(uint64_t startingCycle, uint64_t action_bytes,
				float *bw_logger, float bytes_per_ns);

		// Find out what is the A row size we need to fetch
		//	after LLB intersection
		void PreCalculateARowsSize(int j_start, int j_end,
				int k_start, int k_end);

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
		void ExtractAMiddleTiles(int i_start_top, int i_end_top, int j_start_middle,
				std::vector<int> & i_indices_middle, std::vector<int> & i_entries_count_pos,
				std::vector<int> & i_offset_idx, std::vector<int> & a_search_overhead);

		// Iterates over the A LLB tile columns and divides the consecutive basic tiles
		//   in each column in a way that each group fits into the middle DOT buffer (PE buffer)
		//   For example: tile 0..5 -> Buffer 0, tile 6..8-> Buffer 1, ...
		// The functions gets the id of the column, the start and end I idx,
		//   Produces all the i_idxs (starts of the groups)
		//   and number of tiles in each group
		void ExtractBMiddleTiles(int j_index_middle, int k_start_top, int k_end_top,
				std::vector<int> & b_indices_middle, std::vector<int> & b_offset_idx);

		void PickPEsAccordingToPolicy(
				int count_needed, std::vector<int> & pe_indices);

		// FindBBaiscTileRanges function find the consecutive group of B basic tiles
		//  that can be fit into the bottom DOT memory at the same time.
		// This grouping is necessary to figure out when we need to wait
		//	or we can run computation without obstruction
		void FindBBasicTileRanges(int j_index_middle, int k_start_top, int k_end_top,
				std::vector<uint32_t> & b_ranges_start, std::vector<uint32_t> & b_ranges_end);

		// Calculates the earliest start time after taking into account
		//	current PE time, top overhead, and middle overhead
		void CalcTopMiddleDoneTime(int batch_starting_cycle,
				std::vector<int> a_search_overhead,
				std::vector<uint64_t> & top_middle_done_time);


		// This is a hard synchronization. Only happens when the last row of the
		//   matrix A & O are done and we need to fetch a new LLB tile of B
		void SyncPETimes();
		void SortPEs();

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

		// Find horizonatal sum of B tiles in LLB; this is used in
		//   deciding whether to bring A PE tiles to LLB or not
		// If horizontalsum of the row is zero then do not bring the
		//   corresponding PE tile of A
		void CalcBLLBHorizontalSum(int j_start, int j_end,
			int k_start, int k_end, int * b_llb_horizontalSum);


		void printPEs();

		// Sorting a and b according to a (b is dependent)
		// This function is used for finding to corresponding PE units
		//	a is the nnz count for each pe, and pe is the pe progress time
		template<class A, class B> void QuickSort2Desc(
			A a[], B b[], int l, int r);
		// Swap two registers
		template<typename T>
			void Swap(T &a, T &b);

};

#endif
