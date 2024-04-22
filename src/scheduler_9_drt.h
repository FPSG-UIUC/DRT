#ifndef SCHEDULER_9_DRT_H
#define SCHEDULER_9_DRT_H

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
class Scheduler_9_drt{

	public:
		/* CONSTRUCTOR AND DESTRUCTOR*/
		// constructor -> intializer of the scheduler
		Scheduler_9_drt(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb);
		// Destructor -> delete [] bunch of dynamically allocated arrays
		~Scheduler_9_drt();

		/* MAIN EXECUTION INTERFACE FUNCTIONS */
		// If returns 0 it means that everything has been successful
		// if returns 1 (in static tiling) it means sth has gone wrong,
		//   currently sized of tiles have surpassed the LLB size
		int Run();
		// Reset all the internal stats; Used when there are multiple runs in one main file
		//	Usually used for bandwidth scaling sweep
		void Reset();

		void PrintBWUsage();
	private:
		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem *llb;

		/* Static tiling (ExTensor) related variables*/
		int I_top_tile, J_top_tile, K_top_tile;

		// Constant value for B_cols in LLB tile (in reflexive dynamic tiling)
		//	This variable will have the constant user entered value since
		//	the params value can change according	to the tile LLB chunk
		//	(might go further than a_reuse to fill the LLB)
		int b_reuse_const;

		int llb_reset_count;
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

		uint64_t * pe_utilization_logger;

		uint64_t * preCalculated_BColSize;

		int o_tiled_rows;
		int o_tiled_cols;
		int b_tiled_rows;

		/* Round-Robin Scheduler related variable */
		int round_robin_slot;

		// Gets A and B LLB tile and schedules them
		// Each processing element will get one basic tile of A and B
		//	in an A-stationary fashion
		void ScheduleBottomSDOT(int i_start_top, int i_end_top,
				int j_start_top, int j_end_top, int k_start_top, int k_end_top);

		/*
		// Multiplies one basic tile (micro tile) to another one and measures the bandwidth it needs
		void ScheduleBasicTileMult(int i_idx, int j_idx, int k_idx, uint64_t finalProductSize,
			 	std::vector<uint64_t> &a_traffic_vec, std::vector<uint64_t> &b_traffic_vec,
				std::vector<uint64_t> &o_traffic_vec, std::vector<uint64_t> &a_elements_in_col,
				std::vector<uint64_t> &macc_count_vec);*/
		// Multiplies one basic tile (micro tile) to another one and measures the bandwidth it needs
		void ScheduleBasicTileMult(int i_idx, int j_idx, int k_idx, uint64_t finalProductSize,
			 	uint64_t * a_traffic_vec, uint64_t * b_traffic_vec,
				uint64_t * o_traffic_vec, uint64_t * a_elements_in_col,
				uint64_t * macc_count_vec, int & vec_counter);

		// Gets the row boundary and start address of matrix A rows,
		//   returns the start and stop address of the matrix A cols, i.e., o_reuse
		// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
		void ExtractATopTile(int k_idx, int j_idx_start);

		// Iterates over the B LLB tile rows and finds all the basic tiles
		void ExtractBBasicTiles(int j_index_middle, int k_start_top, int k_end_top,
				std::vector<int> & b_indices_middle, std::vector<int> & b_offset_idx);

		// ExtractBTopTiles gets the dimensions of the B LLB tile and grows number of cols
		// until filling up the top buffer (LLB) for tensor B
		//	Tasks: 1) Report k_end_top 2) Fetch B top tiles into LLB buffer
		void ExtractBTopTile(int i_start_top, int i_end_top,
				int j_start_top, int j_end_top, int k_start_top, int &k_end_top);

		// Iterates over the A LLB tile columns and finds all the basic tiles
		void ExtractABasicTiles(int i_start_top, int i_end_top, int j_start_middle,
				std::vector<int> & a_indices_middle, std::vector<int> & a_offset_idx);

		// Find out what is the B col sizes we need to fetch
		//	after LLB intersection
		void PreCalculateBColsSize(int j_start, int j_end, int k_start, int k_end);

		// Find horizonatal sum of B tiles in LLB; this is used in
		//   deciding whether to bring A PE tiles to LLB or not
		// If horizontalsum of the row is zero then do not bring the
		//   corresponding PE tile of A
		void CalcALLBVerticalSum(int i_start, int i_end,
				int j_start, int j_end, int * a_llb_verticalSum);

		// Gets the start and end address of both dimensions of either matrices (A, B, and O)
		//   and returns the size that block of tiles would occupy in LLB
		uint64_t AccumulateSize(char mat_name, int d1_start, int d1_end,
			 	int d2_start, int d2_end, CSX inp_format);

		// Finds the size of B Macro tile wrt A macro tile
		//  looks at the A tiles in the LLB to see whether
		//  each B micro tile should be loaded or not
		uint64_t AccumulateSize_BwrtA(int j_start, int j_end, int k_idx, int *a_llb_verticalSum);

		// Gets the start and end address of both dimensions of either matrices (A, B, O, and O_log)
		//   and returns the total number of non-zero elements in that range
		uint64_t AccumulateNNZ(char mat_name, int d1_start, int d1_end,
			 	int d2_start, int d2_end);

		void PickPEsAccordingToPolicy(int count_needed, std::vector<int> & pe_indices);
		void PickPEsAccordingToPolicy(int count_needed, int * pe_indices);

		// This is a beautiful bw logger that gets the start cycle
		//   of each tile multiplication and in a cycle accurate way says
		//   when the data transfer is finished
		// Its role is to keep track of bandwidth either for the top level
		//  or middle level, depending on the bw_logger and bytes_per_ns assignment
		uint64_t updateBWLog(uint64_t starting_cycle, uint64_t action_bytes,
				float *bw_logger, float bytes_per_ns);

		//Multiplies one LLB row of A to B LLB tile and reports what the iutput size will be
		//Please note that this is just for the ideal llb partition policy and meant to
		//	produce SoL variant result.
		//This function is used straight out of ExTensor scheduler code (we needed the same
		//	computation dataflow);Thus, it has been tested and works correctly.
		//	There has been just some clean ups to keep the essential parts
		uint64_t multiplyOneARowInLogOutput(int i_idx,
				int j_start_top, int j_end_top, int k_start_top, int k_end_top);

		// Gets two vectors and finds the intersection coordinates between them
		// This function is used in ideal llb partitioning SoL pocily only
		void intersectTwoVectors(int * vec1_begin, int * vec1_end,
				int * vec2_begin,	int * vec2_end, int lower_bound, int upper_bound,
				std::vector<int> & intersect_vector);

		// Sorting a, b, and c according to a (b and c are dependent)
		// This function is used for finding to corresponding PE units, and pe_utilization_logger
		//	a is the nnz count for each pe, and pe is the pe progress time
		template<class A, class B, class C> void QuickSort3Desc(A a[], B b[], C c[], int l, int r);

		// Sorting a and b according to a (b is dependent)
		// This function is used for finding to corresponding PE units
		//	a is the nnz count for each pe, and pe is the pe progress time
		template<class A, class B> void QuickSort2Desc(A a[], B b[], int l, int r);

		template<typename T> void Swap(T &a, T &b);

		// DEBUGGING AND EXTRA INFORMATION VARIABLES
		// The traffic variable are for DEBUGGING purposes
		uint64_t deb_a_traffic;
		uint64_t deb_b_traffic;
		uint64_t deb_o_traffic_read;
		uint64_t deb_o_traffic_write;
		uint64_t deb_total_traffic;
};

#endif
