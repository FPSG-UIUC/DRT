#ifndef SCHEDULER_5_H
#define SCHEDULER_5_H

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

// 1 second in 1GHz, 10^9 ns
#define MAX_TIME 2000000000
class Scheduler_5{

	public:
		Scheduler_5(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb);
		~Scheduler_5();
		void Run();
		void PrintBWUsage();
		void PrintBWLog();

		// Reset all the internal stats
		void Reset();

		void * multiplierThread();
	private:

		pthread_t * threads;
		int num_worker_threads;
		int * thread_info_out;
		int ** thread_info_in;
		pthread_mutex_t * mutex;

		static void *multiplierHelperThread(void * ptr);

		// The traffic variable are for DEBUGGING purposes
		/*
		uint64_t a_traffic;
		uint64_t b_traffic;
		uint64_t o_traffic_read;
		uint64_t o_traffic_write;
		*/
		uint64_t a_bwl_traffic;
		uint64_t b_bwl_traffic;
		uint64_t o_bwl_traffic_read;
		uint64_t o_bwl_traffic_write;
		
		uint64_t total_traffic;

		int first_row_notFetched;
		int * b_llb_horizontalSum;
		// Keeps per cycle information of bandwidth to calculate
		//   accurate timing information +  excess cycles
		float * bw_logger;

		float bytes_per_ns;

		int load_b_tiles;

		uint64_t * pe_time;

		int * vecCommittedRows_iidx;
		uint64_t * vecCommittedRows_cycle;
		uint64_t * vecCommittedRows_ASize;
		uint64_t * vecCommittedRows_OSize;
		int num_vecCommittedRows;

		int * vecCyclesComp;
		uint64_t * OSize_preComp; 

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
		void Schedule(int i_idx, int j_idx_start, int j_idx_stop, int k_idx, 
			int *b_llb_horizontalSum);

		void checkForFinishedThreads(int j_start, int k_start);
		void submitTileMultiplicatin(int i_idx, int j_inner, int k_inner);
		void checkUntilAllThreadsFinished(int j_start, int k_idx);


		// Gets the A, B, and O tile addresses, does the MKL computation, updates the 
		//   cyclesComp array, and updates the busy cycles stats
		void DoTheTileMultiplication(int i_idx, int j_start, int j_inner,
	 	int k_start, int k_inner);

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
				uint64_t ending_cycle, uint64_t action_bytes);
		uint64_t updateBWLogAndReturnExcess(uint64_t starting_cycle, 
				uint64_t ending_cycle, uint64_t action_bytes, uint64_t a_update,
				uint64_t b_update, uint64_t o_r_update, uint64_t o_w_update );

		// Calculate O_reuse based on a_reuse and LLB partitioning constraints
		// Gets the Column boundary and start address of matrix B row,
		//   returns the start and stop address or the matrix B row, i.e., o_reuse
		// Find output reuse (number of matrix B rows to load in respect to a_reuse parameter)
		void CalcOReuse(int k_idx, int j_idx_start, int &j_idx_stop);
		
		void FetchAORows(int & numRows, int i_idx, 
				int j_idx_start, int j_idx_stop, int k_idx, int *b_llb_horizontalSum );

		// Gets the start and end address of both dimensions of either matrices (A, B, and O)
		//   and returns the size that block of tiles would occupy in LLB
		uint64_t AccumulateSize(char mat_name, int d1_start, int d1_end, int d2_start, int d2_end);

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

		// Swap two registers
		template<typename T>
			void Swap(T &a, T &b);
		
		void printPEs();

		pid_t gettid( void );

};

#endif
