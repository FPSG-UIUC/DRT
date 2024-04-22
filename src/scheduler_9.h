#ifndef SCHEDULER_9_H
#define SCHEDULER_9_H

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
class Scheduler_9{

	public:
		/* CONSTRUCTOR AND DESTRUCTOR*/
		// constructor -> intializer of the scheduler
		Scheduler_9(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb);
		// Destructor -> delete [] bunch of dynamically allocated arrays
		~Scheduler_9();

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
		// The key objects in TACTile simulator
		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem *llb;

		// Number of rows/columns in A, B, and O
		int a_rows;
		int a_cols;
		int b_cols;

		// Keeps per cycle information of top and middle bandwidth to calculate
		//   accurate timing information +  excess cycles
		float * top_bw_logger;
		float * middle_bw_logger;

		// Bandwidth for top and middle layers in bytes/second
		float top_bytes_per_ns;
		float middle_bytes_per_ns;
		/* Bandwidth related variables until here */

		// PE and scheduler related variable
		uint64_t * pe_time;
		// array used for nnz based scheduler
		uint64_t * nnz_counts;
		// PE utilization logger for debugging purposes
		uint64_t * pe_utilization_logger;
		// Round-Robin Scheduler related variable
		int round_robin_slot;
		/* PE and scheduler related variables until here */

		// Arrays and variables used to keep track of the committed
		//	A cols/B rows that are wqaiting to be retired
		// This is a method to implement a first come first server
		//	scheduler while keeping track of the LLB usage
		int * vecCommittedACols_jidx;
		uint64_t * vecCommittedACols_cycle;
		int num_vecCommittedCols;

		// Retires the earliest finished outstanding committed column of A/Row of B
		void RetireOneAColumnTask(uint32_t * a_size_per_pe, uint32_t * b_size_per_pe,
				uint32_t * o_size_per_pe, uint32_t * a_elements_in_col);

		// This is a beautiful bw logger that gets the start cycle
		//   of each tile multiplication and in a cycle accurate way says
		//   when the data transfer is finished
		// Its role is to keep track of bandwidth either for the top level
		//  or middle level, depending on the bw_logger and bytes_per_ns assignment
		uint64_t UpdateBWLog(uint64_t starting_cycle, uint64_t action_bytes,
				float *bw_logger, float bytes_per_ns);

		// Pre calculate the A, B, and O size per PE of each col A/Row B
		// This function also figures out how many non-zeros are in each column of A
		//		and how many MACC (more precisely multiply) operations each corresponding
		//		PE unit needs to perform
		void PreCalculatePerColStats(uint32_t * a_size_per_pe, uint32_t * b_size_per_pe,
				uint32_t * o_size_per_pe,	uint32_t * macc_count_per_pe, uint32_t * a_elements_per_col);

		// Find the earliest (pe_time) start time among the chosen pe indices
		uint64_t FindBatchStartingCycle(uint32_t	a_elements_in_col, int * pe_indices);

		// Find the earliest (pe_time) end time among the chosen pe indices
		uint64_t FindBatchEndingCycle(uint32_t	a_elements_in_col, int * pe_indices);

		// This synchronization happens when a row related calculation is over
		//		So, when we are bringing a new row, we make sure that the timing has been updated
		//		min_val is the end_cycle of the committed row, which will be the start_cycle of
		//		the fetched row
		void SyncPETimesWithMinCycle(uint64_t min_val);

		// Returns count_needed pe indices according to the scheduling policy
		void PickPEsAccordingToPolicy(int count_needed, int * pe_indices);

		// Sorting a, b, and c according to a (b and c are dependent)
		// This function is used for finding to corresponding PE units, and pe_utilization_logger
		//	a is the nnz count for each pe, and pe is the pe progress time
		template<class A, class B, class C> void QuickSort3Desc(
				A a[], B b[], C c[], int l, int r);

		// Sorting a and b according to a (b is dependent)
		// This function is used for finding to corresponding PE units
		//	a is the nnz count for each pe, and pe is the pe progress time
		template<class A, class B> void QuickSort2Desc(
				A a[], B b[], int l, int r);

		// Swap two variables
		template<typename T> void Swap(T &a, T &b);

};
#endif
