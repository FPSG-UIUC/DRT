#ifndef SCHEDULER_10_H
#define SCHEDULER_10_H

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
class Scheduler_10{

	public:
		/* CONSTRUCTOR AND DESTRUCTOR*/
		// constructor -> intializer of the scheduler
		Scheduler_10(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb);
		// Destructor -> delete [] bunch of dynamically allocated arrays
		~Scheduler_10();

		/* MAIN EXECUTION INTERFACE FUNCTIONS */
		// If returns 0 it means that everything has been successful
		// if returns 1 (in static tiling) it means sth has gone wrong,
		//   currently sized of tiles have surpassed the LLB size
		int Run();
		// Reset all the internal stats; Used when there are multiple runs in one main file
		//	Usually used for bandwidth scaling sweep
		void Reset();

	private:
		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem *llb;

		// Keeps per cycle information of top and middle bandwidth to calculate
		//   accurate timing information +  excess cycles
		float * top_bw_logger;
		// Bandwidth for top and middle layers in bytes/second
		float top_bytes_per_ns;

		/* Bandwidth related variables until here */
		uint64_t * pe_time;

		uint64_t * pe_utilization_logger;

		int a_rows;
		int a_cols;
		int b_cols;
};

#endif
