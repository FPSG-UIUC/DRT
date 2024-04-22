#ifndef SCHEDULER_3_H
#define SCHEDULER_3_H

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "parameters.h"
#include "stats.h"
//#include "llb_mem.h"

// 1 second in 1GHz, 10^9 ns
#define MAX_TIME 2000000000
class Scheduler_3{

	public:
		Scheduler_3(Matrix * mtx, Parameters * in_params, Stats * stats, LLB_Mem *llb);
		~Scheduler_3();
		void Run();
		//void Scheduler_3::Schedule(int i_idx, int j_idx);
		
	private:
		// Keeps per cycle information of bandwidth to calculate
		//   accurate timing information +  excess cycles
		float * bw_logger;
		// Start and end of a B col for each A tiles
		// At each round we look into A_rows tiles
		int * start_idx;
		int * end_idx;

		float bytes_per_ns;

		int load_b_tiles;

		int start_available;
		int end_available;

		uint64_t * pe_time;
	
		int o_tiled_rows;
		int o_tiled_cols;
		int b_tiled_rows;

		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem * llb;

		void Schedule(int i_idx, int j_idx);
		void TestSchedule(int i_idx, int j_idx);
		uint64_t FindExcessCycles(int cycles, uint64_t dram_bytes);
		void SyncPETimes();
		uint64_t updateBWLogAndReturnExcess(uint64_t starting_cycle, 
				uint64_t ending_cycle, uint64_t action_bytes);

};

#endif
