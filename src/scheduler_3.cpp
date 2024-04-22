#ifndef SCHEDULER_3_CPP
#define SCHEDULER_3_CPP

#include "scheduler_3.h"

Scheduler_3::Scheduler_3(Matrix * mtx, Parameters * in_params, Stats * stats, LLB_Mem * llb){
	matrix = mtx;
	params = in_params;
	this->stats = stats; 
	this->llb = llb;

	o_tiled_rows = matrix->getTiledORowSize();
	o_tiled_cols = matrix->getTiledOColSize();
	b_tiled_rows = matrix->getTiledBRowSize();

	bw_logger = new float[MAX_TIME];

	start_idx = new int[o_tiled_rows];
	end_idx   = new int[o_tiled_rows];
	std::fill(start_idx, start_idx+o_tiled_rows, 0);
	std::fill(end_idx, end_idx+o_tiled_rows, 0);

	start_available = -1;
	end_available = -1;

	pe_time = new uint64_t[params->getPECount()];
	std::fill(pe_time, pe_time+params->getPECount(), 0);

	bytes_per_ns = params->getBandwidth() / params->getFrequency();

	return;
}

Scheduler_3::~Scheduler_3(){
	delete [] bw_logger;

	delete [] start_idx;
	delete [] end_idx;

	delete [] pe_time;

	return;
}

void Scheduler_3::Run(){

	// loop over B rows (J) / A colds (J)
	printf("i %d, j %d\n",o_tiled_rows, b_tiled_rows);
	for(int j_idx=0; j_idx< b_tiled_rows; j_idx++){
		// read a new set of B row tiles
		SyncPETimes();
		load_b_tiles = 1; 
		// loop over A rows (I)
		for (int i_idx=0; i_idx< o_tiled_rows; i_idx++){
			Schedule(i_idx,j_idx);
		}
	}	
	return;
}

void Scheduler_3::Schedule(int i_idx, int j_idx){

	// Early termination! If A tile size is zero then we should skip
	//   all the computation
	if (matrix->getCSFSize('A', i_idx, j_idx) == 0){
		return;
	}

	uint64_t starting_cycle = 0, ending_cycle = 0, excessCycles = 0, 
					 bytes_rd = 0, bytes_wr = 0, action_bytes = 0;
	int cycles_comp = 0;
	
	// load A's tile
	uint64_t a_size_llb = matrix->getCSFSize('A', i_idx, j_idx);
	action_bytes += a_size_llb;

	// Stats for input A
	stats->Set_o_size(0);
	if(load_b_tiles) {stats->Set_b_size(0);}
	stats->Set_a_size(a_size_llb);
	stats->Accumulate_a_read(a_size_llb);

	for (int k_idx = 0; k_idx < o_tiled_cols; k_idx++){
		// Early termination! If B tile has zero nnz, then do not waste time on it
		if(matrix->getCSFSize('B', j_idx, k_idx) == 0)
				continue;

		// TODO: for more accurate timing, add bytes_read and bytes_write
		//  bytes_read should be added starting from the bw_logger[starting_cycles] and 
		//  bytes_write starting from bw_logger[starting_cycles+cycles]
		starting_cycle = *std::min_element(pe_time, pe_time+params->getPECount());
		if(load_b_tiles){
			uint64_t b_size_llb = 0;
			b_size_llb = matrix->getCSFSize('B', j_idx, k_idx);
			stats->Accumulate_b_size(b_size_llb);
		}
		matrix->CSRTimesCSR(i_idx, j_idx, k_idx, &cycles_comp, &bytes_rd, &bytes_wr);
		action_bytes += (bytes_rd + bytes_wr);
		ending_cycle = cycles_comp + starting_cycle;
		stats->Accumulate_pe_busy_cycles((uint64_t)cycles_comp);

		excessCycles = updateBWLogAndReturnExcess(starting_cycle, ending_cycle, action_bytes);

		*std::min_element(pe_time, pe_time+params->getPECount()) += 
			(uint64_t)cycles_comp + excessCycles;

		// Stats update
		stats->Accumulate_o_size(bytes_wr);
		stats->Accumulate_o_write(bytes_wr);
		stats->Accumulate_o_read(bytes_rd);

		action_bytes = 0;
	}
	load_b_tiles = 0;

	stats->Set_cycles(*std::max_element(pe_time, pe_time+params->getPECount()));
	stats->Set_runtime((double)stats->Get_cycles()/params->getFrequency());

	return;
}

uint64_t Scheduler_3::updateBWLogAndReturnExcess(
		uint64_t starting_cycle, uint64_t ending_cycle, uint64_t action_bytes){

	for(int i_idx = starting_cycle; i_idx< MAX_TIME; i_idx++){
		float pre_change = bw_logger[i_idx];
		if( bw_logger[i_idx] < bytes_per_ns ){
			bw_logger[i_idx] = bytes_per_ns;
			action_bytes -= (bw_logger[i_idx] - pre_change);
		}
		// We are done
		if(action_bytes<=0){
			// if there is extra deduction, add it back
			bw_logger[i_idx] += action_bytes;
			if (i_idx <= ending_cycle)
				return 0;
			else
				return i_idx - ending_cycle;
		}
	}
	printf("%d bandwidth loggers is not enough\n", MAX_TIME);
	exit(1);

}

uint64_t Scheduler_3::FindExcessCycles(int cycles, uint64_t dram_bytes){
	double bandwidth = params->getBandwidth();
	uint64_t frequency = params->getFrequency();

	uint64_t max_cycles = (uint64_t) (((double)dram_bytes/bandwidth) * frequency); 
	if(cycles > max_cycles )
		return cycles - max_cycles;
	else
		return 0;
}

void Scheduler_3::SyncPETimes(){

	int max_val = *std::max_element(pe_time, pe_time+params->getPECount());
	std::fill(pe_time, pe_time+params->getPECount(), max_val);

	return;
}

// A test scheduler for debugging purposes
void Scheduler_3::TestSchedule(int i_idx, int j_idx){
	//int starting_cycle = *std::max_element(pe_time, pe_time+params->getPECount());
	long int dram_bytes = 0;
	int cycles_comp = 0;
	uint64_t bytes_rd = 0, bytes_wr=0;

	for (int k_idx = 0; k_idx < o_tiled_cols; k_idx++){
		matrix->CSRTimesCSR(i_idx, j_idx, k_idx, &cycles_comp, &bytes_rd, &bytes_wr);
		*std::min_element(pe_time, pe_time+params->getPECount()) += cycles_comp;
		dram_bytes += bytes_rd + bytes_wr;

		//printf("(%d,%d),(%d,%d) -- \n",i_idx,j_idx,j_idx,k_idx);
	}
	return;
}

#endif
