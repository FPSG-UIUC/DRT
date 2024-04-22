#ifndef SCHEDULER_10_CPP
#define SCHEDULER_10_CPP

#include "scheduler_10.h"

// MatRaptor Scheduler!

// constructor -> intializer of the scheduler
Scheduler_10::Scheduler_10(Matrix * mtx, Parameters * params, Stats * stats, LLB_Mem * llb){
	this->matrix = mtx;
	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Dimensions of inputs and the output tensors
	a_rows = matrix->GetARowSize();
	a_cols = matrix->GetAColSize();
	b_cols = matrix->GetBColSize();

	// #Bytes that can be transferred every cycle
	top_bytes_per_ns = (float)params->getTopBandwidth() / params->getFrequency();

	// BW logger to have bw usage in cycle level accuracy
	//	This is a cool approach.
	// The top variable is for the top level DRAM->LLB
	// The middle variable is for the middle level LLB->PE
	top_bw_logger = new float[MAX_TIME];

	// PE times
	pe_time = new uint64_t[params->getPECount()];
	std::fill(pe_time, pe_time+params->getPECount(), 0);

	// Initialize PE arrays and BW_logger to all zeros
	std::fill(top_bw_logger, top_bw_logger+MAX_TIME, 0.0);

	pe_utilization_logger = new uint64_t[params->getPECount()];
	std::fill(pe_utilization_logger, pe_utilization_logger+params->getPECount(), 0);

	return;
}

// Destructor -> delete [] bunch of dynamically allocated arrays
Scheduler_10::~Scheduler_10(){
	delete [] top_bw_logger;
	delete [] pe_time;
	delete [] pe_utilization_logger;

	return;
}

// Reset all the internal stats; Used when there are multiple runs in one main file
//	Usually used for bandwidth scaling sweep
void Scheduler_10::Reset(){

	// Reset PE_time and BW_logger
	std::fill(pe_time, pe_time+params->getPECount(), 0);
	std::fill(top_bw_logger, top_bw_logger+MAX_TIME, 0.0);
	// Sometimes bandwidth changes before reset, so update it
	top_bytes_per_ns = (float)params->getTopBandwidth() / params->getFrequency();

	return;
}

/* Dataflow in pseudo code is :
 *
 * for(i=0; i<a_rows; j++)
 *	for(j=0; j<a_cols; i++)
 *		for(k=0; k<b_cols; k++)
 *			a[i][k] += a[i][j]*b[j][k];
 *
 */
int Scheduler_10::Run(){

	uint64_t a_traffic, b_traffic, o_traffic, macc_count;
	uint64_t total_a_traffic = 0, total_b_traffic = 0, total_o_traffic = 0,
			total_traffic = 0, total_maccs = 0;
	std::vector<uint64_t> a_traffic_vec, b_traffic_vec, o_traffic_vec, macc_count_vec;

	// Calculate the output product so it can be referenced everytime through
	//		GustavsonProduct; GP will return the access size per tensor and #MACCs
    //This function updates the o_csr structure with the final result
	matrix->CalculateNotTiledMatrixProduct();

	// iterate over rows of A
	for(int i_idx=0; i_idx< a_rows; i_idx++){
		// This function just looks at the numbers and does not do actual multiplication
		matrix->GustavsonProduct(i_idx, a_traffic, b_traffic, o_traffic, macc_count);
		//a_traffic_vec.push_back(a_traffic); b_traffic_vec.push_back(b_traffic);
		//o_traffic_vec.push_back(o_traffic); macc_count_vec.push_back(macc_count);

		total_o_traffic += o_traffic;
		total_a_traffic += a_traffic;
		total_b_traffic += b_traffic;
		total_maccs += macc_count;
	}

	total_traffic = total_a_traffic + total_b_traffic + total_o_traffic;

	printf("Total traffic: %0.3f GB, Total MACCs: %lu\n", (double)total_traffic/(double)(1024*1024*1024), total_maccs);
	printf("A traffic: %0.3f GB, B traffic: %0.3f GB, O traffic: %0.3f GB\n", (double)total_a_traffic/(double)(1024*1024*1024), 
			                                                          (double)total_b_traffic/(double)(1024*1024*1024),
										  (double)total_o_traffic/(double)(1024*1024*1024));
	return 0;
}

#endif
