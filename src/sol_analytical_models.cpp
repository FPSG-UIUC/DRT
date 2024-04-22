#ifndef SOL_ANANLYTICAL_MODELS_CPP
#define SOL_ANANLYTICAL_MODELS_CPP

#include "sol_analytical_models.h"

// Computes the model 0 runtime of the dataset
//	it works only when ideal Intersect model is used
// MODEL 0: This is a theoretical maximum speed-up (SoL) that takes into
//   account only the number of MACC units. It finds the total number of
//   multiplications and divide them by the number of available PEs
void SoL_Analytical_Models::model_0(){

	float runtime = (double)stats->Get_pe_busy_cycles() /
			((double)params->getPECount() * params->getFrequency());
	if(params->getIntersectModel() == intersect::idealModel){
		printf("Model 0: %f\n", runtime);
	}
	else{
		printf("Model 0 report is based on skip model, changes with tile size\n");
		printf("Model 0: %f\n", runtime);

	}
	return;
}

// Computes the model 1 runtime of the dataset
// MODEL 1: This is a theoretical maximum speed-up (SoL) that takes into
//   account only the bandwidth limit constraint. It finds the total size of
//   the matrices A, B, and O iff we have only one pass over each of them
//   (the case of infinity LLB memory); Then, divides them by the peak bandwidth
//   (baseline is 68.256GB/s)
void SoL_Analytical_Models::model_1(){
	uint64_t size_a = 0, size_b = 0, size_o_coo = 0, size_o_csf = 0;

	// Dimensions of inputs and the output tensors
	int a_rows, a_cols, b_cols, b_rows, o_rows, o_cols;
	if(params->getCompKernel() == kernel::SpMM){
		a_rows = matrix->getTiledARowSize();
		a_cols = matrix->getTiledAColSize();
		b_cols = (int)ceil((float)params->getNumDenseCols()/params->getTileSize());
		b_rows = a_cols; o_rows = a_rows; o_cols = b_cols;
	}
	else{
	// Get the dimensions for each matrix
		a_rows = matrix->getTiledORowSize();
		a_cols = matrix->getTiledBRowSize();
		b_cols = matrix->getTiledOColSize();
		b_rows = a_cols; o_rows = a_rows; o_cols = b_cols;
	}
	// Find the size of each matrix
	for(int i_idx = 0; i_idx<a_rows; i_idx++)
		size_a += matrix->accumulateCSFSize_sparse('A',i_idx, 0, a_cols, CSX::CSR);
	if(params->getBFormat() == CSX::Dense){
		size_b = b_cols*b_rows*params->getDenseTileSize();
	}
	else{
		for(int j_idx = 0; j_idx<b_rows; j_idx++)
			size_b += matrix->accumulateCSFSize_sparse('B',j_idx, 0, b_cols, CSX::CSR);
	}

	if(params->getOFormat() == CSX::Dense){
		size_o_csf = o_cols*o_rows*params->getDenseTileSize();
	}
	else{
		for(int i_idx = 0; i_idx<o_rows; i_idx++)
			for(int k_idx = 0; k_idx<o_cols; k_idx++){
				size_o_coo += matrix->getCOOSize('O', i_idx, k_idx);
				size_o_csf += matrix->getCSFSize('O', i_idx, k_idx);
			}
	}
	// Compute the maximum theoretical speed-up concerning the bandwidth
	double runtime = 0;
	if(params->getTilingMechanism() == tiling::t_static)
		runtime = double(size_a+size_b+size_o_coo) / (double)params->getBandwidth();
	else if(params->getTilingMechanism() == tiling::t_dynamic)
		runtime = double(size_a+size_b+size_o_csf) / (double)params->getBandwidth();
	else{
		printf("There is no such tiling mechanism! sol_model1 error");
		exit(1);
	}
	printf("Model 1: %f\n", runtime);
	printf("A_csf: %lu, B_csf: %lu, O_csf: %lu, O_COO: %lu, bandwidth: %f GB/s\n",
			size_a, size_b, size_o_csf, size_o_coo, params->getBandwidth()/(1024*1024*1024));

	return;
}

// Computes the model 1 runtime of the dataset
// MODEL 1: This is a theoretical maximum speed-up (SoL) that takes into
//   account only the bandwidth limit constraint. It finds the total size of
//   the matrices A, B, and O iff we have only one pass over each of them
//   (the case of infinity LLB memory); Then, divides them by the peak bandwidth
//   (baseline is 68.256GB/s)
void SoL_Analytical_Models::model_1_lazyFetch(){
	uint64_t size_a = 0, size_b = 0, size_o_coo = 0, size_o_csf = 0;

	// Dimensions of inputs and the output tensors
	int a_rows, a_cols, b_cols, b_rows, o_rows, o_cols;
	// Get the dimensions for each matrix
	a_rows = matrix->getTiledORowSize();
	a_cols = matrix->getTiledBRowSize();
	b_cols = matrix->getTiledOColSize();
	b_rows = a_cols; o_rows = a_rows; o_cols = b_cols;

	uint64_t * a_vertical_sum = new uint64_t[a_cols];
	uint64_t * b_horizontal_sum = new uint64_t[b_rows];
	std::fill(a_vertical_sum, a_vertical_sum + a_cols, 0);
	std::fill(b_horizontal_sum, b_horizontal_sum + b_rows, 0);

	for(int j_idx = 0; j_idx< a_cols; j_idx++){
		// sum of a column of A micro tile sizes
		a_vertical_sum[j_idx] = matrix->accumulateCSFSize_sparse('A',j_idx, 0, a_rows, CSX::CSC);
		// sum of a row of B micro tile sizes
		b_horizontal_sum[j_idx] = matrix->accumulateCSFSize_sparse('B',j_idx, 0, b_cols, CSX::CSR);
		// If any of them is empty then the corresponding data should not be fetched
		if((a_vertical_sum[j_idx] == 0) | (b_horizontal_sum[j_idx] == 0)){
			a_vertical_sum[j_idx] = 0; b_horizontal_sum[j_idx] = 0;
		}
	}

	for(int j_idx = 0; j_idx< a_cols; j_idx++){
		size_a += a_vertical_sum[j_idx];
		size_b += b_horizontal_sum[j_idx];
	}

	for(int i_idx = 0; i_idx<o_rows; i_idx++){
		for(int k_idx = 0; k_idx<o_cols; k_idx++){
			size_o_coo += matrix->getCOOSize('O', i_idx, k_idx);
			size_o_csf += matrix->getCSFSize('O', i_idx, k_idx);
		}
	}

	// Compute the maximum theoretical speed-up concerning the bandwidth
	double runtime = 0;
	if(params->getTilingMechanism() == tiling::t_static)
		runtime = double(size_a+size_b+size_o_coo) / (double)params->getBandwidth();
	else if(params->getTilingMechanism() == tiling::t_dynamic)
		runtime = double(size_a+size_b+size_o_csf) / (double)params->getBandwidth();
	else{
		printf("There is no such tiling mechanism! sol_model1 error");
		exit(1);
	}
	printf("Model 1 Lazy Fetch: %f\n", runtime);
	printf("A_csf: %lu, B_csf: %lu, O_csf: %lu, O_COO: %lu, bandwidth: %f GB/s\n",
			size_a, size_b, size_o_csf, size_o_coo, params->getBandwidth()/(1024*1024*1024));

	delete [] a_vertical_sum;
	delete [] b_horizontal_sum;

	return;
}

// Computes the model 1 runtime of the dataset
// MODEL 1: This is a theoretical maximum speed-up (SoL) that takes into
//   account only the bandwidth limit constraint. It finds the total size of
//   the matrices A, B, and O iff we have only one pass over each of them
//   (the case of infinity LLB memory); Then, divides them by the peak bandwidth
//   (baseline is 68.256GB/s)
void SoL_Analytical_Models::model_1_lazyFetch_no_output(){
	uint64_t size_a = 0, size_b = 0;

	// Dimensions of inputs and the output tensors
	int a_rows, a_cols, b_cols, b_rows;
	// Get the dimensions for each matrix
	a_rows = matrix->getTiledORowSize();
	a_cols = matrix->getTiledBRowSize();
	b_cols = matrix->getTiledOColSize();
	b_rows = a_cols;

	uint64_t * a_vertical_sum = new uint64_t[a_cols];
	uint64_t * b_horizontal_sum = new uint64_t[b_rows];
	std::fill(a_vertical_sum, a_vertical_sum + a_cols, 0);
	std::fill(b_horizontal_sum, b_horizontal_sum + b_rows, 0);

	for(int j_idx = 0; j_idx< a_cols; j_idx++){
		// sum of a column of A micro tile sizes
		a_vertical_sum[j_idx] = matrix->accumulateCSFSize_sparse('A',j_idx, 0, a_rows, CSX::CSC);
		// sum of a row of B micro tile sizes
		b_horizontal_sum[j_idx] = matrix->accumulateCSFSize_sparse('B',j_idx, 0, b_cols, CSX::CSR);
		// If any of them is empty then the corresponding data should not be fetched
		if((a_vertical_sum[j_idx] == 0) | (b_horizontal_sum[j_idx] == 0)){
			a_vertical_sum[j_idx] = 0; b_horizontal_sum[j_idx] = 0;
		}
	}

	for(int j_idx = 0; j_idx< a_cols; j_idx++){
		size_a += a_vertical_sum[j_idx];
		size_b += b_horizontal_sum[j_idx];
	}

	printf("Model 1 Lazy Fetch No Output Size\n");
	printf("A_csf_fetch_size: %lu, B_csf_fetch_size: %lu\n", size_a, size_b);

	delete [] a_vertical_sum;
	delete [] b_horizontal_sum;

	return;
}




void SoL_Analytical_Models::model_2(){

	uint64_t size_a = 0, size_b = 0, size_o =0, size_o_coo = 0, size_o_csf = 0;
	double a_row_size = 0, b_col_size = 0, o_block_size = 0;
	double llb_size = (double)llb->GetCapacity();
	// Dimensions of inputs and the output tensors
	int a_rows, a_cols, b_cols, b_rows, o_rows, o_cols;
	if(params->getCompKernel() == kernel::SpMM){
		a_rows = matrix->getTiledARowSize();
		a_cols = matrix->getTiledAColSize();
		b_cols = (int)ceil((float)params->getNumDenseCols()/params->getTileSize());
		b_rows = a_cols; o_rows = a_rows; o_cols = b_cols;
	}
	else{
	// Get the dimensions for each matrix
		a_rows = matrix->getTiledORowSize();
		a_cols = matrix->getTiledBRowSize();
		b_cols = matrix->getTiledOColSize();
		b_rows = a_cols; o_rows = a_rows; o_cols = b_cols;
	}
	// Find the size of each matrix
	for(int i_idx = 0; i_idx<a_rows; i_idx++)
		size_a += matrix->accumulateCSFSize_sparse('A',i_idx, 0, a_cols, CSX::CSR);
	if(params->getBFormat() == CSX::Dense){
		size_b = b_cols*b_rows*params->getDenseTileSize();
	}
	else{
		for(int j_idx = 0; j_idx<b_rows; j_idx++)
			size_b += matrix->accumulateCSFSize_sparse('B',j_idx, 0, b_cols, CSX::CSR);
	}

	if(params->getOFormat() == CSX::Dense){
		size_o_csf = o_cols*o_rows*params->getDenseTileSize();
	}
	else{
		for(int i_idx = 0; i_idx<o_rows; i_idx++)
			for(int k_idx = 0; k_idx<o_cols; k_idx++){
				size_o_coo += matrix->getCOOSize('O', i_idx, k_idx);
				size_o_csf += matrix->getCSFSize('O', i_idx, k_idx);
			}
	}
	// Compute the maximum theoretical speed-up concerning the bandwidth
	a_row_size = (double)size_a / (double)a_rows;
	b_col_size = (double)size_b / (double)b_cols;
	if(params->getTilingMechanism() == tiling::t_static){
		o_block_size = (double)size_o_coo / (double)(a_rows*b_cols);
		size_o = size_o_coo;
	}
	else if(params->getTilingMechanism() == tiling::t_dynamic){
		o_block_size = (double)size_o_csf / (double)(a_rows*b_cols);
		size_o = size_o_csf;
	}
  int max_A_rows = 1, max_B_cols = 1;
  double found_size = max_A_rows*a_row_size + max_B_cols*b_col_size + max_A_rows*max_B_cols*o_block_size;
	double new_size =0;
  while(found_size < llb_size){
		new_size = (max_A_rows+1)*a_row_size + (max_B_cols+1)*b_col_size + ((max_A_rows+1)*(max_B_cols+1))*o_block_size;
		if (new_size > llb_size)
			break;
		max_A_rows++;
		max_B_cols++;
		found_size = new_size;
		if ((max_A_rows == a_rows) | (max_B_cols == b_cols))
			break;
	}
  uint64_t total_traffic = size_o + size_b + (uint64_t) ceil((double)a_rows / (double)max_A_rows) * size_a;
  double total_traffic_gb = double(total_traffic) / double(1024*1024*1024);

	double runtime = double(total_traffic) / (double)params->getBandwidth();

	printf("Model 2 runtime: %.6f\ttraffic: %.3f\n", runtime, total_traffic_gb);
}
#endif
