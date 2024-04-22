#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include "matrix.h"
#include "input_arg_parser.h"
#include "scheduler_8.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"
#include "sol_analytical_models.h"

using namespace std;

void initializeInputParameters(input_parameters_st *input_params);

int main(int argc, char *argv[])
{
	input_parameters_st * ip = new input_parameters_st;

	initializeInputParameters(ip);

	parseInput(argc, argv, ip);

	Parameters * params= new Parameters(ip->p_kernel,
			ip->p_a_format, ip->p_b_format, ip->p_o_format, ip->p_dense_cols,
			ip->p_tiling, ip->p_static_dist, ip->p_middle_dataflow,
			ip->p_static_i, ip->p_static_j, ip->p_static_k,
			ip->p_tile_dim, ip->p_pe_count, ip->p_cam_nums, ip->p_top_bw, ip->p_middle_bw,
			ip->p_chip_freq, ip->p_thread_count, ip->p_a_reuse,
			ip->p_bottom_intersect,
			ip->p_middleDOT_traffic, ip->p_middle_intersect, ip->p_middle_search, ip->p_middle_metadata, ip->p_middle_parallelism,
			ip->p_top_intersect, ip->p_basic_tile_build, ip->p_a_btilebuild_par, ip->p_b_btilebuild_par, ip->p_o_btilebuild_par,
			ip->p_top_search, ip->p_top_search_parallelism, ip->p_top_metadata, ip->p_top_metadata_parallelism,
			ip->p_llb_partition_policy, ip->p_bottom_buffer, ip->p_a_bottom_buffer_perc);

	omp_set_num_threads(params->getNumThreads());
	// Keeps track of cycles, timing, traffic, and bandwidth
	Stats *stats = new Stats();

	LLB_Mem * llb = new LLB_Mem(stats, params, ip->p_llb_size*1024*1024, ip->p_a_llb_perc, ip->p_b_llb_perc, ip->p_o_llb_perc);

	Matrix * mat = new Matrix(ip->p_input1, params, stats, llb);

	Scheduler_8 * sched_8 = new Scheduler_8(mat, params, stats, llb);
	sched_8->Run();

	printf("runtime: %f, cycles: %lu, busy_cycles: %lu\n",
			stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles());

	printf("Top DOT NoC:\n total_top: %f GBs, a_top: %lu, b_top: %lu, o_r_top: %lu, o_w_top: %lu\n",
			(double)stats->Get_total_traffic() / (1024*1024*1024),
			stats->Get_a_read(), stats->Get_b_read(), stats->Get_o_read(), stats->Get_o_write());

	auto total_middle_traffic = stats->Get_a_read_middle() + stats->Get_b_read_middle()
		+ stats->Get_o_read_middle()+ stats->Get_o_write_middle();

	printf("Middle DOT NoC:\n total_mid: %f GBs, a_mid: %lu, b_mid: %lu, o_r_mid: %lu, o_w_mid: %lu\n",
			(double)total_middle_traffic / (double)GB,
			stats->Get_a_read_middle(), stats->Get_b_read_middle(),
			stats->Get_o_read_middle(), stats->Get_o_write_middle());

	//sched_8->PrintBWUsage();

	//SoL_Analytical_Models * sol = new SoL_Analytical_Models(mat, params, stats, llb);
	//sol->model_0();
	//sol->model_1();

	sched_8->ReportPEUtilization();
	return 0;
}

void initializeInputParameters(input_parameters_st *input_params){
	// tiling: {t_static, t_dynamic}, static_distributor: {oracle, round_robin, nnz_based, oracle_relaxed}
	// Static tile dimensions: I, J, K
	// tile_size = 128/tile_dim, pe_count = 128, PE_CAMs = 32, BW_top = 68.25GB/s, BW_middle = 512 GB/s
	// freq = 1GHz, num_threads= 10, a_reuse = 64...
	// intersect Bottom (PE->MACC): {skipModel, naiveModel, idealModel, instantModel},
	// intersect Middle (LLB->PE): {skipModel, idealModel, instantModel}, int par_middle
	// intersect TOP(DRAM->LLB): {skipModel, idealModel, instantModel},
	// Extract Middle (1) search_tiles: {instant, seriali, parallel} (2) mbuild: {instant, serial, parallel}
	// Extract Top (1) tbuild: {noTileBuild, instant, serial, parallel}
	// Parallelism factor for tbuild (tensor A, tensor B, tensor O)
	// Extract Top search_tiles: {instant, serial, parallel} , Parallelism factor
	// Extract Top mbuild: {instant, serial, parallel}, Parallelism factor

	input_params->p_kernel = kernel::SpMSpM; input_params->p_a_format = CSX::CSF;
	input_params->p_b_format = CSX::CSF; input_params->p_o_format = CSX::CSF; input_params->p_dense_cols=1;
	input_params->p_tiling = tiling::t_dynamic; input_params->p_static_dist = static_distributor::round_robin;
	input_params->p_middle_dataflow = arch::outerProdMiddle;
	input_params->p_static_i = 128; input_params->p_static_j = 128; input_params->p_static_k = 128;
	input_params->p_tile_dim = 32; input_params->p_pe_count = 128; input_params->p_cam_nums =33;
	input_params->p_top_bw = 68.25; input_params->p_middle_bw = 2048.0;
	input_params->p_chip_freq = 1000000000; input_params->p_thread_count = 10; input_params->p_a_reuse = 128;
	input_params->p_bottom_intersect= intersect::idealModel; input_params->p_middleDOT_traffic = middleDOTTrafficStatus::yes;
	input_params->p_middle_intersect = intersect::instantModel;	input_params->p_middle_search = search_tiles::instant;
	input_params->p_middle_metadata = metadata_build::instant;
	input_params->p_middle_parallelism = 32;
	input_params->p_top_intersect = intersect::instantModel;
	input_params->p_basic_tile_build = basic_tile_build::instant;
	input_params->p_a_btilebuild_par = 128;	input_params->p_b_btilebuild_par = 128;	input_params->p_o_btilebuild_par = 128;
	input_params->p_top_search = search_tiles::instant;	input_params->p_top_search_parallelism = 32;
	input_params->p_top_metadata = metadata_build::instant;	input_params->p_top_metadata_parallelism = 32;
	// Size of each bottom buffers
	input_params->p_bottom_buffer = 32768;
	// Bottom buffer a percentage (of the total buffer size)
	input_params->p_a_bottom_buffer_perc = 0.5;
	// Static LLB partitioning percentages for a, b, and O
	input_params->p_a_llb_perc = 0.05;	input_params->p_b_llb_perc = 0.5;	input_params->p_o_llb_perc = 0.45;
	// LLB Size
	input_params->p_llb_size = 30;
	// LLB partitioning policy
	input_params->p_llb_partition_policy = llbPartitionPolicy::adaptive_min;

	return;
}
