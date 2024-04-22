#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "scheduler_SpMM_8.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"
#include "sol_analytical_models.h"

using namespace std;
int main(int argc, char *argv[])
{

	if (argc < 3)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename] [a_reuse] [tile_dim]\n", argv[0]);
		exit(1);
	}
	// Getting the a_reuse constant value; o_reuse is derived from a_value and LLB limit
	int a_reuse = atoi(argv[2]);
	int tile_dim = atoi(argv[3]);
	int dense_col_size = atoi(argv[4]);
	/*
	int i_top = atoi(argv[4]);
	int j_top = atoi(argv[5]);
	int k_top = atoi(argv[6]);
	*/
	//int cam_nums = atoi(argv[4]);
	//int extraction_par_factor = atoi(argv[4]);
	//int btile_build_par_factor = atoi(argv[4]);

	// Computation kernel:{SpMSpM, SpMM}, a_format, b_format, o_format: {CSF,CSR,Dense,...}, dense_tensor_cols
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
 	Parameters * params= new Parameters(kernel::SpMM, CSX::CSF, CSX::Dense, CSX::Dense, dense_col_size,
			tiling::t_dynamic, static_distributor::oracle, arch::outerProdMiddle,
			128, 128, 128,
			//i_top, j_top, k_top,
			//tile_dim, 128, cam_nums, 68.25, 2048.0 , 1000000000, 10, a_reuse,
			tile_dim, 128, 33, 68.25, 2048.0 , 1000000000, 10, a_reuse,
			// Bottom
			//intersect::skipModel,
			//intersect::parBidirecSkipModel,
			intersect::idealModel,
			// Middle
			intersect::instantModel, search_tiles::instant, metadata_build::instant, 32,
			//intersect::instantModel, search_tiles::parallel, metadata_build::parallel, 32,
			//intersect::instantModel, search_tiles::parallel, metadata_build::parallel, extraction_par_factor,
			// Top
			intersect::instantModel, basic_tile_build::instant, 128, 128, 128,
			//intersect::instantModel, basic_tile_build::parallel, btile_build_par_factor, btile_build_par_factor, btile_build_par_factor,
			//intersect::instantModel, basic_tile_build::parallel, 128, 128, 128,
			search_tiles::instant, 32, metadata_build::instant, 32,
			//search_tiles::parallel, 32, metadata_build::parallel, 32,
			//search_tiles::parallel, extraction_par_factor, metadata_build::parallel, extraction_par_factor,
			32768, 0.5);
			//262768, 0.5);

	//Parameters * params= new Parameters(32, 128, 68.25, 1000000000, 80, a_reuse,
	//		intersect::idealModel, intersect_dist::sequential);
	//Parameters * params= new Parameters(128, 128, 68.25, 1000000000, 80, a_reuse, intersect::skipModel);
	omp_set_num_threads(params->getNumThreads());
	// Keeps track of cycles, timing, traffic, and bandwidth
	Stats *stats = new Stats();
	// 30MB LLB memory
	LLB_Mem * llb = new LLB_Mem(stats, params, 30*1024*1024, 0.01, 0.5, 0.94);

	//Matrix * mat = new Matrix(argv[1], params, stats, llb);
	//This is created for SpMM
	Matrix * mat = new Matrix(argv[1], 32, params, stats, llb);
	mat->PrintTileSizeHist();
	exit(1);

	//	Tests passed
	// mat->InputTilingDebug();
	// mat->TableDebug();
	// mat->PrintNNZTiles();


	Scheduler_SpMM_8 * sched_8 = new Scheduler_SpMM_8(mat, params, stats, llb);
	sched_8->Run();
	if(stats->Get_cycles()){
	printf("runtime: %f, cycles: %lu, busy_cycles: %lu\n",
			stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles());
/*
	printf("Top DOT NoC:\n total_top: %f GBs, a_top: %lu, b_top: %lu, o_r_top: %lu, o_w_top: %lu\n",
			(double)stats->Get_total_traffic() / (1024*1024*1024),
			stats->Get_a_read(), stats->Get_b_read(), stats->Get_o_read(), stats->Get_o_write());

	auto total_middle_traffic = stats->Get_a_read_middle() + stats->Get_b_read_middle()
		+ stats->Get_o_read_middle()+ stats->Get_o_write_middle();

	printf("Middle DOT NoC:\n total_mid: %f GBs, a_mid: %lu, b_mid: %lu, o_r_mid: %lu, o_w_mid: %lu\n",
			(double)total_middle_traffic / (double)GB,
			stats->Get_a_read_middle(), stats->Get_b_read_middle(),
			stats->Get_o_read_middle(), stats->Get_o_write_middle());
*/
	sched_8->PrintBWUsage();

	SoL_Analytical_Models * sol = new SoL_Analytical_Models(mat, params, stats);
	sol->model_0();
	sol->model_1();
	}

	//mat->PrintNNZTilesAndFiberHist();

	return 0;
}
