#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "scheduler_3.h"
#include "scheduler_4.h"
#include "scheduler_7.h"
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
	//int a_reuse = atoi(argv[2]);
	int tile_dim = atoi(argv[2]);

	// tiling: {t_static, t_dynamic}, static_distributor: {oracle, round_robin}
	// Static tile dimensions: I, J, K
	// tile_size = 128/tile_dim, pe_count = 128, PE_CAMs = 32, BW = 68.25GB/s,
	// freq = 1GHz, num_threads= 10, a_reuse = 64...
	// intersect Bottom (PE->MACC): {skipModel, naiveModel, idealModel, instantModel},
	// intersect Middle (LLB->PE): {skipModel, idealModel, instantModel}, int par_middle
	// intersect TOP(DRAM->LLB): {skipModel, idealModel, instantModel},
	// Extract Middle (1) search_tiles: {instant, serial} (2) mbuild: {instant, serial}
	// Extract Top (1) tbuild: {noTileBuild, instant, serial, parallel}
	// Parallelism factor for tbuild (tensor A, tensor B, tensor O)
	// Extract Top search_tiles: {instant, serial, parallel} , Parallelism factor
	// Extract Top mbuild: {instant, serial, parallel}, Parallelism factor
 	Parameters * params= new Parameters(tiling::t_static, static_distributor::oracle,
			32, 64, 128,
			tile_dim, 128, 31, 68.25, 1000000000, 10, 128,
			intersect::idealModel,
			intersect::instantModel, search_tiles::instant, metadata_build::instant, 32,
			intersect::instantModel, basic_tile_build::instant, 128, 128, 128,
			search_tiles::instant, 32, metadata_build::instant, 32,
			32768, 0.5);

	//Parameters * params= new Parameters(32, 128, 68.25, 1000000000, 80, a_reuse,
	//		intersect::idealModel, intersect_dist::sequential);
	//Parameters * params= new Parameters(128, 128, 68.25, 1000000000, 80, a_reuse, intersect::skipModel);
	omp_set_num_threads(params->getNumThreads());
	// Keeps track of cycles, timing, traffic, and bandwidth
	Stats *stats = new Stats();
	// 30MB LLB memory
	LLB_Mem * llb = new LLB_Mem(stats, 30*1024*1024, 0.1, 0.5, 0.4);

	Matrix * mat = new Matrix(argv[1], params, stats, llb);

	Scheduler_7 * sched_7 = new Scheduler_7(mat, params, stats, llb);

	int i_idx [] = {16,24,32};
	int j_idx [] = {48,64,96};
	int k_idx [] = {64,96,128};
	for(int i=0; i < 3; i++)
		for(int j=0; j < 3; j++)
			for(int k=0; k < 3; k++){

				params->setIJKTopTile(i_idx[i], j_idx[j], k_idx[k]);
				if(sched_7->Run()){
					//printf("%d, %d, %d\n\n", i_idx[i], j_idx[j], k_idx[k]);
					//printf("Static LLB tiles do not fit in LLB I:%d, J:%d, K:%d\n",
					//		params->getITopTile(), params->getJTopTile(), params->getKTopTile());
				}
				else{
					printf("%d, %d, %d\n", i_idx[i], j_idx[j], k_idx[k]);
					printf("%f\n",stats->Get_runtime());
					/*
					printf("runtime: %f, cycles: %lu, busy_cycles: %lu, total traffic: %f GBs,\n"
						 	"a traffic %lu, b traffic: %lu, o traffic r/w : %lu, %lu\n",
							stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles(),
							(double)stats->Get_total_traffic() / (1024*1024*1024), stats->Get_a_read(), stats->Get_b_read(),
							stats->Get_o_read(), stats->Get_o_write());
					*/
					//sched_7->PrintBWUsage();
					//SoL_Analytical_Models * sol = new SoL_Analytical_Models(mat, params, stats);
					//sol->model_0();
					//sol->model_1();
				}
				stats->Reset(); llb->Reset(); mat->Reset(); sched_7->Reset();
			}

	return 0;
}
