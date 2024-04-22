#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "scheduler_3.h"
#include "scheduler_4.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"
#include "sol_analytical_models.h"

using namespace std;
int main(int argc, char *argv[])
{

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	// Getting the a_reuse constant value; o_reuse is derived from a_value and LLB limit
	int a_reuse = atoi(argv[2]);
	// tile_size = 128, pe_count = 128, BW = 68.25GB/s, freq = 1GHz, num_threads= 80, a_reuse = 1-64
	// intersect: skipModel, naiveModel, idealModel
	Parameters * params= new Parameters(128, 128, 68.25, 1000000000, 80, a_reuse, 
			intersect::skipModel, intersect_dist::parallel);
	//Parameters * params= new Parameters(128, 128, 68.25, 1000000000, 80, a_reuse, intersect::skipModel);
	omp_set_num_threads(params->getNumThreads());
	// Keeps track of cycles, timing, traffic, and bandwidth 
	Stats *stats = new Stats();
	// 30MB LLB memory
	LLB_Mem * llb = new LLB_Mem(stats, 30*1024*1024, 0.1, 0.5, 0.4);

	Matrix * mat = new Matrix(argv[1], params, stats, llb);

	Scheduler_4 * sched_4 = new Scheduler_4(mat, params, stats, llb);
	sched_4->Run();

	printf("runtime: %f, cycles: %lu, busy_cycles: %lu, total traffic: %f GBs,\n"
		 	"a traffic %lu, b traffic: %lu, o traffic r/w : %lu, %lu\n",
			stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles(), 
			(double)stats->Get_total_traffic() / (1024*1024*1024), stats->Get_a_read(), stats->Get_b_read(),
			stats->Get_o_read(), stats->Get_o_write());
  
	SoL_Analytical_Models * sol = new SoL_Analytical_Models(mat, params, stats);
	sol->model_0();
	sol->model_1();

	// Simulation number 2
	params->setBandwidth(2*68.256);
	stats->Reset();	llb->Reset();	mat->Reset();	sched_4->Reset();
	sched_4->Run();
	printf("\nSimulation 2: BW %f\n",params->getBandwidth());
	printf("runtime: %f, cycles: %lu, busy_cycles: %lu, total traffic: %f GBs,\n"
			"a traffic %lu, b traffic: %lu, o traffic r/w : %lu, %lu\n",
			stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles(), 
			(double)stats->Get_total_traffic() / (1024*1024*1024), stats->Get_a_read(), stats->Get_b_read(),
			stats->Get_o_read(), stats->Get_o_write());
	sol->model_0();
	sol->model_1();

	// Simulation number 3
	params->setBandwidth(4*68.256);
	stats->Reset();	llb->Reset();	mat->Reset();	sched_4->Reset();
	sched_4->Run();
	printf("\nSimulation 3: BW %f\n",params->getBandwidth());
	printf("runtime: %f, cycles: %lu, busy_cycles: %lu, total traffic: %f GBs,\n"
			"a traffic %lu, b traffic: %lu, o traffic r/w : %lu, %lu\n",
			stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles(), 
			(double)stats->Get_total_traffic() / (1024*1024*1024), stats->Get_a_read(), stats->Get_b_read(),
			stats->Get_o_read(), stats->Get_o_write());
	sol->model_0();
	sol->model_1();

	// Simulation number 4
	params->setBandwidth(8*68.256);
	stats->Reset();	llb->Reset();	mat->Reset();	sched_4->Reset();
	sched_4->Run();
	printf("\nSimulation 4: BW %f\n",params->getBandwidth());
	printf("runtime: %f, cycles: %lu, busy_cycles: %lu, total traffic: %f GBs,\n"
			"a traffic %lu, b traffic: %lu, o traffic r/w : %lu, %lu\n",
			stats->Get_runtime(), stats->Get_cycles(), stats->Get_pe_busy_cycles(), 
			(double)stats->Get_total_traffic() / (1024*1024*1024), stats->Get_a_read(), stats->Get_b_read(),
			stats->Get_o_read(), stats->Get_o_write() );
	sol->model_0();
	sol->model_1();

	return 0;
}

