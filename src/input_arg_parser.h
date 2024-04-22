#ifndef INPUT_ARG_PARSE_H
#define INPUT_ARG_PARSE_H

#include <stdlib.h>
#include <string>
#include <sstream>
#include <getopt.h> /* getopt */
#include "matrix.h"
#include "input_arg_parser.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"
#include "sol_analytical_models.h"


struct input_parameters_st{
	kernel p_kernel;
	CSX p_a_format;
	CSX p_b_format;
	CSX p_o_format;
	int p_dense_cols;
	tiling p_tiling;
	static_distributor p_static_dist;
	arch p_middle_dataflow;
	int p_static_i;
	int p_static_j;
	int p_static_k;
	int p_tile_dim;
	int p_pe_count;
	int p_cam_nums;
	float p_top_bw;
	float p_middle_bw;
	long int p_chip_freq;
	int p_thread_count;
	int p_a_reuse;
	intersect p_bottom_intersect;
	middleDOTTrafficStatus p_middleDOT_traffic;
	intersect p_middle_intersect;
	search_tiles p_middle_search;
	metadata_build p_middle_metadata;
	int p_middle_parallelism;
	intersect p_top_intersect;
	basic_tile_build p_basic_tile_build;
	int p_a_btilebuild_par;
	int p_b_btilebuild_par;
	int p_o_btilebuild_par;
	search_tiles p_top_search;
	int p_top_search_parallelism;
	metadata_build p_top_metadata;
	int p_top_metadata_parallelism;
	// Size of each bottom buffers
	int p_bottom_buffer;
	// Bottom buffer a percentage (of the total buffer size)
	float p_a_bottom_buffer_perc;
	// Input file name and address
	char * p_input1;
	char * p_input2;
	// Static LLB partitioning percentages for a, b, and O
	float p_a_llb_perc;
	float p_b_llb_perc;
	float p_o_llb_perc;
	// LLB Size
	int p_llb_size;
	// LLB static partition update policy
	llbPartitionPolicy p_llb_partition_policy;
};



// A More systematic way to parse the input
void parseInput(int argc, char *argv[], input_parameters_st * input_params){

	int c;

	while (1)
	{
		int option_index = 0;
		static struct option long_options[] =
		{
			{"inp1",	required_argument, NULL,  'x' },
			{"inp2",	required_argument, NULL,  'y' },
			{"tiling", required_argument, NULL, 's'},
			{"staticdist",	required_argument, NULL,  'd' },
			{"aperc",  required_argument, NULL,  'a' },
			{"bperc",  required_argument, NULL,  'b' },
			{"operc",	required_argument, NULL,  'o' },
			{"llbsize",	required_argument, NULL,  'l' },
			{"llbpartition",	required_argument, NULL,  'p' },
			{"intersect",	required_argument, NULL,  'u' },
			{"tiledim",	required_argument, NULL, 't' },
			{"constreuse",	required_argument, NULL, 'r' },
			{"topbw",	required_argument, NULL, 'g' },
			{"middlebw",	required_argument, NULL, 'm' },
			{"itop",	required_argument, NULL, 'i' },
			{"jtop",	required_argument, NULL, 'j' },
			{"ktop",	required_argument, NULL, 'k' },
			{"help", no_argument, NULL, 'h' },
			{NULL,      0,                 NULL,  0 }
		};

		c = getopt_long(argc, argv, "-:x:y:s:i:j:d:a:b:o:l:u:t:r:b:m:h", long_options, &option_index);
		if (c == -1)
			break;

		switch (c)
		{
			case 0:
				printf("long option %s", long_options[option_index].name);
				if (optarg)
					printf(" with arg %s", optarg);
				printf("\n");
				break;

			case 1:
				printf("regular argument '%s'\n", optarg); /* non-option arg */
				break;

			// Input 1 file address
			case 'x':
				input_params->p_input1 = strdup(optarg);
				break;

			// Input 2 file address
			case 'y':
				input_params->p_input2 = strdup(optarg);
				break;

				// Tiling mechanism option
			case 's':
				// Parsing static distributor
				if(strcmp(optarg, "static") == 0){
					input_params->p_tiling = tiling::t_static;
				}
				else if(strcmp(optarg, "dynamic") == 0){
					input_params->p_tiling = tiling::t_dynamic;
				}
				else if(strcmp(optarg, "notiling") == 0){
					input_params->p_tiling = tiling::t_no_tiling;
				}
				else{
					printf("No such distributor found!\n");
					exit(1);
				}
				break;

			// Static Distributor option
			case 'd':
				// Parsing static distributor
				if(strcmp(optarg, "rr") == 0){
					input_params->p_static_dist = static_distributor::round_robin;
				}
				else if(strcmp(optarg, "nnz") == 0){
					input_params->p_static_dist = static_distributor::nnz_based;
				}
				else if(strcmp(optarg, "oracle") == 0){
					input_params->p_static_dist = static_distributor::oracle;
				}
				else if(strcmp(optarg, "oraclerelaxed") == 0){
					input_params->p_static_dist = static_distributor::oracle_relaxed;
				}
				else if(strcmp(optarg, "oraclerelaxedct") == 0){
					input_params->p_static_dist = static_distributor::oracle_relaxed_ct;
				}
				else{
					printf("No such distributor found!\n");
					exit(1);
				}
				break;

			// LLB static partition percentage for a
			case 'a':
				input_params->p_a_llb_perc = atof(optarg);
				break;

			// LLB static partition percentage for b
			case 'b':
				input_params->p_b_llb_perc = atof(optarg);
				break;

			// LLB static partition percentage for o
			case 'o':
				input_params->p_o_llb_perc = atof(optarg);
				break;

			// LLB size in MB
			case 'l':
				input_params->p_llb_size = atoi(optarg);
				break;

			case 'p':
				// LLB partition policy: {constant_static, adaptive_min, adaptive_prev, adaptive_avg}
				if(strcmp(optarg, "const") == 0){
					input_params->p_llb_partition_policy = llbPartitionPolicy::constant_initial;
				}
				else if(strcmp(optarg, "min") == 0){
					input_params->p_llb_partition_policy = llbPartitionPolicy::adaptive_min;
				}
				else if(strcmp(optarg, "avg") == 0){
					input_params->p_llb_partition_policy = llbPartitionPolicy::adaptive_avg;
				}
				else if(strcmp(optarg, "prev") == 0){
					input_params->p_llb_partition_policy = llbPartitionPolicy::adaptive_prev;
				}
				else if(strcmp(optarg, "ideal") == 0){
					input_params->p_llb_partition_policy = llbPartitionPolicy::ideal;
				}

				else{
					printf("No such intersection unit found!\n");
					exit(1);
				}
				break;

			case 'u':
				// Parsing static distributor
				if(strcmp(optarg, "ideal") == 0){
					input_params->p_bottom_intersect = intersect::idealModel;
				}
				else if(strcmp(optarg, "skip") == 0){
					input_params->p_bottom_intersect = intersect::skipModel;
				}
				else if(strcmp(optarg, "parbi") == 0){
					input_params->p_bottom_intersect = intersect::parBidirecSkipModel;
				}
				else{
					printf("No such intersection unit found!\n");
					exit(1);
				}
				break;

			case 't':
				input_params->p_tile_dim = atoi(optarg);
				break;

			case 'r':
				input_params->p_a_reuse = atoi(optarg);
				break;

			case 'g':
				input_params->p_top_bw = atof(optarg);
				break;

			case 'm':
				input_params->p_middle_bw = atof(optarg);
				break;

			case 'i':
				input_params->p_static_i = atoi(optarg);
				break;

			case 'j':
				input_params->p_static_j = atoi(optarg);
				break;

			case 'k':
				input_params->p_static_k = atoi(optarg);
				break;

			case 'h':
				printf("These are the input options:\n" \
				"\t--inp1 or -x: input directory for the first tensor\n" \
				"\t--inp2 or -y: input directory for the second tensor\n" \
				"\t--tiling or -s: tiling mechanism (static, dynamic, notiling)\n" \
				"\t--staticdist or -d: static distributor for the middle DOT (rr, nnz, oracle, oraclerelaxed, oraclerelaxedct)\n" \
				"\t--aperc or -a: LLB percentage assigned for tensor a\n"	\
				"\t--bperc or -b: LLB percentage assigned for tensor b\n" \
				"\t--operc or -o: LLB percentage assigned for tensor o\n" \
				"\t--llbsize or -l: LLB total size\n" \
				"\t--llbpartition or -p: LLB size partitioning policy (const, min, avg, prev)\n" \
				"\t--intersect or -u: Intersect unit for the middle DOT (ideal, skip, parbi)\n" \
				"\t--tiledim or -t: Basic tile dimension\n" \
				"\t--constreuse or -r: Constant a_reuse value\n" \
				"\t--topbw or -g: Top DOT (DRAM) bandwidth in GB/s\n" \
				"\t--middlebw or -m: Middle DOT NoC Bandwidth in GB/s\n" \
				"\t--itop or -i: Staic I for ExTensor top (LLB) tile\n" \
				"\t--jtop or -j: Staic J for ExTensor top (LLB) tile\n" \
				"\t--ktop or -k: Staic K for ExTensor top (LLB) tile\n" \
				"\t--help or -h: Cry for help\n");
				exit(1);
				break;

			case '?':
				printf("Unknown option %c\n", optopt);
				exit(1);
				break;

			case ':':
				printf("Missing option for %c\n", optopt);
				exit(1);

				break;

			default:
				printf("?? getopt returned character code 0%o ??\n", c);
		}
	}

}

#endif
