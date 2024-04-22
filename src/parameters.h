#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

#define GB (1024*1024*1024)
#define MIDDLE_BW_DEF 512.0

enum class llbPartitionPolicy{
	ideal,
	constant_initial,
	adaptive_min,
	adaptive_avg,
	adaptive_prev
};

enum class middleDOTTrafficStatus{
	yes,
	no
};

enum class kernel{
	SpMSpM,
	SpMM,
	TTM,
	TTV,
	SDDMM
};

enum class arch{
	innerProdMiddle,
	outerProdMiddle
};

enum class tiling{
	t_static,
	t_dynamic,
    t_no_tiling
};

// Oracle -> Chooses the PE with lowest cycles that the A basic tile group is pinned to
// Oracle_relaxed_ct (correct traffic) -> Each (basic_tile x basic_tile) is assigned to
//  the lowest PE cycle available. The Top DOT traffic is accurate since they are applied
//  while a data is fetched	or written back
// Oracle_relxed -> Models an ideal on-chip SoL variant. Each {basic_tile x basic_tile} is
//  assigned to the PE with the lowest cycles. The traffic is assumed to be distributed uniformly
//  through the computation. Absolutely no dataflow, everything including data fetch is idealized
enum class static_distributor{
	oracle,
	round_robin,
	nnz_based,
	oracle_relaxed,
	oracle_relaxed_ct
};

enum class intersect{
	parBidirecSkipModel,
	parUnidirecSkipModel,
	skipModel,
	naiveModel,
	idealModel,
	instantModel
};

enum class intersect_dist{
	parallel,
	sequential
};

// First phase of extraction: Basic Tile Build
enum class basic_tile_build{
	instant,
	serial,
	parallel,
	noTilebuild
};

// Second phase of extraction: Search
enum class search_tiles{
	instant,
	serial,
	parallel
};

// Third phase of extraction: Build metadata
enum class metadata_build{
	instant,
	serial,
	parallel
};

// Technically COO is not CSX but anyway :D
enum class CSX{
	CSR,
	CSC,
	CSF,
	COO,
	Dense
};


class Parameters{
	public:

		// USED IN ALL THE TOP FILES
		Parameters(kernel compKernel, CSX a_format, CSX b_format, CSX o_format, int dense_cols,
				tiling tiling_mechanism, static_distributor dealer, arch middleDOTDataflow,
				int I_top, int J_top, int K_top,
				int tile_size, int pe_count, int num_cams, double top_bandwidth, double middle_bandwidth,
				uint64_t frequency,	int num_threads, int a_reuse, intersect intersect_model,
				middleDOTTrafficStatus middleDOTTraffic, intersect intersect_model_middle,
				search_tiles search_middle,	metadata_build m_build_middle,
				int par_middle,
				intersect intersect_model_top,
				basic_tile_build t_build_top,
				int par_t_build_top_A, int par_t_build_top_B, int par_t_build_top_O,
				search_tiles search_top, int par_search_top,
				metadata_build m_build_top, int par_m_build_top,
				llbPartitionPolicy llb_partition, int pe_buffer_size, float a_tensor_perc)
		{
			this->compKernel=compKernel; this->a_format=a_format; this->b_format=b_format;
			this->o_format = o_format; this->dense_cols=dense_cols;

			this->I_top = I_top; this->J_top = J_top; this->K_top = K_top;

			this->tiling_mechanism= tiling_mechanism; this->dealer = dealer;
			this->middleDOTDataflow = middleDOTDataflow;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = middle_bandwidth*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = num_cams; this->a_reuse = a_reuse; this->o_reuse =0;
            		this->b_reuse = a_reuse;
			this->intersect_model = intersect_model;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->middleDOTTraffic = middleDOTTraffic;
			this->intersect_model_middle = intersect_model_middle;
			this->par_middle = par_middle;
			this->intersect_model_top = intersect_model_top;
			this->num_cams_top = 128; /*this->num_cams_middle = 128;*/
			this->t_build_top = t_build_top; this->search_top = search_top;
			this->m_build_top = m_build_top;
			this->par_t_build_top_A = par_t_build_top_A; this->par_t_build_top_B = par_t_build_top_B;
			this->par_t_build_top_O = par_t_build_top_O; this->par_search_top = par_search_top;
			this->par_m_build_top = par_m_build_top;
			this->search_middle = search_middle; this->par_search_middle = 1;
			this->m_build_middle = m_build_middle; this->par_m_build_middle = 1;
			this->pe_buffer_size = pe_buffer_size; this->a_tensor_perc = a_tensor_perc;
			this->llb_partition = llb_partition;
			this->dense_tile_size =
				(this->tile_size) * std::min((this->tile_size), this->dense_cols) * this->data_size;
			this->outerSpaceSim = 0; this->matRaptorSim = 0;
		}


		Parameters(
				tiling tiling_mechanism, static_distributor dealer, arch middleDOTDataflow,
				int tile_size, int pe_count, int num_cams, double top_bandwidth,
				uint64_t frequency,	int num_threads, int a_reuse, intersect intersect_model,
				middleDOTTrafficStatus middleDOTTraffic, intersect intersect_model_middle,
				search_tiles search_middle,	metadata_build m_build_middle,
				int par_middle,
				intersect intersect_model_top,
				basic_tile_build t_build_top,
				int par_t_build_top_A, int par_t_build_top_B, int par_t_build_top_O,
				search_tiles search_top, int par_search_top,
				metadata_build m_build_top, int par_m_build_top,
				int pe_buffer_size, float a_tensor_perc)
		{
			this->tiling_mechanism= tiling_mechanism; this->dealer = dealer;
			this->middleDOTDataflow = middleDOTDataflow;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = MIDDLE_BW_DEF*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = num_cams; this->a_reuse = a_reuse; this->o_reuse =0;
            		this->b_reuse = a_reuse;
			this->intersect_model = intersect_model;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->middleDOTTraffic = middleDOTTraffic;
			this->intersect_model_middle = intersect_model_middle;
			this->par_middle = par_middle;
			this->intersect_model_top = intersect_model_top;
			this->num_cams_top = 128; /*this->num_cams_middle = 128;*/
			this->t_build_top = t_build_top; this->search_top = search_top;
			this->m_build_top = m_build_top;
			this->par_t_build_top_A = par_t_build_top_A; this->par_t_build_top_B = par_t_build_top_B;
			this->par_t_build_top_O = par_t_build_top_O; this->par_search_top = par_search_top;
			this->par_m_build_top = par_m_build_top;
			this->search_middle = search_middle; this->par_search_middle = 1;
			this->m_build_middle = m_build_middle; this->par_m_build_middle = 1;
			this->pe_buffer_size = pe_buffer_size; this->a_tensor_perc = a_tensor_perc;
			this->compKernel = kernel::SpMSpM; this->a_format=CSX::CSF; this->b_format=CSX::CSF;
			this->o_format = CSX::CSF;
		}

		Parameters(int tile_size, int pe_count, int num_cams, double top_bandwidth,
				uint64_t frequency,	int num_threads, int a_reuse, intersect intersect_model,
				intersect intersect_model_top, int num_cams_top, basic_tile_build t_build_top,
				int par_t_build_top_A, int par_t_build_top_B, int par_t_build_top_O,
				search_tiles search_top, int par_search_top,
				metadata_build m_build_top, int par_m_build_top)
		{
			this->tiling_mechanism= tiling::t_dynamic; this->dealer = static_distributor::oracle;
			this->middleDOTDataflow = arch::outerProdMiddle;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = MIDDLE_BW_DEF*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = num_cams; this->a_reuse = a_reuse; this->o_reuse =0;
            		this->b_reuse = a_reuse;
			this->intersect_model = intersect_model;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->intersect_model_top = intersect_model_top;
			this->num_cams_top = num_cams_top;
			this->t_build_top = t_build_top; this->search_top = search_top;
			this->m_build_top = m_build_top;
			this->par_t_build_top_A = par_t_build_top_A; this->par_t_build_top_B = par_t_build_top_B;
			this->par_t_build_top_O = par_t_build_top_O; this->par_search_top = par_search_top;
			this->par_m_build_top = par_m_build_top;
			this->search_middle = search_tiles::instant; this->par_search_middle = 1;
			this->m_build_middle = metadata_build::instant; this->par_m_build_middle = 1;
			this->pe_buffer_size = 65536; this->a_tensor_perc = 0.5;
			this->compKernel = kernel::SpMSpM; this->a_format=CSX::CSF; this->b_format=CSX::CSF;
			this->o_format = CSX::CSF;

		}


		Parameters(int tile_size, int pe_count, int num_cams, double top_bandwidth,
				uint64_t frequency,	int num_threads, int a_reuse, intersect intersect_model,
				intersect intersect_model_top, int num_cams_top)
		{
			this->tiling_mechanism= tiling::t_dynamic; this->dealer = static_distributor::oracle;
			this->middleDOTDataflow = arch::outerProdMiddle;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = MIDDLE_BW_DEF*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = num_cams; this->a_reuse = a_reuse; this->o_reuse =0;
            		this->b_reuse = a_reuse;
			this->intersect_model = intersect_model;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->intersect_model_top = intersect_model_top;
			this->num_cams_top = num_cams_top;
			this->t_build_top = basic_tile_build::noTilebuild; this->search_top = search_tiles::instant;
			this->m_build_top = metadata_build::instant;
			this->par_t_build_top_A = 1; this->par_t_build_top_B = 1;
			this->par_t_build_top_O = 1; this->par_search_top = 1;
			this->par_m_build_top = 1;
			this->search_middle = search_tiles::instant; this->par_search_middle = 1;
			this->m_build_middle = metadata_build::instant; this->par_m_build_middle = 1;
			this->pe_buffer_size = 65536; this->a_tensor_perc = 0.5;
			this->compKernel = kernel::SpMSpM; this->a_format=CSX::CSF; this->b_format=CSX::CSF;
			this->o_format = CSX::CSF;

		}

		Parameters(int tile_size, int pe_count, int num_cams, double top_bandwidth,
				uint64_t frequency,	int num_threads, int a_reuse, intersect intersect_model)
		{
			this->tiling_mechanism= tiling::t_dynamic; this->dealer = static_distributor::oracle;
			this->middleDOTDataflow = arch::outerProdMiddle;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = MIDDLE_BW_DEF*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = num_cams; this->a_reuse = a_reuse; this->o_reuse =0;
           		this->b_reuse = a_reuse;
			this->intersect_model = intersect_model;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->intersect_model_top = intersect::instantModel;
			this->num_cams_top = 32;
			this->t_build_top = basic_tile_build::noTilebuild; this->search_top = search_tiles::instant;
			this->m_build_top = metadata_build::instant;
			this->par_t_build_top_A = 1; this->par_t_build_top_B = 1;
			this->par_t_build_top_O = 1; this->par_search_top = 1;
			this->par_m_build_top = 1;
			this->search_middle = search_tiles::instant; this->par_search_middle = 1;
			this->m_build_middle = metadata_build::instant; this->par_m_build_middle = 1;
			this->pe_buffer_size = 65536; this->a_tensor_perc = 0.5;
			this->compKernel = kernel::SpMSpM; this->a_format=CSX::CSF; this->b_format=CSX::CSF;
			this->o_format = CSX::CSF;
		}

		Parameters(int tile_size, int pe_count, double top_bandwidth,
				uint64_t frequency, int num_threads, int a_reuse, intersect intersect_model)
		{
			this->tiling_mechanism= tiling::t_dynamic; this->dealer = static_distributor::oracle;
			this->middleDOTDataflow = arch::outerProdMiddle;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = MIDDLE_BW_DEF*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = 32; this->a_reuse = a_reuse; this->o_reuse =0;
            		this->b_reuse = a_reuse;
			this->intersect_model = intersect_model;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->intersect_model_top = intersect::instantModel;
			this->num_cams_top = 32;
			this->t_build_top = basic_tile_build::noTilebuild; this->search_top = search_tiles::instant;
			this->m_build_top = metadata_build::instant;
			this->par_t_build_top_A = 1; this->par_t_build_top_B = 1;
			this->par_t_build_top_O = 1; this->par_search_top = 1;
			this->par_m_build_top = 1;
			this->search_middle = search_tiles::instant; this->par_search_middle = 1;
			this->m_build_middle = metadata_build::instant; this->par_m_build_middle = 1;
			this->pe_buffer_size = 65536; this->a_tensor_perc = 0.5;
			this->compKernel = kernel::SpMSpM; this->a_format=CSX::CSF; this->b_format=CSX::CSF;
			this->o_format = CSX::CSF;

		}

		void setParameters(int tile_size, int pe_count, double top_bandwidth, int frequency)
		{
			this->tiling_mechanism= tiling::t_dynamic; this->dealer = static_distributor::oracle;
			this->middleDOTDataflow = arch::outerProdMiddle;
			this->tile_size = tile_size; this->pe_count = pe_count;
			this->top_bandwidth = top_bandwidth*GB; this->middle_bandwidth = MIDDLE_BW_DEF*GB;
			this->frequency = frequency; this->num_threads = num_threads;
			this->data_size = 8; this->idx_size = 4; this->pos_size = 4;
			this->num_cams = 32; this->a_reuse = 128; this->o_reuse =0;
            		this->b_reuse = 128;
			this->intersect_model = intersect::skipModel;
			this->intersect_model_dist = intersect_dist::sequential;
			this->intersect_parallelism = 1; this->num_worker_threads = 1;
			this->intersect_model_top = intersect::instantModel;
			this->num_cams_top = 32;
			this->t_build_top = basic_tile_build::noTilebuild; this->search_top = search_tiles::instant;
			this->m_build_top = metadata_build::instant;
			this->par_t_build_top_A = 1; this->par_t_build_top_B = 1;
			this->par_t_build_top_O = 1; this->par_search_top = 1;
			this->par_m_build_top = 1;
			this->search_middle = search_tiles::instant; this->par_search_middle = 1;
			this->m_build_middle = metadata_build::instant; this->par_m_build_middle = 1;
			this->pe_buffer_size = 65536; this->a_tensor_perc = 0.5;
			this->compKernel = kernel::SpMSpM; this->a_format=CSX::CSF; this->b_format=CSX::CSF;
			this->o_format = CSX::CSF;

		}


		tiling getTilingMechanism(){return this->tiling_mechanism;}
		static_distributor getStaticDistributorModelMiddle(){return this->dealer;}
		arch getMiddleDOTDataflow(){return this->middleDOTDataflow;}
		int getITopTile(){return this->I_top;}
		int getJTopTile(){return this->J_top;}
		int getKTopTile(){return this->K_top;}
		int getPECount(){return pe_count;}
		int getTileSize(){return tile_size;}
		// Legacy code support function
		double getBandwidth(){return top_bandwidth;}
		double getTopBandwidth(){return top_bandwidth;}
		double getMiddleBandwidth(){return middle_bandwidth;}
		uint64_t getFrequency(){return frequency;}
		int getNumThreads(){return num_threads;}
		int getNumWorkerThreads(){return num_worker_threads;}
		int getDataSize(){return data_size;}
		int getIdxSize(){return idx_size;}
		int getPosSize(){return pos_size;}
		int getNumCAMs(){return num_cams;}
		int getNumCAMsLLB(){return num_cams_top;}
		int getAReuse(){return a_reuse;}
        int getBReuse(){return b_reuse;}
		int getOReuse(){return o_reuse;}
		// Bottom Level Intersection (PE->MACC units)
		intersect getIntersectModel(){return this->intersect_model;}
		intersect_dist getIntersectModelDist(){return this->intersect_model_dist;}
		int getIntersectParallelism(){return this->intersect_parallelism;}

		// Top Level Intersection (DRAM->LLB)
		intersect getIntersectModelTop(){return this->intersect_model_top;}
		// Top Level Extract (DRAM->LLB)
		basic_tile_build getBasicTileBuildModelTop(){return this->t_build_top;}
		search_tiles getSearchModelTop(){return this->search_top;}
		metadata_build getMetadataBuildModelTop(){return this->m_build_top;}
		int getParallelismBasicTileBuildTop_A(){return this->par_t_build_top_A;}
		int getParallelismBasicTileBuildTop_B(){return this->par_t_build_top_B;}
		int getParallelismBasicTileBuildTop_O(){return this->par_t_build_top_O;}
		int getParallelismSearchTop(){return this->par_search_top;}
		int getParalellismMetadataBuildTop(){return this->par_m_build_top;}

		// Middle Level Intersection (LLB->PE)
		intersect getIntersectModelMiddle(){return this->intersect_model_middle;}
		// Middle Level Extract (LLB->PE)
		search_tiles getSearchModelMiddle(){return this->search_middle;}
		metadata_build getMetadataBuildModelMiddle(){return this->m_build_middle;}
		int getParallelismSearchMiddle(){return this->par_search_middle;}
		int getParalellismMetadataBuildlMiddle(){return this->par_m_build_middle;}

		int getParallelismMiddle(){return this->par_middle;}
		// Middle level NoC traffic
		middleDOTTrafficStatus doesMiddleDOTTrafficCount(){return this->middleDOTTraffic;}
		// PE buffer
		int getPEBufferSize(){return this->pe_buffer_size;}
		float getATensorPercPEBuffer(){return this->a_tensor_perc;}
		// LLB Buffer
		llbPartitionPolicy getLLBPartitionPolicy(){return this->llb_partition;}
		// Maximum 20% of LLB can be used for A
		// This is used for SoL model where LLB partitioning is always ideal
		float getAMaxPercLLB(){return (float)0.2;}

		void setITopTile(int I_top){this->I_top = I_top; return;}
		void setJTopTile(int J_top){this->J_top = J_top; return;}
		void setKTopTile(int K_top){this->K_top = K_top; return;}
		void setIJKTopTile(int I_top, int J_top, int K_top)
			{this->I_top = I_top; this->J_top = J_top; this->K_top = K_top; return;}
		void setPECount(int pe_count){this->pe_count = pe_count; return;}
		void setNumCAMs(int num_cams){this->num_cams = num_cams; return;}
		void setNumCAMsTop(int num_cams_top){this->num_cams_top = num_cams_top; return;}
		void setOReuse(int o_reuse){this->o_reuse = o_reuse; return;}
		void setAReuse(int a_reuse){this->a_reuse = a_reuse; return;}
        	void setBReuse(int b_reuse){this->b_reuse = b_reuse; return;}
		// Legacy code support function
		void setBandwidth(double top_bandwidth){this->top_bandwidth = top_bandwidth*GB; return;}
		void setTopBandwidth(double top_bandwidth){
			this->top_bandwidth = top_bandwidth*GB; return;}
		void setMiddleBandwidth(double middle_bandwidth){
			this->middle_bandwidth = middle_bandwidth*GB; return;}
		void setIntersectModel(intersect intersect_model){
			this->intersect_model = intersect_model; return;}
		void setIntersectModelLLB(intersect intersect_model_top){
			this->intersect_model_top = intersect_model_top; return;}
		void setIntersectDist(intersect_dist intersect_model_dist){
			this->intersect_model_dist = intersect_model_dist; return;}
		void setIntersectParallelism(int intersect_parallelism){
			this->intersect_parallelism = intersect_parallelism; return;}

		// SpMM related functions
		kernel getCompKernel(){return this->compKernel;}
		CSX getAFormat(){return this->a_format;}
		CSX getBFormat(){return this->b_format;}
		CSX getOFormat(){return this->o_format;}
		int getNumDenseCols(){return this->dense_cols;}
		int getDenseTileSize(){
			return dense_tile_size;
		}
		// OuterSpace related function
		int IsOuterSpaceSim(){return this->outerSpaceSim;}
		void SetOuterSpaceSim(){this->outerSpaceSim = 1;}
		// MatRaptor related function
		int IsMatRaptorSim(){return this->matRaptorSim;}
		void SetMatRaptorSim(){this->matRaptorSim = 1;}


	private:
		// Tiling mechanism for top and middle DOTs.
		//	Either static (ExTensor) or dynamic reflexive (TACTile)
		tiling tiling_mechanism;
		// Static distributo (dealer)
		static_distributor dealer;
		// Status of static llb partition; whether it is going to stay constant throughout
		//   the run-time or it is going to change
		llbPartitionPolicy llb_partition;
		// The top tile static sizes (LLB tile I, J, K)
		int I_top, J_top, K_top;
		// number of PE units
		int pe_count;
		// bandwidth in bytes
		double top_bandwidth;
		double middle_bandwidth;
		// Do we need to count middle DOT traffic
		middleDOTTrafficStatus middleDOTTraffic;
		// frequency of the chip
		uint64_t frequency;
		// PE tile size
		int tile_size;
		// number of threads used for computations
		int num_threads;
		// Number of worker threads for doing parallel matrix mult.
		//   for different tiles
		int num_worker_threads;
		// Size of the pos in metadata (CSR/CSF/...)
		int pos_size;
		// Size of data entries in tensors
		int data_size;
		// Size of the idx in metadata (CSR/CSF/...)
		int idx_size;
		// Number of CAMS for intersect unit in PE
		int num_cams;
		// Number of CAMS for intersect unit in Top level (DRAM->LLB)
		int num_cams_top;

		// A reuse for dynamic tiling
		int a_reuse;
        	// B reuse for dynamic tiling
        	int b_reuse;
		// O reuse for dynamic tiling; Computed during scheduling part
		int o_reuse;

		// Intersect mode skipModel/naiveModel/idealModel in PE
		intersect intersect_model;
		// Intersect mode skipModel/naiveModel/idealModel in Top level (DRAM->LLB)
		intersect intersect_model_top;
		// Intersect unit distribution among PE units sequential/parallel
		intersect_dist intersect_model_dist;
		// Intersect parallelism ways; Does the parallelism in n units
		int intersect_parallelism;
		// Basic tile build model in LLB tile extraction
		basic_tile_build t_build_top;
		// Parallelism factor in basic tile build LLB
		int par_t_build_top_A;
		int par_t_build_top_B;
		int par_t_build_top_O;
		// search model in LLB tile extraction
		search_tiles search_top;
		// Parallelism factor in search LLB
		int par_search_top;
		// Metadata build model in LLB tile extraction
		metadata_build m_build_top;
		// Parallelism factor in metadata build LLB
		int par_m_build_top;
		// Intersect mode skipModel/naiveModel/idealModel in the Middle level (LLB->PE)
		intersect intersect_model_middle;
		// search model in PE tile extraction
		search_tiles search_middle;
		// Parallelism factor in search PE
		int par_search_middle;
		// Metadata build model in PE tile extraction
		metadata_build m_build_middle;
		// Parallelism factor in metadata build PE
		int par_m_build_middle;

		int par_middle;
		// PE buffer size in bytes
		int pe_buffer_size;
		// Percentage of PE buffer that belongs to tensor A
		float a_tensor_perc;
		// The dataflow in the middle DOT:
		//	it is either outer product (TACTile) or inner product (ExTensor)
		arch middleDOTDataflow;
		// Defines the computation kernel: SpMSpM, SpMM, ...
		kernel compKernel;
		// SpMM related variables
		CSX a_format;
		CSX b_format;
		CSX o_format;
		int dense_cols;
		int dense_tile_size;
		// Outerspace related experiment
		int outerSpaceSim;
		int matRaptorSim;
};

#endif
