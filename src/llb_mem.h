#ifndef LLB_MEM_H_
#define LLB_MEM_H_

#include <stdlib.h>
#include <stdio.h>

#include "stats.h"
#include "parameters.h"
// LLB request types; Either read or write
enum class Req{
	read,
	write
};

class LLB_Mem{
	public:

		// Constructor 1: Specify everything; This is being used by default in runs
		LLB_Mem(Stats *stats, Parameters * params,
				uint64_t total_size, float ratio_a, float ratio_b, float ratio_o)
			{this->total_size = total_size; this->max_used_size = 0;
				this->used_size = 0; this->a_size = 0; this->b_size = 0;
				this->o_size = 0;	this->ratio_a = ratio_a; this->ratio_b = ratio_b;
				this-> ratio_o = ratio_o;

				this->stats = stats; this->params = params;
				// Calculate the capacity of each tensor proportion in LLB
				max_a_size = (uint64_t) (ratio_a*total_size);
				max_b_size = (uint64_t) (ratio_b*total_size);
				max_o_size = (uint64_t) (ratio_o*total_size);

				//printf("max sizes(a,b,o): (%lu, %lu, %lu)\n", max_a_size, max_b_size, max_o_size);
				return; }

		// Constructor 2: LLB proportions are hard-coded in the connstructor
		// (a,b,o) = (20%, 30%, 50%) LLB
		LLB_Mem(Stats *stats, Parameters * params, uint64_t total_size)
			{this->total_size = total_size; this->max_used_size = 0;
				this->used_size = 0; this->a_size = 0; this->b_size = 0;
				this->o_size = 0;	this->ratio_a = 0.2; this->ratio_b = 0.3;
				this-> ratio_o = 0.5;

				this->stats = stats; this->params = params;

				max_a_size = (uint64_t) (ratio_a*total_size);
				max_b_size = (uint64_t) (ratio_b*total_size);
				max_o_size = (uint64_t) (ratio_o*total_size);

				return; }

		// Constructor 3: Everything is hard-coded in
		// 30 MB default; (a,b,o) = (20%, 30%, 50%) LLB
		LLB_Mem(Stats *stats, Parameters *params)
			{this->total_size = 30*1024*1024; this->max_used_size = 0;
				this->used_size = 0; this->a_size = 0; this->b_size = 0;
				this->o_size = 0;	this->ratio_a = 0.2; this->ratio_b = 0.3;
				this-> ratio_o = 0.5;

				this->stats = stats; this->params = params;

				max_a_size = (uint64_t) ratio_a*total_size;
				max_b_size = (uint64_t) ratio_b*total_size;
				max_o_size = (uint64_t) ratio_o*total_size;

				return; }

		// Reset the LLB partitions (used for adaptive LLB partitioning)
		void SetRatios(float ratio_a, float ratio_b, float ratio_o){
			this->ratio_a = ratio_a;
			this->ratio_b = ratio_b;
			this->ratio_o = ratio_o;
			// Calculate the capacity of each tensor proportion in LLB
			max_a_size = (uint64_t) (ratio_a*total_size);
			max_b_size = (uint64_t) (ratio_b*total_size);
			max_o_size = (uint64_t) (ratio_o*total_size);

		}
		// Reset the LLB partitions (used for adaptive LLB partitioning)
		void SetSizes(uint64_t max_a_size, uint64_t max_b_size, uint64_t max_o_size){
			this->max_a_size = max_a_size;
			this->max_b_size = max_b_size;
			this->max_o_size = max_o_size;

		}

		// Get the used size of the LLB
		uint64_t GetSize(){return used_size;}
		// Get the maximum capacity of the LLB
		uint64_t GetCapacity(){return total_size;}
		// Get the maximum ever used of the LLB
		uint64_t GetMaxUsedSize(){return this->max_used_size;}

		// Get the used size of A part of the LLB
		uint64_t GetASize(){return a_size;}
		// Get the used size of B part of the LLB
		uint64_t GetBSize(){return b_size;}
		// Get the used size of the LLB
		uint64_t GetOSize(){return o_size;}

		uint64_t GetMaxOSize(){return max_o_size;}

		// Get the used size of A part of the LLB
		float GetARatio(){return ratio_a;}
		// Get the used size of B part of the LLB
		float GetBRatio(){return ratio_b;}
		// Get the used size of the LLB
		float GetORatio(){return ratio_o;}

		// Add a PE tile(s) to LLB
		int AddToLLB(char mat_name, Req rd_wr, uint64_t bytes, int update_traffic);

		// Just check if we can fit it or not; Do not do any action
		// 0: no , 1: yes
		int DoesFitInLLB(char mat_name, uint64_t bytes);

		// Remove #bytes from the LLB portion of a specific tensor
		void RemoveFromLLB(char mat_name, Req rd_wr, uint64_t bytes, int update_traffic);

		// Completely erase the tiles corresponding to a matrix from LLB
		void EvictMatrixFromLLB(char mat_name, int update_traffic);

		// Add data to LLB; This is a minimal version used for no-tiled versions
		int AddToLLB_Minimal(uint64_t bytes);

		// Remove #bytes from LLB ; This is a minimal version used for no-tiled versions
		void RemoveFromLLB_Minimal(uint64_t bytes);

		// Checks if a specific size of data can fit into LLB
		// The routine will not take any action
		// returns 0 upon a failure and 1 otherwise
		//  ; This is a minimal version used for no-tiled versions
		int DoesFitInLLB_Minimal(uint64_t bytes);

		void Reset(){
				this->max_used_size = 0;  this->used_size = 0;
				this->a_size = 0; this->b_size = 0;
				return;
		}

	private:
		// Update stats as well
		Stats * stats;
		Parameters *params;
		// Capacity of the LLB
		uint64_t total_size;
		// Used size of the LLB
		uint64_t used_size;
		// Used size of corresponding LLB size to a/b/o tensors
		uint64_t a_size;
		uint64_t b_size;
		uint64_t o_size;
		// A log variable to record the maximum ever used size of LLB
		uint64_t max_used_size;
		// Capacity of LLB for a/b/o tensors
		uint64_t max_a_size;
		uint64_t max_b_size;
		uint64_t max_o_size;
		// ratio of a/b/o tensor LLB portitions to total_size
		float ratio_a;
		float ratio_b;
		float ratio_o;
};

#endif
