#ifndef SOL_ANANLYTICAL_MODELS_H
#define SOL_ANANLYTICAL_MODELS_H

#include "matrix.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"
#include <cmath>
// This class gets the stats and matrix produced results and
// based on them calculates model 0 & 1 runtimes

class SoL_Analytical_Models{
	public:
		SoL_Analytical_Models(Matrix * matrix, Parameters * params, Stats * stats, LLB_Mem * llb){
			this->matrix = matrix; this->stats = stats; this->params = params; this->llb = llb;
		}
		~SoL_Analytical_Models();

		// MODEL 0 is a theoretical maximum speed-up (SoL) that takes into
		//   account only the number of MACC units. It finds the total number of
		//   multiplications and divide them by the number of available PEs
		void model_0();

		// MODEL 1 is a theoretical maximum speed-up (SoL) that takes into
		//   account only the bandwidth limit constraint. It finds the total size of
		//   the matrices A, B, and O iff we have only one pass over each of them
		//   (the case of infinity LLB memory); Then, divides them by the peak bandwidth
		//   (baseline is 68.256GB/s)
		void model_1();

		void model_2();

		void model_1_lazyFetch();

		void model_1_lazyFetch_no_output();

	private:
		Matrix * matrix;
		Parameters * params;
		Stats * stats;
		LLB_Mem * llb;
};

#endif
