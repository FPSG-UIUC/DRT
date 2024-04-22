#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <vector>
#include <set>
#include <list>
#include <omp.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <queue>
#include <unordered_map>
#include <numeric>
#include <limits.h>
#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_spblas.h>

#include "mmio.h"
#include "parameters.h"
#include "stats.h"
#include "llb_mem.h"

//using namespace std;

// ALIGN is necessary for MKL csrXcsr; otherwise, it will cause segfault
#define ALIGN 128
#define EOS_MSG INT_MAX
struct COO_format{
	int * rows;
	int * cols;
	double * vals;
	int nnz;
	int row_size;
	int col_size;
};

struct CSR_format{
	int * idx;
	int * pos;
	double * vals;
	int nnz;
	int nnr;
	int row_size;
	int col_size;
	int csfSize;
	int cooSize;
	int csrSize;
};

// They have the same data structure
typedef CSR_format CSC_format;
/*
// Technically COO is not CSX but anyway :D
enum class CSX{
	CSR,
	CSC,
	CSF,
	COO,
	Dense
};
*/
struct CSR_tile_format{
	CSR_format ** csr_tiles;
	int row_size;
	int col_size;
	// Number of nnz tiles
	int nnz;
} ;

class Matrix{
	public:
		// Load the matrix
		Matrix(char * file_name, Parameters * params, Stats * stats, LLB_Mem * llb);
		// constructor when we have two matrices
		Matrix(char * file_name_A, char * file_name_B, Parameters * params, Stats *statsi, LLB_Mem *llb);

		// constructor when we are running SpMM
		Matrix(char * file_name_A, int dense_cols, Parameters * params, Stats *statsi, LLB_Mem *llb);

		// destructor
		~Matrix();

		int getTiledARowSize(){return a_csr_tiled->row_size;}
		int getTiledAColSize(){return a_csr_tiled->col_size;}
		int getTiledORowSize(){return o_csr_tiled->row_size;}
		int getTiledOColSize(){return o_csr_tiled->col_size;}
		int getTiledBRowSize(){return b_csr_tiled->row_size;}

		// Please note that this fucntion only applies output products to the LOG datastructure
		//	and does not do anything to the actual output datastructure
		// Gets the coordinate of A and B and calculates the output partial sums
		void CSRTimesCSROnlyLogUpdated(int i, int j, int k);

		// Gets the coordinate of A and B and calculates the output partial sums
		// (adds that up to prev. psums too)
		void CSRTimesCSR(int i, int j, int k, int * cycles_comp,
				uint64_t * bytes_rd, uint64_t * bytes_wr);

		// Gets the coordinate of A and B and calculates the output partial sums
		// (adds that up to prev. psums too)
		// Please note this version updates the output log matreix as well!
		// ONLY USE IT WITH SCHEDULER_8
		void CSRTimesCSR(int i, int j, int k, int * cycles_comp, uint32_t & pproductSize);

		// CSRTimesDense function is the core computation function for SpMM
		// and higher order tensors
		// The ID of the A & B basic tiles are given to the function.
		//	It does the computation, updates the output,
		//	and output log matrices and finally
		//	reports the number of cycles it took for the multiplication
		void CSRTimesDense(int i, int j, int k, int * cycles_comp, uint32_t & pproductSize);

		// Get the CSF size of a tile
		int getCSFSize(char mat_name, int i_idx, int j_idx);
		// Get the CSR size of a tile
		int getCSRSize(char mat_name, int i_idx, int j_idx);
		// Get the COO size of a tile
		int getCOOSize(char mat_name, int i_idx, int j_idx);

		// Get the COO size of a tile range
		int accumulateCOOSize(char mat_name, int i_idx, int j_idx_start, int j_idx_end);
		int accumulateCOOSize_sparse(char mat_name, int i_idx,
				int j_idx_start, int j_idx_end, CSX inp_format);

		// Get the CSF size of a tile range
		int accumulateCSFSize(char mat_name, int i_idx, int j_idx_start, int j_idx_end);
		int accumulateCSFSize_sparse(char mat_name, int i_idx,
				int j_idx_start, int j_idx_end, CSX inp_format);

		// Get the total NNZ of a tile range
		int accumulateNNZ(char mat_name, int i_idx, int j_idx_start,
			 	int j_idx_end, CSX inp_format);
		// Get the total NNR of a tile range
		int accumulateNNR(char mat_name, int i_idx, int j_idx_start,
			 	int j_idx_end, CSX inp_format);
		// Get the total NNZ tiles of a tile range
		int accumulateNNZTiles(char mat_name, int i_idx, int j_idx_start,
			 	int j_idx_end, CSX inp_format);



		// Return the # of non-zeros in a tile
		int getNNZOfATile(char mat_name, int d1, int d2);

		// Debugging related functions
		void InputTilingDebug();
		void OutputResultDebug();
		void TableDebug();

		// Get the size of matrix A and O in different formats: csr, csf, coo(only O)
		void PrintMatrixSizes();
		// Debug functions for CSR outer dimensions matrix
		void PrintNNZTiles();

		// Debug function; prints the number of fibers with X nnz
		void PrintNNZTilesAndFiberHist();
		// Debug function; prints the number of fibers with X nnz
		void PrintTileSizeHist();

		// Delete the output and pre-tile it again. This is used when multiple simulations
		// are run back to back
		void Reset();

		/******* Output log related functions ******/
		void SetLogMatrixStart(int i_start);
		// Initialize the output log matrix
		// Gets the row and column sizes and creates an empty top level tile (LLB tile)
		void initOutputLogMatrix(int i_start, int i_end, int k_start, int k_end);
		// Delete the output log matrix. We are done with the LLB tile and
		//	now we need to deallocate everything for it
		void deleteOutputLogMatrix();
		// Evict the output log tiles. Their data is evicted to the memory
		//	to be read and merged later. However, we should keep the structure
		//	since it is going to be used after this
		void evictOutputLogMatrix();

		uint64_t partiallyEvictOutputLogMatrix(double perc_free, uint64_t max_size_o);

		// Returns the COO size of the ouput log matrix
		uint64_t getOutputLogNNZCOOSize();

		uint64_t getNNZToCOOSize(uint64_t nnz);

		// Returns the NNZ count of the ouput log matrix
		int getOutputLogNNZCount();

		/************* Until here ******************/

		// CSR Format for outer dimensions to use a real {c+} or {dc+} overhead estimate
		CSR_format * a_csr_outdims;
		CSR_format * a_csc_outdims;
		CSR_format * b_csr_outdims;
		CSR_format * b_csc_outdims;


		// outerSpace related functions
		// Multiplies a_csr by b_csr and returns the output matrix size;
		// DOES THE ACTUAL MULTIPLICATION
		uint64_t CalculateNotTiledMatrixProduct();

		// Return the #non-zero entries in the non-tiled output matrix
		uint64_t GetNotTiledOutputNNZCount();

		// Gets the A column/B row id and calculates the o_traffic and #MACCs
		// Please note that this function does not do the actual multiplication
		// And also this is for not tiled (no temporal tiling -- Joel's terminology)
		void OuterProduct(int j_idx, uint32_t &a_traffic_per_pe,
				uint32_t &b_traffic_per_pe, uint32_t & o_traffic_per_pe,
				uint32_t & macc_count_per_pe, uint32_t &a_elements);

		//Gets the [i, j , k] and A column/B row id of the specific basic tiles and calculates
		// the o_traffic and #MACCs! Please note that this function does not do the actual
		// multiplication and also this is for not tiled (no temporal tiling--Joel's terminology)
		void OuterProduct(int i_idx, int j_idx, int k_idx, int j_micro_tile_idx,
				uint64_t & a_traffic, uint64_t & b_traffic, uint64_t & o_traffic, uint64_t & macc_count);

		//Gets the [i, j , k] and A row / B column id of the specific basic tiles and calculates
		// the o_traffic and #MACCs! Please note that this function does not do the actual
		// multiplication and also this is for not tiled (no temporal tiling--Joel's terminology)
		//Each PE takes one scalar of A and multiplies to a row of B micro tile
		void OuterProduct(int i_idx, int j_idx, int k_idx, int j_micro_tile_idx,
				uint64_t & a_traffic_per_pe, uint64_t & b_traffic_per_pe, uint64_t & o_traffic_per_pe,
				uint64_t & macc_count_per_pe, int & a_elements);

        void RowWiseProduct(int i_idx, int j_idx, int k_idx, int i_micro_tile_idx,
		    uint64_t & a_traffic, uint64_t & b_traffic, uint64_t & o_traffic, uint64_t & macc_count);

		// Gets the A row/O row id and calculates A, B, & O_traffics and #MACCs
		// Please note that this function does not do the actual multiplication
		void GustavsonProduct(int j_idx, uint64_t &a_traffic,
				uint64_t &b_traffic, uint64_t & o_traffic, uint64_t & macc_count);


		// Rows in the A non-tiled input matrix
		int GetARowSize();
		// Cols in the A non-tiled input matrix
		int GetAColSize();
		// Cols in the B non-tiled input matrix
		int GetBColSize();
	private:
		// These variables say where the first row and col of	the output log matrix
		//	points to with respect to the output matrix
		int row_start_log;
		int col_start_log;

		// Is matrix B same as matrix A? (A x A or A x B)
		int aIsEqualToB;
		// Parameteres/ Statistics/ LLB classes
		Parameters * params;
		Stats * stats;
		LLB_Mem * llb;

		// A and B in COO format
		// A and B are first loaded in COO then converted to BCSR fomrat (tiled)
		COO_format * a_coo;
		COO_format * b_coo;

		// A in csc and csr; B and O in csr
		// These are used for
		CSR_format * a_csr;
		CSR_format * a_csc;
		CSR_format * b_csr;
		CSR_format * o_csr;

		// A, B, and O in Tiled CSR (BCSR) format
		// CSR is used only for the internal computation; you can get the sizes in CSR, COO, or CSF
		CSR_tile_format *a_csr_tiled;
		CSR_tile_format *b_csr_tiled;
		CSR_tile_format *o_csr_tiled;

		// This variable will help book keeping for output log in scheduler 8
		CSR_tile_format *o_csr_tiled_log;

		// We should have B in the tiled CSC format for skip table construction
		//	also A csc for OuterSpace
		CSR_tile_format *a_csc_tiled;
		CSR_tile_format *b_csc_tiled;

		// Keeps the skipCycles in a hash table
		std::unordered_map<long long int, int> skipCycles_map;

		// Pre-process the matrices and builds a skip model table
		//  so during the computation, there is just a lookup
		void ConstructSkipModelTable();
		// It takes one row of matrix and builds the skipModel table for it
		//  Used in ConstructSkipModelTable in parallel
		//void calcRowSkipModelTable(int i_idx, std::queue< std::pair<long long int,int> > & skipCycles );
		void calcRowSkipModelTable(int i_idx, std::vector< std::pair<long long int,int> > & skipCycles );

		// Calculated how many cycles will spend in N-CAM model to multiply two basic tiles
		//	Skip cycles of intersect(A(i,j), B(j,k))
		void parBidirecSkipModelCyclesCalc(int i, int j, int k, int & cycles_comp);

		// Gets the filename and a pointer to the dynamically allocated
		// COO matrix. Copies the contents of the file to the COO format
		// Used in the constructor
		void ReadMatrix(char * filename, COO_format * matrix_coo);

		// Gets either of the input COO matrices and tiles it into (B)CSR
		void TileInput(COO_format * matrix_coo, CSR_tile_format * matrix_csr_tiled,
				CSX inp_format, CSR_format * csr_outer_dims, CSR_format * csc_outer_dims);

		// Creates the output tiles in (B)CSR and zeros all of them
		//  Just the POS entries are initialized and the rest is left blank
		void PreTileOutput(CSR_tile_format * matrix_csr_tiled, int i, int k);

		// Gets a COO format input and converts it to CSR/CSC format
		void convertCOOToCSR(CSR_format * matrix_csr,
				COO_format * matrix_coo, CSX inp_format);

		// Gets the CSR tile coords [i_idx, j_idx] with helper idxs that shows where
		// the corresponding row starts.
		// COO data is sorted based on the row values, so with an early scan we avoid
		// searching the COO format for every thread (to find start_idx and end_idx)
		// Used in the tiling process
		void convertCOOToCSR(CSR_tile_format * matrix_csr_tiled, COO_format * matrix_coo,
				int i_idx, int j_idx, int start_idx, int end_idx, int tile_size, CSX inp_format);

		// In order to copy COO to tiled CSR we need to know how many nnz the block has,
		// then the tile var arrays can be dynamically allocated
		// The function finds nnzs, then dynamically allocates each variable of the tile
		// Used in the tiling process
		// It does one more thing! Creates a CSR format tiled representation to use for
		//   intersections (Will help to have accurate overhead estimate and sim. speed-up)
		void fillOutNNZCountEachTile(COO_format * matrix_coo, CSR_tile_format * matrix_csr_tiled,
				CSX inp_format, CSR_format * csr_outer_dims, CSR_format * csc_outer_dims);

		// Creates csr_outer_dims and csc_outer_dims for SpMM Dense tensor
		void CreateOuterDimCSRsForDense(int row_size, int col_size,
				CSR_format * csr_outer_dims, CSR_format * csc_outer_dims);


		// Gets the CSR information and calculates the CSF size
		int calculateCSFSize(int matrix_nnz, int matrix_row_size, int * matrix_pos);

		// Gets a CSR matrix and calculates the CSR size
		void calculateCSRSize_csr(CSR_format * matrix_csr);

		// Gets a CSR matrix and calculates the COO size
		void calculateCOOSize_csr(CSR_format * matrix_csr);

		// Gets a CSR matrix and calculates the CSF size
		void calculateCSFSize_csr(CSR_format * matrix_csr);
		void skipModelCyclesCalc(int i, int j, int k, int * cycles_comp);
		int handleParallelIntersect(CSR_format *a_csr, CSR_format *b_csc,
				int i_idx, int j_idx);
		int skipModelPartialFiber(CSR_format *a_csr, CSR_format *b_csc, int i_idx, int j_idx,
				int start_a, int end_a, int start_b, int end_b, int &effectual_MACCs);
		int skipModelFiber(CSR_format *a_csr, CSR_format *b_csr, int i_idx, int j_idx);
		int nextPos(CSR_format *a_csr, int pos_start, int pos_end, int coord, int *start_pos);
		int nextPos(CSR_format *a_csr, int i_idx, int coord, int *start_pos);

		// Gets Rows, Cols, and Data of the COO format and sorts them first based on rows,
		// and for equal rows based on cols.
		// Used in ReadMatrix
		template<class A, class B, class C> void QuickSort2Desc(A a[], B b[], C c[], int l, int r);

		// Gets a COO format and returns a CSR format
		// Used in the titling process
		template <class I, class T> void coo_tocsr(const I n_row, const I n_col, const I nnz,
				const I Ai[], const I Aj[], const T Ax[], I Bp[], I Bj[], T Bx[]);
		template<class I, class T> void coo_tocsc(const I n_row, const I n_col, const I nnz,
				const I Ai[], const I Aj[], const T Ax[], I Bp[], I Bi[], T Bx[]);
		template <class I, class T> void coo_todense(const I n_row, const I n_col, const I nnz,
				const I Ai[], const I Aj[], const T Ax[], T Bx[], int fortran);


		template <class I> void coo_tocsr_nodata(const I n_row, const I n_col, const I nnz,
				const I Ai[], const I Aj[], I Bp[], I Bj[]);
		template<class I> void coo_tocsc_nodata(const I n_row, const I n_col, const I nnz,
				const I Ai[], const I Aj[], I Bp[], I Bi[]);

		void print_csr_sparse_d(const sparse_matrix_t csrA);

};

#endif
