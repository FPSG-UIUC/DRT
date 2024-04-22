#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"

Matrix::Matrix(char * file_name, Parameters * params, Stats * stats, LLB_Mem * llb){

	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Matrix A load then tiling
 	a_coo = new COO_format;
	ReadMatrix(file_name, this->a_coo);

	if((params->getIntersectModel() == intersect::idealModel)
			| (params->getIntersectModel() == intersect::parBidirecSkipModel)){
		std::fill(a_coo->vals, a_coo->vals+a_coo->nnz, 1.0);
	}
	a_csr_tiled = new CSR_tile_format; a_csc_tiled = new CSR_tile_format;
	a_csr_outdims = new CSR_format; a_csc_outdims = new CSR_format;
	TileInput(this->a_coo, this->a_csr_tiled, CSX::CSR,
			a_csr_outdims, a_csc_outdims);

	// Point to Matrix A coo and tiled CSR format data structures
	this->b_coo = this->a_coo;
	this->b_csr_tiled = this->a_csr_tiled;
	//this->b_csr_outdims = this->a_csr_outdims;
	aIsEqualToB = 1;

	// B CSC tiling for
	b_csc_tiled = new CSR_tile_format;
	b_csc_outdims = new CSR_format; b_csr_outdims = new CSR_format;
	TileInput(this->a_coo, this->b_csc_tiled, CSX::CSC,
			b_csr_outdims, b_csc_outdims);
	a_csc_tiled = b_csc_tiled;

	// Matrix O tiling
	o_csr_tiled = new CSR_tile_format; o_csr_tiled_log = new CSR_tile_format;
	PreTileOutput(this->o_csr_tiled, this->a_csr_tiled->row_size, this->a_csr_tiled->col_size);

	// Create the pre-built skip intersect computation for skipModel (others don't need it)
	if((params->getIntersectModel() == intersect::skipModel)
			| (params->getIntersectModel() == intersect::parBidirecSkipModel)
			| (params->getIntersectModel() == intersect::parUnidirecSkipModel)){
		printf("Start Builing the Look-up Table!\n"); fflush(stdout);
		ConstructSkipModelTable();
		printf("Done!\n"); fflush(stdout);
	}

	// If the experiment is related to the outerspace evaluations create non-tiled
	//	a_csr, a_csc, and b_csr
	if((params->IsOuterSpaceSim()) | (params->IsMatRaptorSim())){
		// construct a_csr
		a_csr = new CSR_format;
		convertCOOToCSR(this->a_csr, this->a_coo, CSX::CSR);
		// construct a_csc
		a_csc = new CSR_format;
		convertCOOToCSR(this->a_csc, this->a_coo, CSX::CSC);
		// b_csr is the same as a_csr
		b_csr = a_csr;
		o_csr = new CSR_format;
	}

	return;
}

Matrix::Matrix(char * file_name_A, char * file_name_B, Parameters * params, Stats * stats, LLB_Mem * llb){

	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Matrix A load then tiling
 	a_coo = new COO_format;
	ReadMatrix(file_name_A, this->a_coo);
	if((params->getIntersectModel() == intersect::idealModel)
			| (params->getIntersectModel() == intersect::parBidirecSkipModel)){
		std::fill(a_coo->vals, a_coo->vals+a_coo->nnz, 1.0);
	}

	a_csr_tiled = new CSR_tile_format; a_csc_tiled = new CSR_tile_format;
	a_csr_outdims = new CSR_format;	a_csc_outdims = new CSR_format;
	TileInput(this->a_coo, this->a_csr_tiled, CSX::CSR,
			a_csr_outdims, a_csc_outdims);
	TileInput(this->a_coo, this->a_csc_tiled, CSX::CSC,
			NULL, NULL);
    printf("Finished pretiling input A\n"); fflush(stdout);

	// Matrix B load then tiling
 	b_coo = new COO_format;
	ReadMatrix(file_name_B, this->b_coo);

	if((params->getIntersectModel() == intersect::idealModel)
			| (params->getIntersectModel() == intersect::parBidirecSkipModel)){

		std::fill(b_coo->vals, b_coo->vals+b_coo->nnz, 1.0);
	}
    printf("Finished reading the matrix\n"); fflush(stdout);

	aIsEqualToB = 0;
	b_csr_tiled = new CSR_tile_format; b_csc_tiled = new CSR_tile_format;
	TileInput(this->b_coo, this->b_csr_tiled, CSX::CSR,
			NULL, NULL);
	b_csc_outdims = new CSR_format; b_csr_outdims = new CSR_format;
	TileInput(this->b_coo, this->b_csc_tiled, CSX::CSC,
			b_csr_outdims, b_csc_outdims);

    printf("Finished pretiling input B\n"); fflush(stdout);

	// Matrix O tiling
	o_csr_tiled = new CSR_tile_format; o_csr_tiled_log = new CSR_tile_format;
	PreTileOutput(this->o_csr_tiled, this->a_csr_tiled->row_size, this->b_csr_tiled->col_size);

	// Create the pre-built skip intersect computation for skipModel (others don't need it)
	if((params->getIntersectModel() == intersect::skipModel)
			| (params->getIntersectModel() == intersect::parBidirecSkipModel)
			| (params->getIntersectModel() == intersect::parUnidirecSkipModel)){
		printf("Start Builing the Look-up Table!\n"); fflush(stdout);
		ConstructSkipModelTable();
		printf("Done!\n"); fflush(stdout);
	}

	// If the experiment is related to the outerspace evaluations create non-tiled
	//	a_csr, a_csc, and b_csr
	if((params->IsOuterSpaceSim()) | (params->IsMatRaptorSim())){
		// construct a_csr
		a_csr = new CSR_format;
		convertCOOToCSR(this->a_csr, this->a_coo, CSX::CSR);
		// construct a_csc
		a_csc = new CSR_format;
		convertCOOToCSR(this->a_csc, this->a_coo, CSX::CSC);
		// construct b_csr
		b_csr = new CSR_format;
		convertCOOToCSR(this->b_csr, this->b_coo, CSX::CSR);
		o_csr = new CSR_format;
	}

	return;
}

// Constructor for SpMM
Matrix::Matrix(char * file_name, int dense_cols, Parameters * params, Stats * stats, LLB_Mem * llb){

	this->params = params;
	this->stats = stats;
	this->llb = llb;

	// Matrix A load then tiling
 	a_coo = new COO_format;
	ReadMatrix(file_name, this->a_coo);

	a_csr_tiled = new CSR_tile_format;
	a_csr_outdims = new CSR_format; a_csc_outdims = new CSR_format;
	TileInput(this->a_coo, this->a_csr_tiled, CSX::CSR,
			a_csr_outdims, a_csc_outdims);

	// B OuterDim CSR/C tiling
	b_csc_outdims = new CSR_format; b_csr_outdims = new CSR_format;

	CreateOuterDimCSRsForDense(a_coo->col_size, dense_cols,
			b_csr_outdims, b_csc_outdims);

	return;
}

// Creates csr_outer_dims and csc_outer_dims for SpMM Dense tensor
void Matrix::CreateOuterDimCSRsForDense(int row_size, int col_size,
		CSR_format * csr_outer_dims, CSR_format * csc_outer_dims){

	int tile_size = params->getTileSize();

	// Find the row and column size of the tiled (B)CSR representation;
	//	with a high chance it needs padding
	int b_outer_rows = (int)ceil((float)row_size/tile_size);
	int b_outer_cols = (int)ceil((float)col_size/tile_size);

	// Initializing the COO representation
	int * rowIdx = new int[b_outer_rows*b_outer_cols];
	int * colIdx = new int[b_outer_rows*b_outer_cols];

	int counter=0;
	// Dynamically allocates the memory for the idx, pos, and data of each tile
	for(int i_idx = 0; i_idx < b_outer_rows; i_idx++){
		for(int j_idx = 0; j_idx < b_outer_cols; j_idx++){
			rowIdx[counter] = i_idx;
			colIdx[counter] = j_idx;
			counter++;
		}
	}

	int nnz_count = b_outer_rows*b_outer_cols;
	// csr_outerdims initialization
	csr_outer_dims->pos = new int[b_outer_rows+1];
	csr_outer_dims->idx = new int[nnz_count];
	csr_outer_dims->vals = new double[nnz_count];
	std::fill(csr_outer_dims->vals, csr_outer_dims->vals + nnz_count, 0);
	csr_outer_dims->nnz = nnz_count;
	// csc_outerdims initialization
	csc_outer_dims->pos = new int[b_outer_cols+1];
	csc_outer_dims->idx = new int[nnz_count];
	csc_outer_dims->vals = new double[nnz_count];
	std::fill(csc_outer_dims->vals, csc_outer_dims->vals + nnz_count, 0);
	csc_outer_dims->nnz = nnz_count;

	coo_tocsr_nodata(b_outer_rows, b_outer_cols, nnz_count,
			rowIdx, colIdx, csr_outer_dims->pos, csr_outer_dims->idx);

	coo_tocsc_nodata(b_outer_rows, b_outer_cols, nnz_count,
			rowIdx, colIdx, csc_outer_dims->pos, csc_outer_dims->idx);

	/*
	printf("B CSR Representation:\n");
	for(int i=0;i<b_outer_rows;i++){
		printf("Pos %d\n",csr_outer_dims->pos[i]);
		for(int j = csr_outer_dims->pos[i]; j<csr_outer_dims->pos[i+1];j++)
			printf("%d ",csr_outer_dims->idx[j]);
		printf("\n");
	}
*/
	delete[] rowIdx;
	delete[] colIdx;

	return;
}



void Matrix::ReadMatrix(char * filename, COO_format * matrix_coo){
	int ret_code;
  MM_typecode matcode;
  FILE *f;
  int M, N, nz;
  int i;

 	if (filename==NULL)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", filename);
		exit(1);
	}
  else
  {
    if ((f = fopen(filename, "r")) == NULL){
			fprintf(stderr, "Cannot open: %s \n", filename);
			exit(1);
		}
  }

  if (mm_read_banner(f, &matcode) != 0)
  {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) )
  {
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
  }

  /* find out size of sparse matrix .... */
	if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
		exit(1);

	matrix_coo->row_size = M; matrix_coo->col_size = N; matrix_coo->nnz = nz;
	// Damn it! This library does not support symmtric so I should add it manually
	if (mm_is_symmetric(matcode)){
		int row, col; double val;
		for (i=0; i<nz; i++){
			ret_code = fscanf(f, "%d %d %lg\n", &row, &col, &val);
			if (val == 0.0) {matrix_coo->nnz--;}
			else if (row != col) matrix_coo->nnz++;
		}
		if (f !=stdin) fclose(f);
		f = fopen(filename, "r");
		ret_code = mm_read_banner(f, &matcode);
		ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz);
	}
	else{
		int row, col; double val;
		for (i=0; i<nz; i++){
			ret_code = fscanf(f, "%d %d %lg\n", &row, &col, &val);
			if (val == 0.0) {matrix_coo->nnz--;}
		}
		if (f !=stdin) fclose(f);
		f = fopen(filename, "r");
		ret_code = mm_read_banner(f, &matcode);
		ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz);
	}

	/* reseve memory for matrices in COO format */
	matrix_coo->rows = new int[matrix_coo->nnz];
	matrix_coo->cols = new int[matrix_coo->nnz];
	matrix_coo->vals = new double[matrix_coo->nnz];

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	int row, col; double val;
  for (int i_idx=0, entry_idx = 0; i_idx<matrix_coo->nnz; i_idx++)
  {
		ret_code = fscanf(f, "%d %d %lg\n", &row, &col, &val);
		if (val != 0.0){
			matrix_coo->rows[entry_idx] = row; matrix_coo->cols[entry_idx] = col; matrix_coo->vals[entry_idx] = val;
			matrix_coo->rows[entry_idx]--;  /* adjust from 1-based to 0-based */
			matrix_coo->cols[entry_idx]--;
			if((mm_is_symmetric(matcode)) & (matrix_coo->rows[entry_idx] != matrix_coo->cols[entry_idx])){
				matrix_coo->cols[entry_idx+1] = matrix_coo->rows[entry_idx];
				matrix_coo->rows[entry_idx+1] = matrix_coo->cols[entry_idx];
				matrix_coo->vals[entry_idx+1] = matrix_coo->vals[entry_idx];
				entry_idx++;
				i_idx++;
			}
			entry_idx++;
		}
		else{ i_idx--;}
  }

	// Sort using quicksort; This is a customized function!
	//   Do not use it in other places without knowing what is going on inside
	QuickSort2Desc(matrix_coo->rows, matrix_coo->cols, matrix_coo->vals, 0, matrix_coo->nnz-1);

	if (f !=stdin) fclose(f);

  // mm_write_banner(stdout, matcode);
/*
	printf("Matrix (I, J, NNZ): ");
	mm_write_mtx_crd_size(stdout, matrix_coo->row_size, matrix_coo->col_size, matrix_coo->nnz);
*/
	/*
	for (i=0; i<matrix_coo->nnz; i++)
    fprintf(stdout, "%d %d %20.19g\n", matrix_coo->rows[i], matrix_coo->cols[i], matrix_coo->vals[i]);
	*/
	return;

}

Matrix::~Matrix(){
	// Delete COO format
	delete []	a_coo->rows;
	delete [] a_coo->cols;
	delete [] a_coo->vals;

	// Delete input tiled CSR format
	for(int i_idx = 0; i_idx < a_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx < a_csr_tiled->col_size; j_idx++){
			if(a_csr_tiled->csr_tiles[i_idx][j_idx].nnz){
				delete [] a_csr_tiled->csr_tiles[i_idx][j_idx].pos;
				delete [] a_csr_tiled->csr_tiles[i_idx][j_idx].idx;
				delete [] a_csr_tiled->csr_tiles[i_idx][j_idx].vals;
			}
		}
	}
	for(int i = 0; i < a_csr_tiled->row_size; i++) {
		delete [] a_csr_tiled->csr_tiles[i];
	}
	delete [] a_csr_tiled->csr_tiles;

	delete [] a_csr_outdims->pos;
	delete [] a_csr_outdims->idx;
	delete [] a_csr_outdims->vals;

	delete [] a_csc_outdims->pos;
	delete [] a_csc_outdims->idx;
	delete [] a_csc_outdims->vals;

	// Do the same things as we did for A if B is an independent variable
	if(params->getCompKernel() == kernel::SpMSpM){
		// If a and b are not the same tensors!
		if(!aIsEqualToB){
			delete []	b_coo->rows;
			delete [] b_coo->cols;
			delete [] b_coo->vals;

			for(int i_idx = 0; i_idx < b_csr_tiled->row_size; i_idx++){
				for(int j_idx = 0; j_idx < b_csr_tiled->col_size; j_idx++){
					if(b_csr_tiled->csr_tiles[i_idx][j_idx].nnz){
						delete [] b_csr_tiled->csr_tiles[i_idx][j_idx].pos;
						delete [] b_csr_tiled->csr_tiles[i_idx][j_idx].idx;
						delete [] b_csr_tiled->csr_tiles[i_idx][j_idx].vals;
					}
					if(a_csc_tiled->csr_tiles[i_idx][j_idx].nnz){
						delete [] a_csc_tiled->csr_tiles[i_idx][j_idx].pos;
						delete [] a_csc_tiled->csr_tiles[i_idx][j_idx].idx;
						delete [] a_csc_tiled->csr_tiles[i_idx][j_idx].vals;
					}
				}
			}
			for(int i = 0; i < b_csr_tiled->row_size; i++) {
				delete [] b_csr_tiled->csr_tiles[i];
			}
			for(int i = 0; i < a_csc_tiled->row_size; i++) {
				delete [] a_csc_tiled->csr_tiles[i];
			}

			delete [] b_csr_tiled->csr_tiles;
            		delete [] a_csc_tiled->csr_tiles;

		} //!aIsEqualToB

		// Delete input tiled CSC format B
		for(int i_idx = 0; i_idx < b_csc_tiled->row_size; i_idx++){
			for(int j_idx = 0; j_idx < b_csc_tiled->col_size; j_idx++){
				if(b_csc_tiled->csr_tiles[i_idx][j_idx].nnz){
					delete [] b_csc_tiled->csr_tiles[i_idx][j_idx].pos;
					delete [] b_csc_tiled->csr_tiles[i_idx][j_idx].idx;
					delete [] b_csc_tiled->csr_tiles[i_idx][j_idx].vals;
				}
			}
		}
		for(int i = 0; i < b_csc_tiled->row_size; i++) {
			delete [] b_csc_tiled->csr_tiles[i];
		}
		delete [] b_csc_tiled->csr_tiles;

		// Delete output tiled CSR format
		for(int i_idx = 0; i_idx < o_csr_tiled->row_size; i_idx++){
			for(int j_idx = 0; j_idx < o_csr_tiled->col_size; j_idx++){
				if(o_csr_tiled->csr_tiles[i_idx][j_idx].nnz){
					mkl_free(o_csr_tiled->csr_tiles[i_idx][j_idx].idx);
					mkl_free(o_csr_tiled->csr_tiles[i_idx][j_idx].vals);
					mkl_free(o_csr_tiled->csr_tiles[i_idx][j_idx].pos);
				}
			}
		}
		for(int i = 0; i < o_csr_tiled->row_size; i++) {
			delete [] o_csr_tiled->csr_tiles[i];
		}
		delete [] o_csr_tiled->csr_tiles;

		delete [] o_csr_tiled;
		delete [] o_csr_tiled_log;
		delete [] b_csr_tiled;
	} // if kernel::SpMSpM

	delete [] a_csr_tiled;
	delete [] a_csc_tiled;

	delete [] b_csr_outdims->pos;
	delete [] b_csr_outdims->idx;
	delete [] b_csr_outdims->vals;

	delete [] b_csc_outdims->pos;
	delete [] b_csc_outdims->idx;
	delete [] b_csc_outdims->vals;

	// If doing outerspace simulation then deallocate related variables as well
	if(params->IsOuterSpaceSim()){
		delete [] a_csr->vals; delete [] a_csr->pos; delete [] a_csr->idx; delete [] a_csr;
		delete [] a_csc->vals; delete [] a_csc->pos; delete [] a_csc->idx; delete [] a_csc;
		if(!aIsEqualToB){
			delete [] b_csr->vals; delete [] b_csr->pos; delete [] b_csr->idx; delete [] b_csr;
		}
		mkl_free(o_csr->vals); mkl_free(o_csr->pos); mkl_free(o_csr->idx); delete [] o_csr;
	}

	return;
}

void Matrix::Reset(){
	// Delete output tiled CSR format
	for(int i_idx = 0; i_idx < o_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx < o_csr_tiled->col_size; j_idx++){
			//delete [] o_csr_tiled->csr_tiles[i_idx][j_idx].pos;
			if(o_csr_tiled->nnz){
				delete [] o_csr_tiled->csr_tiles[i_idx][j_idx].idx;
				delete [] o_csr_tiled->csr_tiles[i_idx][j_idx].vals;
				mkl_free(o_csr_tiled->csr_tiles[i_idx][j_idx].pos);
			}
		}
	}
	for(int i = 0; i < o_csr_tiled->row_size; i++) {
		delete [] o_csr_tiled->csr_tiles[i];
	}
	delete [] o_csr_tiled->csr_tiles;
	// Matrix O tiling
	PreTileOutput(this->o_csr_tiled, this->a_csr_tiled->row_size, this->b_csr_tiled->col_size);

	return;
}


void Matrix::ConstructSkipModelTable(){

	//uint64_t total_intersect_cycles = 0;
	// Make #row_size queues to keep the skil model info for each coresponding row
	std::vector< std::pair<long long int,int> > * skipCycles =
		new std::vector< std::pair<long long int,int> >[a_csr_tiled->row_size];
	std::pair<long long int,int> temp;

	// Run the skip model for rows in parallel
	//  Each row will return one queue, then they will be merged in
	//  an unordered_map hash table
	#pragma omp parallel for
	for (int i_idx = 0; i_idx<a_csr_tiled->row_size; i_idx++){
		calcRowSkipModelTable(i_idx, skipCycles[i_idx]);
	}

	printf("Copying information to the main hash table!\n"); fflush(stdout);
	// Copy the contents of all queues to an unordered_map (hash)
	for (int i_idx = 0; i_idx<a_csr_tiled->row_size; i_idx++){
		for (std::vector< std::pair<long long int,int> >::iterator it = skipCycles[i_idx].begin();
				it != skipCycles[i_idx].end(); ++it){

			skipCycles_map.insert({it->first, it->second});
			//printf("(%lld,%d)\n", it->first, it->second); fflush(stdout);
			//total_intersect_cycles += it->second;
		}
		skipCycles[i_idx].clear();
	}

	//printf("Intersection cycles: %lu\n", total_intersect_cycles);
	// Deallocate/delete all the temporary local queues
	delete [] skipCycles;
	return;
}

// Calculates the skip model intersection cycles for one row of A (row of basic tiles)
void Matrix::calcRowSkipModelTable(int i_idx, std::vector< std::pair<long long int,int> > &skipCycles ){

	int cycles_comp = 0;
	int keyI = i_idx * a_csr_tiled->col_size * b_csr_tiled->col_size;
	// Go through all the corresponding elemnts to a specific a_row
	for (int j_idx = 0; j_idx < a_csr_tiled->col_size; j_idx++){
		int keyJ = j_idx * b_csr_tiled->col_size;
		for (int k_idx = 0; k_idx < b_csr_tiled->col_size; k_idx++){
			// Only find the cycles when both the tiles have nnz > 0
			// if you read the next line and scream, then hold your horses!
			//	nnz in a tile is the same for CSR and CSC representations
			if((a_csr_tiled->csr_tiles[i_idx][j_idx].nnz > 0 ) &
					(b_csr_tiled->csr_tiles[j_idx][k_idx].nnz > 0 )){
				cycles_comp = 0;
				// Find the skip model cycles for the specific tile (i_idx, j_idx)
				if(params->getIntersectModel() == intersect::skipModel){
					skipModelCyclesCalc(i_idx, j_idx, k_idx, &cycles_comp);
				}
				else if(params->getIntersectModel() == intersect::parBidirecSkipModel){
					parBidirecSkipModelCyclesCalc(i_idx, j_idx, k_idx, cycles_comp);
				}
				// Calculate the key for the hash; This is an arbitrary unique key generation
				//   It is choice does not have any specific reason other than being unique
				int key = keyI + keyJ + k_idx;
				std::pair<long long int,int> temp(key, cycles_comp);
				//std::pair<long long int,int> temp(1, 1);

				skipCycles.push_back(temp);
				//printf("(%d %d %d): %d\n",i_idx, j_idx, k_idx, cycles_comp); fflush(stdout);
			}
		}
	}
	return;
}

/* Just creates the template of the output CSR tiled matrix
 * Sets nnz in each tile az zero. This data structure will be
 * filled out during the computations
 */
void Matrix::PreTileOutput(CSR_tile_format * matrix_csr_tiled, int i, int k){

	int tile_size = params->getTileSize();
	matrix_csr_tiled->nnz = 0;
	matrix_csr_tiled->row_size = i;
	matrix_csr_tiled->col_size = k;

	// Dynamically allocate the tiles of the (B)CSR representation
	//	It is a 2D [row_size][column_size] structure
	matrix_csr_tiled->csr_tiles = new CSR_format*[matrix_csr_tiled->row_size];
	for(int i_idx=0; i_idx < matrix_csr_tiled->row_size; i_idx++){
		matrix_csr_tiled->csr_tiles[i_idx] = new CSR_format[matrix_csr_tiled->col_size];
	}

	// Intialize each tile of the output. Only POS needs to be dynamically allocated
	for(int i_idx = 0; i_idx < matrix_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx < matrix_csr_tiled->col_size; j_idx++){
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz = 0;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnr = 0;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].csrSize = 0;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].csfSize = 0;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].cooSize = 0;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].row_size = tile_size;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].col_size = tile_size;
		}
	}

	return;
}

void Matrix::SetLogMatrixStart(int i_start){
	row_start_log = i_start;
	return;
}
// Initialize the output log matrix
// Gets the row and column sizes and creates an empty top level tile (LLB tile)
void Matrix::initOutputLogMatrix(int i_start, int i_end, int k_start, int k_end){

	int tile_size = params->getTileSize();

	row_start_log = i_start;
	col_start_log = k_start;

	o_csr_tiled_log->nnz=0;
	o_csr_tiled_log->row_size = i_end - i_start;
	o_csr_tiled_log->col_size = k_end - k_start;

	// Dynamically allocate the tiles of the (B)CSR representation
	//	It is a 2D [row_size][column_size] structure
	o_csr_tiled_log->csr_tiles = new CSR_format*[o_csr_tiled_log->row_size];
	for(int i_idx=0; i_idx < o_csr_tiled_log->row_size; i_idx++){
		o_csr_tiled_log->csr_tiles[i_idx] = new CSR_format[o_csr_tiled_log->col_size];
	}

	// Intialize each tile of the output. Only POS needs to be dynamically allocated
	for(int i_idx = 0; i_idx < o_csr_tiled_log->row_size; i_idx++){
		for(int j_idx = 0; j_idx < o_csr_tiled_log->col_size; j_idx++){
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnr = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].csrSize = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].csfSize = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].cooSize = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].row_size = tile_size;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].col_size = tile_size;
		}
	}
	return;
}

// Delete the output log matrix. We are done with the LLB tile and
//	now we need to deallocate everything for it
void Matrix::deleteOutputLogMatrix(){
	o_csr_tiled_log->nnz=0;

	// First deallocate pos, idx, and val of the nnz tiles
	for(int i_idx = 0; i_idx < o_csr_tiled_log->row_size; i_idx++){
		for(int j_idx = 0; j_idx < o_csr_tiled_log->col_size; j_idx++){
			if(o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz){
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].vals;
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].idx;
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].vals);
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].idx);
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].pos);
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].pos;
			}
		}
	}
	// Now in two steps deallocate csr_tiles
	for(int i = 0; i < o_csr_tiled_log->row_size; i++) {
		delete [] o_csr_tiled_log->csr_tiles[i];
	}
	delete [] o_csr_tiled_log->csr_tiles;

	return;
}

// Evict the output log tiles. Their data is evicted to the memory
//	to be read and merged later. However, we should keep the structure
//	since it is going to be used after this
uint64_t Matrix::partiallyEvictOutputLogMatrix(double perc_free, uint64_t max_size_o){

	uint64_t free_up_size = (uint64_t)(perc_free * (double)max_size_o);

	uint64_t removed_nnz = 0;
	//o_csr_tiled_log->nnz=0;

	// First deallocate pos, idx, and val of the nnz tiles
	for(int i_idx = 0; i_idx < o_csr_tiled_log->row_size; i_idx++){
		for(int j_idx = 0; j_idx < o_csr_tiled_log->col_size; j_idx++){
			if(o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz){
				removed_nnz += o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz;
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].vals;
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].idx;
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].idx);
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].vals);
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].pos);
				o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz = 0;
				o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnr = 0;
				o_csr_tiled_log->csr_tiles[i_idx][j_idx].csrSize = 0;
				o_csr_tiled_log->csr_tiles[i_idx][j_idx].csfSize = 0;
				o_csr_tiled_log->csr_tiles[i_idx][j_idx].cooSize = 0;
			}
		}
		if(getNNZToCOOSize(removed_nnz) > free_up_size){
			o_csr_tiled_log->nnz -= removed_nnz;
			return getNNZToCOOSize(removed_nnz);
		}
	}

	o_csr_tiled_log->nnz -= removed_nnz;
	return getNNZToCOOSize(removed_nnz);
}


// Evict the output log tiles. Their data is evicted to the memory
//	to be read and merged later. However, we should keep the structure
//	since it is going to be used after this
void Matrix::evictOutputLogMatrix(){

	o_csr_tiled_log->nnz=0;

	// First deallocate pos, idx, and val of the nnz tiles
	for(int i_idx = 0; i_idx < o_csr_tiled_log->row_size; i_idx++){
		for(int j_idx = 0; j_idx < o_csr_tiled_log->col_size; j_idx++){
			if(o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz){
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].vals;
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].idx;
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].vals);
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].idx);
				mkl_free(o_csr_tiled_log->csr_tiles[i_idx][j_idx].pos);
				//delete [] o_csr_tiled_log->csr_tiles[i_idx][j_idx].pos;
			}
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnr = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].csrSize = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].csfSize = 0;
			o_csr_tiled_log->csr_tiles[i_idx][j_idx].cooSize = 0;
		}
	}

	return;
}

// Returns the COO size of a specific size of NNZ
uint64_t Matrix::getNNZToCOOSize(uint64_t nnz){

	uint64_t total_coo_size = nnz *
		(params->getDataSize() + 2*params->getIdxSize());

	return total_coo_size;
}

// Returns the COO size of the ouput log matrix
uint64_t Matrix::getOutputLogNNZCOOSize(){

	uint64_t total_coo_size = o_csr_tiled_log->nnz *
		(params->getDataSize() + 2*params->getIdxSize());

	return total_coo_size;
}

// Returns the NNZ count of the ouput log matrix
int Matrix::getOutputLogNNZCount(){

	return o_csr_tiled_log->nnz;
}

/* Gets the input (either A or B) COO representation and a pointer
 * to its tiled CSR data structure and fills out the tiles
 */
void Matrix::TileInput(COO_format * matrix_coo, CSR_tile_format * matrix_csr_tiled,
		CSX inp_format, CSR_format * csr_outer_dims, CSR_format * csc_outer_dims){

	int tile_size = params->getTileSize();

	// Find the row and column size of the tiled (B)CSR representation;
	//	with a high chance it needs padding
	matrix_csr_tiled->row_size = (int)ceil((float)matrix_coo->row_size/tile_size);
	matrix_csr_tiled->col_size = (int)ceil((float)matrix_coo->col_size/tile_size);
	matrix_csr_tiled->nnz = 0;

	// Dynamically allocate the tiles of the (B)CSR representation
	//	It is a 2D [row_size][column_size] structure
	matrix_csr_tiled->csr_tiles = new CSR_format*[matrix_csr_tiled->row_size];
	for(int i_idx=0; i_idx < matrix_csr_tiled->row_size; i_idx++){
		// Toluwa reported this as a bug so I changed it from row_size to col_size
		matrix_csr_tiled->csr_tiles[i_idx] = new CSR_format[matrix_csr_tiled->col_size];
	}

	// this functions does lots of critical tasks:
	//   1 - Dynamically allocates the memory for the idx, pos,
	//		and data of each tile
	//	 2 - Fills out the nnz, row_size, col_size, and nnz of each tile
	//   3 - Constructs the CSR representation of the outer dimension
	//		(The BCSR metadata)
	fillOutNNZCountEachTile(matrix_coo, matrix_csr_tiled, inp_format,
			csr_outer_dims, csc_outer_dims);

	//printf("NNZ tiles: %d out of %d\n",matrix_csr_tiled->nnz,
	//		matrix_csr_tiled->row_size*matrix_csr_tiled->col_size);

	// The next 15 lines try to find the start and end addresses
	//  of the COO representation (remember it is sorted) for
	//  the corresponding row of the (B)CSR representation.
	//  So, the start and end shows the addresses for a specific row
	//  of the (B)CSR rep. . Now, we can reduce the search space for
	//  each row and parallelize them efficiently
	int * start_idx = new int[matrix_csr_tiled->row_size];
	int * end_idx = new int[matrix_csr_tiled->row_size];
	// Zero out all the indexes first
	std::fill(start_idx, start_idx + matrix_csr_tiled->row_size, 0);
	std::fill(end_idx, end_idx + matrix_csr_tiled->row_size, 0);

	// Comp: keeps track of the row address in coordinate space
	//	comp+= tile_size; every time we move to the next row
	// itr : keeps track of the row id
	//  itr++; every time we move to the next row
    int itr = 0;
	for(int i=0, comp = 0; i<matrix_coo->nnz; i++){
		if(matrix_coo->rows[i]>= comp){
			// There are matrices like cop20k that have empty rows
			//	"While" makes sure that it does not get messed up
			while(matrix_coo->rows[i]>=comp){
				comp += tile_size;
				start_idx[itr] = i;
				if(itr != 0)
					end_idx[itr-1] = start_idx[itr];
				itr++;
			}
		}
	}

    //Last element of row pointer/segment array is always set to NNZ 
    //Toluwa - I found a bug. We need to handle the case where the last tiled row is empty
    //old code with bug: end_idx[matrix_csr_tiled->row_size-1] = matrix_coo->nnz 
    //new code:
    if (itr == matrix_csr_tiled->row_size) {
        end_idx[matrix_csr_tiled->row_size-1] = matrix_coo->nnz;
    } else { //we had an empty last few rows!
        int num_empty_rows = matrix_csr_tiled->row_size - itr;

        //Set the end index of the last non-zero tile 
        end_idx[itr-1]  = matrix_coo->nnz;

        for (int c=0; c < num_empty_rows; c++) {
            //Toluwa - my changes to fix the bug:
            start_idx[(matrix_csr_tiled->row_size-1)-c]    = matrix_coo->nnz;
            end_idx[(matrix_csr_tiled->row_size-1)-c]      = matrix_coo->nnz; 
        }
    }


	// Debugging codes to manually check if start, end are correct
	/*
	for(int i =0; i< matrix_csr_tiled->row_size; i++){
		printf("%d : (%d, %d)  - ",i ,start_idx[i],end_idx[i]);
	}
	printf("\n");
	*/

	// convert the data in COO format to CSR tiles;
	//   for each tile we pass i_idx and j_idx, which are coordinates
	//   The conversion happens in parallel (major speed-up)
	//   start_idx and end_idx help to have smaller search space so it
	//   reduces wasted computation
	//omp_set_num_threads(params->getNumThreads());
	#pragma omp parallel for
	for(int i_idx = 0; i_idx < matrix_csr_tiled->row_size ; i_idx++){
		#pragma omp parallel for
		for(int j_idx = 0; j_idx < matrix_csr_tiled->col_size; j_idx++){
			convertCOOToCSR(matrix_csr_tiled, matrix_coo, i_idx, j_idx,
					start_idx[i_idx], end_idx[i_idx], tile_size, inp_format);
		}
	}

	// Deallocate/delete local temporary indexes
	delete [] start_idx;
	delete [] end_idx;

	return;
}

// Gets the CSR tile coords [i_idx, j_idx] with helper idxs that shows where
// the corresponding row starts.
// COO data is sorted based on the row values, so with an early scan we avoid
// searching the COO format for every thread (to find start_idx and end_idx)
// Used in the tiling process
void Matrix::convertCOOToCSR(CSR_tile_format * matrix_csr_tiled, COO_format * matrix_coo,
		int i_idx, int j_idx, int start_idx, int end_idx ,int tile_size, CSX inp_format){

	if(matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz == 0){
		// These will update the tile size to zero
		calculateCSFSize_csr(&matrix_csr_tiled->csr_tiles[i_idx][j_idx]);
		calculateCSRSize_csr(&matrix_csr_tiled->csr_tiles[i_idx][j_idx]);
		calculateCOOSize_csr(&matrix_csr_tiled->csr_tiles[i_idx][j_idx]);

		return;
	}

	// Calculate the starting and ending rows and columns of the tile
	int row_start = i_idx*tile_size;
	int row_end   = (i_idx+1)*tile_size - 1;
	int col_start = j_idx*tile_size;
	int col_end   = (j_idx+1)*tile_size - 1;

	COO_format temp_coo;
	temp_coo.row_size = tile_size;
	temp_coo.col_size = tile_size;
	temp_coo.nnz = matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz;
	// Create a local COO datastructure to covert them to CSR/CSC
	temp_coo.vals = new double[temp_coo.nnz];
	temp_coo.rows = new int[temp_coo.nnz];
	temp_coo.cols = new int[temp_coo.nnz];

	int temp = 0;
	// Searches through start and end idxs to find the
	//	entries that belong to the specific tile we are trying to build
	//	They are saved in a new COO DS and passed to SCIPY's conversion
	//	function to get CSR/CSC representation
	for(int i =start_idx, itr = 0; i< end_idx; i++){
		if((matrix_coo->rows[i]>=row_start) & (matrix_coo->rows[i]<=row_end)
			& (matrix_coo->cols[i]>=col_start) & (matrix_coo->cols[i]<=col_end)){

			// Change the genetal coordinate to relative tile coordinate
			temp_coo.rows[itr] = matrix_coo->rows[i]%tile_size;
			temp_coo.cols[itr] = matrix_coo->cols[i]%tile_size;
			temp_coo.vals[itr] = matrix_coo->vals[i];
			//printf("(%d, %d) ",temp_coo.rows[itr],temp_coo.cols[itr]);
			itr++;
			temp++;
		}
	}
	// This is a debugging if. It happens for cop20k style matrices
	//		that have empty rows
	if(temp_coo.nnz != temp){
		printf("** (%d,%d): %d - %d\n",i_idx, j_idx ,temp_coo.nnz, temp);
		printf(" %d %d \n",start_idx, end_idx);
		fflush(stdout);
		exit(1);
	}
	// Get the CSR/SCS tile of the COO input
	if (inp_format == CSX::CSR){
		coo_tocsr(temp_coo.row_size, temp_coo.col_size, temp_coo.nnz,
			temp_coo.rows, temp_coo.cols, temp_coo.vals,
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].pos, matrix_csr_tiled->csr_tiles[i_idx][j_idx].idx,
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].vals);
	}
	else if(inp_format == CSX::CSC){
			coo_tocsc(temp_coo.row_size, temp_coo.col_size, temp_coo.nnz,
			temp_coo.rows, temp_coo.cols, temp_coo.vals,
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].pos, matrix_csr_tiled->csr_tiles[i_idx][j_idx].idx,
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].vals);
	}
	else{
		printf("Only CSR and CSC are supported");
		exit(1);
	}

	// Calculate the tile size for different formats
	//	for Inputs this reduces the computation in later steps
	//	although that's super trivial for COO and CSR.
	calculateCSFSize_csr(&matrix_csr_tiled->csr_tiles[i_idx][j_idx]);
	calculateCSRSize_csr(&matrix_csr_tiled->csr_tiles[i_idx][j_idx]);
	calculateCOOSize_csr(&matrix_csr_tiled->csr_tiles[i_idx][j_idx]);

	// Deallocate/delete the local temporary COO DS
	delete [] temp_coo.vals;
	delete [] temp_coo.cols;
	delete [] temp_coo.rows;

	return;
}

/* Finds number of NNZ for each CSR tile and preserves memory in each tile for data
 	this functions does lots of critical tasks:
	  1 - Dynamically allocates the memory for the idx, pos,
			and data of each tile
		2 - Fills out the nnz, row_size, col_size, and nnz of each tile

	This function is used in the input (B)CSR tiling routine
 */

void Matrix::fillOutNNZCountEachTile(COO_format * matrix_coo, CSR_tile_format * matrix_csr_tiled,
		CSX inp_format, CSR_format * csr_outer_dims, CSR_format * csc_outer_dims){
	int row, col;
	int tile_size = params->getTileSize();

	// Fill out the nnz, row_size, col_size, and nnz of each tile
	for(int i_idx = 0; i_idx < matrix_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx < matrix_csr_tiled->col_size; j_idx++){
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz = 0;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].row_size = tile_size;
			matrix_csr_tiled->csr_tiles[i_idx][j_idx].col_size = tile_size;
		}
	}
	// Go through all the nnz of the coo and find out how many nnz each tile has
	for(int i_idx =0; i_idx<matrix_coo->nnz; i_idx++){
		row = matrix_coo->rows[i_idx] / tile_size;
		col = matrix_coo->cols[i_idx] / tile_size;
		// If a tile has more than 0 non-zero elemnts, then increment
		//   the total number of nnz tiles
		if(matrix_csr_tiled->csr_tiles[row][col].nnz == 0){
			matrix_csr_tiled->nnz += 1;
		}
		matrix_csr_tiled->csr_tiles[row][col].nnz += 1;
	}

	// Initializing the COO representation
	int * rowIdx = new int[matrix_csr_tiled->nnz];
	int * colIdx = new int[matrix_csr_tiled->nnz];
	int counter = 0;

	// Dynamically allocates the memory for the idx, pos, and data of each tile
	for(int i_idx = 0; i_idx < matrix_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx < matrix_csr_tiled->col_size; j_idx++){
			// Allocate if and only if there are non-zero elemnts in that tile
			if(matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz){
				int dyn_allocate = (inp_format==CSX::CSR)?
					(matrix_csr_tiled->csr_tiles[i_idx][j_idx].row_size+1):
					(matrix_csr_tiled->csr_tiles[i_idx][j_idx].col_size+1);
				matrix_csr_tiled->csr_tiles[i_idx][j_idx].pos = new int[dyn_allocate];
				matrix_csr_tiled->csr_tiles[i_idx][j_idx].idx =
					new int[matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz];
				matrix_csr_tiled->csr_tiles[i_idx][j_idx].vals =
					new double[matrix_csr_tiled->csr_tiles[i_idx][j_idx].nnz];


				rowIdx[counter] = i_idx;
				colIdx[counter] = j_idx;
				counter++;

			}
		}
	}

	if((csr_outer_dims !=NULL) & (csc_outer_dims != NULL)){
		// csr_outerdims initialization
		csr_outer_dims->pos = new int[matrix_csr_tiled->row_size+1];
		csr_outer_dims->idx = new int[matrix_csr_tiled->nnz];
		csr_outer_dims->vals = new double[matrix_csr_tiled->nnz];
		std::fill(csr_outer_dims->vals, csr_outer_dims->vals + matrix_csr_tiled->nnz, 0);
		csr_outer_dims->nnz = matrix_csr_tiled->nnz;
		// csc_outerdims initialization
		csc_outer_dims->pos = new int[matrix_csr_tiled->col_size+1];
		csc_outer_dims->idx = new int[matrix_csr_tiled->nnz];
		csc_outer_dims->vals = new double[matrix_csr_tiled->nnz];
		std::fill(csc_outer_dims->vals, csc_outer_dims->vals + matrix_csr_tiled->nnz, 0);
		csc_outer_dims->nnz = matrix_csr_tiled->nnz;

		coo_tocsr_nodata(matrix_csr_tiled->row_size, matrix_csr_tiled->col_size,
				matrix_csr_tiled->nnz, rowIdx, colIdx, csr_outer_dims->pos, csr_outer_dims->idx);

		coo_tocsc_nodata(matrix_csr_tiled->row_size, matrix_csr_tiled->col_size,
				matrix_csr_tiled->nnz, rowIdx, colIdx, csc_outer_dims->pos, csc_outer_dims->idx);

	}

	delete[] rowIdx;
	delete[] colIdx;

	return;
}


/* Sorts the COO format before converting to CSR
 * Since we need a sorted COO to convert it easily to CSR
 * */
template<class A, class B, class C> void Matrix::QuickSort2Desc(A a[], B b[], C c[], int l, int r)
{
	int i = l;
	int j = r;
	A v = a[(l + r) / 2];
	A w = b[(l + r) / 2];
	do {
		while(1){
			if(a[i] < v)i++;
			else if((a[i] == v) & (b[i] < w))i++;
			else break;
		}
		while(1){
			if(a[j] > v)j--;
			else if((a[j] == v) & (b[j] > w))j--;
			else break;
		}

		if (i <= j)
		{
			std::swap(a[i], a[j]);
			std::swap(b[i], b[j]);
			std::swap(c[i], c[j]);
			i++;
			j--;
		};
	} while (i <= j);
	if (l < j)QuickSort2Desc(a, b, c, l, j);
	if (i < r)QuickSort2Desc(a, b, c, i, r);
}

// Return the # of non-zeros in a tile
int Matrix::getNNZOfATile(char mat_name, int d1, int d2){
	switch (mat_name){
		case 'A': return a_csr_tiled->csr_tiles[d1][d2].nnz;
		case 'B': return b_csr_tiled->csr_tiles[d1][d2].nnz;
		case 'O': return o_csr_tiled->csr_tiles[d1][d2].nnz;
		default: printf("It is hard to say what you are looking for!\n"); exit(1);
	}
	return 0;
}



// Return the COO size of a CSR tile
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::getCOOSize(char mat_name, int i_idx, int j_idx){
	switch (mat_name){
		case 'A': return a_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
		case 'B': return b_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
		case 'O': return o_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	return 0;
}

// Return the CSR size of a CSR tile
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::getCSRSize(char mat_name, int i_idx, int j_idx){
	switch (mat_name){
		case 'A': return a_csr_tiled->csr_tiles[i_idx][j_idx].csrSize;
		case 'B': return b_csr_tiled->csr_tiles[i_idx][j_idx].csrSize;
		case 'O': return o_csr_tiled->csr_tiles[i_idx][j_idx].csrSize;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	return 0;
}

// Return the CSF size of a CSR tile
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::getCSFSize(char mat_name, int i_idx, int j_idx){
	switch (mat_name){
		case 'A': return a_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
		case 'B': return b_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
		case 'O': return o_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	return 0;
}

// Return the CSF size of range of tiles
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateCSFSize(char mat_name, int i_idx, int j_idx_start, int j_idx_end){
	int size= 0 ;
	switch (mat_name){
		case 'A':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += a_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
				return size;
		case 'B':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += b_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
				return size;
		case 'O':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += o_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
				return size;
		// output log file
		case 'L':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += o_csr_tiled_log->csr_tiles[i_idx][j_idx].csfSize;
				return size;

		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	return 0;
}

// Return the COO size of range of tiles - The outerdim is CSR
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateCOOSize_sparse(char mat_name, int i_idx,
		int j_idx_start, int j_idx_end, CSX inp_format){
	int size= 0 ;
	if (mat_name=='A'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * a_outdims = (inp_format == CSX::CSR) ?
				a_csr_outdims : a_csc_outdims;
			// Iterate over the column idxs of a specific row (e.g., i_idx)
			for(int t_idx = a_outdims->pos[i_idx]; t_idx < a_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = a_outdims->idx[t_idx];
				// If the col idx is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx matches the range, then add the size
				else if(j_idx<j_idx_end){
					size += (inp_format == CSX::CSR) ?
						a_csr_tiled->csr_tiles[i_idx][j_idx].cooSize:
						a_csr_tiled->csr_tiles[j_idx][i_idx].cooSize;
				}
				// If the col idx is larger than the max, get out. You are done soldier Svejk
				else
					return size;
			}
			return size;
	}
	else if(mat_name == 'B'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * b_outdims = (inp_format == CSX::CSR) ?
				b_csr_outdims : b_csc_outdims;
			for(int t_idx = b_outdims->pos[i_idx]; t_idx < b_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = b_outdims->idx[t_idx];
				// If the col idx(CSR)/ row idx(CSC) is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx(CSR)/ row idx(CSC) matches the range, then add the size
				else if(j_idx<j_idx_end){
					size += (inp_format == CSX::CSR) ?
						b_csr_tiled->csr_tiles[i_idx][j_idx].cooSize:
						b_csr_tiled->csr_tiles[j_idx][i_idx].cooSize;
				}
				// If the col idx(CSR)/ row idx(CSC) is larger than the max, get out.
				//  You are done soldier Svejk
				else
					return size;
			}
			return size;
	}
	else if(mat_name == 'O'){
			for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
				size += o_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
			return size;
	}
	else { printf("Unknown variable is requested!\n"); exit(1);}

	return 0;
}



// Return the CSF size of range of tiles - The outerdim is CSR
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateCSFSize_sparse(char mat_name, int i_idx,
		int j_idx_start, int j_idx_end, CSX inp_format){
	int size= 0 ;
	if (mat_name=='A'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * a_outdims = (inp_format == CSX::CSR) ?
				a_csr_outdims : a_csc_outdims;
			// Iterate over the column idxs of a specific row (e.g., i_idx)
			for(int t_idx = a_outdims->pos[i_idx]; t_idx < a_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = a_outdims->idx[t_idx];
				// If the col idx is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx matches the range, then add the size
				else if(j_idx<j_idx_end){
					size += (inp_format == CSX::CSR) ?
						a_csr_tiled->csr_tiles[i_idx][j_idx].csfSize:
						a_csr_tiled->csr_tiles[j_idx][i_idx].csfSize;
				}
				// If the col idx is larger than the max, get out. You are done soldier Svejk
				else
					return size;
			}
			return size;
	}
	else if(mat_name == 'B'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * b_outdims = (inp_format == CSX::CSR) ?
				b_csr_outdims : b_csc_outdims;
			for(int t_idx = b_outdims->pos[i_idx]; t_idx < b_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = b_outdims->idx[t_idx];
				// If the col idx(CSR)/ row idx(CSC) is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx(CSR)/ row idx(CSC) matches the range, then add the size
				else if(j_idx<j_idx_end){
					size += (inp_format == CSX::CSR) ?
						b_csr_tiled->csr_tiles[i_idx][j_idx].csfSize:
						b_csr_tiled->csr_tiles[j_idx][i_idx].csfSize;
				}
				// If the col idx(CSR)/ row idx(CSC) is larger than the max, get out.
				//  You are done soldier Svejk
				else
					return size;
			}
			return size;
	}
	else if(mat_name == 'O'){
			// Sorry, we cannot do it for output! -> I have some ideas to do it! TODO
			//   Calculate the output and create the table beforehand
			for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
				size += o_csr_tiled->csr_tiles[i_idx][j_idx].csfSize;
			return size;
	}
	else { printf("Unknown variable is requested!\n"); exit(1);}

	return 0;
}

// Return the total number of NNZ of range of tiles - The outerdim is CSR
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateNNZ(char mat_name, int i_idx,
		int j_idx_start, int j_idx_end, CSX inp_format){
	int nnz_count= 0 ;
	if (mat_name=='A'){
			CSR_format * a_outdims = (inp_format == CSX::CSR) ?
				a_csr_outdims : a_csc_outdims;
			// Iterate over the column idxs of a specific row (e.g., i_idx)
			for(int t_idx = a_outdims->pos[i_idx]; t_idx < a_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = a_outdims->idx[t_idx];
				// If the col idx is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx matches the range, then add the size
				else if(j_idx<j_idx_end){
					nnz_count += (inp_format == CSX::CSR) ?
						a_csr_tiled->csr_tiles[i_idx][j_idx].nnz:
						a_csr_tiled->csr_tiles[j_idx][i_idx].nnz;
				}
				// If the col idx is larger than the max, get out. You are done soldier Svejk
				else
					return nnz_count;
			}
			return nnz_count;
	}
	else if(mat_name == 'B'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * b_outdims = (inp_format == CSX::CSR) ?
				b_csr_outdims : b_csc_outdims;
			for(int t_idx = b_outdims->pos[i_idx]; t_idx < b_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = b_outdims->idx[t_idx];
				// If the col idx(CSR)/ row idx(CSC) is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx(CSR)/ row idx(CSC) matches the range, then add the nnz
				else if(j_idx<j_idx_end)
					nnz_count += (inp_format == CSX::CSR) ?
						b_csr_tiled->csr_tiles[i_idx][j_idx].nnz:
						b_csr_tiled->csr_tiles[j_idx][i_idx].nnz;
				// If the col idx(CSR)/ row idx(CSC) is larger than the max, get out.
				//  You are done soldier Svejk
				else
					return nnz_count;
			}
			return nnz_count;
	}
	else if(mat_name == 'O'){
			// Sorry, we cannot do it for output! -> I have some ideas to do it! TODO
			//   Calculate the output and create the table beforehand
			for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
				nnz_count += o_csr_tiled->csr_tiles[i_idx][j_idx].nnz;
			return nnz_count;
	}
	else if(mat_name == 'L'){
			for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
				nnz_count += o_csr_tiled_log->csr_tiles[i_idx][j_idx].nnz;
			return nnz_count;
	}

	else { printf("Unknown variable is requested!\n"); exit(1);}

	return 0;
}


// Return the total number of NNR of range of tiles - The outerdim is CSR
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateNNR(char mat_name, int i_idx,
		int j_idx_start, int j_idx_end, CSX inp_format){
	int nnr_count= 0 ;
	if (mat_name=='A'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * a_outdims = (inp_format == CSX::CSR) ?
				a_csr_outdims : a_csc_outdims;
			// Iterate over the column idxs of a specific row (e.g., i_idx)
			for(int t_idx = a_outdims->pos[i_idx]; t_idx < a_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = a_outdims->idx[t_idx];
				// If the col idx is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx matches the range, then add the size
				else if(j_idx<j_idx_end){
					nnr_count += (inp_format == CSX::CSR) ?
						a_csr_tiled->csr_tiles[i_idx][j_idx].nnr:
						a_csr_tiled->csr_tiles[j_idx][i_idx].nnr;
				}
				// If the col idx is larger than the max, get out. You are done soldier Svejk
				else
					return nnr_count;
			}
			return nnr_count;
	}
	else if(mat_name == 'B'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * b_outdims = (inp_format == CSX::CSR) ?
				b_csr_outdims : b_csc_outdims;
			for(int t_idx = b_outdims->pos[i_idx]; t_idx < b_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = b_outdims->idx[t_idx];
				// If the col idx(CSR)/ row idx(CSC) is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx(CSR)/ row idx(CSC) matches the range, then add the nnz
				else if(j_idx<j_idx_end){
					nnr_count += (inp_format == CSX::CSR) ?
						b_csr_tiled->csr_tiles[i_idx][j_idx].nnr:
						b_csr_tiled->csr_tiles[j_idx][i_idx].nnr;
				}
				// If the col idx(CSR)/ row idx(CSC) is larger than the max, get out.
				//  You are done soldier Svejk
				else
					return nnr_count;
			}
			return nnr_count;
	}
	else if(mat_name == 'O'){
			// Sorry, we cannot do it for output! -> I have some ideas to do it! TODO
			//   Calculate the output and create the table beforehand
			for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
				nnr_count += o_csr_tiled->csr_tiles[i_idx][j_idx].nnr;
			return nnr_count;
	}
	else if(mat_name == 'L'){
			for(int j_idx = j_idx_start; j_idx<j_idx_end;j_idx++)
				nnr_count += o_csr_tiled->csr_tiles[i_idx][j_idx].nnr;
			return nnr_count;
	}

	else { printf("Unknown variable is requested!\n"); exit(1);}

	return 0;
}


// Return the total number of NNZ tiles of range of tiles - The outerdim is CSR
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateNNZTiles(char mat_name, int i_idx,
		int j_idx_start, int j_idx_end, CSX inp_format){
	int nnz_tile_count= 0 ;
	if (mat_name=='A'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * a_outdims = (inp_format == CSX::CSR) ?
				a_csr_outdims : a_csc_outdims;
			// Iterate over the column idxs of a specific row (e.g., i_idx)
			for(int t_idx = a_outdims->pos[i_idx]; t_idx < a_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = a_outdims->idx[t_idx];
				// If the col idx is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx matches the range, then add the size
				else if(j_idx<j_idx_end)
					nnz_tile_count++;
				// If the col idx is larger than the max, get out. You are done soldier Svejk
				else
					return nnz_tile_count;
			}
			return nnz_tile_count;
	}
	else if(mat_name == 'B'){
			// iterate over the column idxs in CSR (or row idx in CSC)
			//   of a specific row/col (e.g., i_idx)
			CSR_format * b_outdims = (inp_format == CSX::CSR) ?
				b_csr_outdims : b_csc_outdims;
			for(int t_idx = b_outdims->pos[i_idx]; t_idx < b_outdims->pos[i_idx+1]; t_idx++){
				int j_idx = b_outdims->idx[t_idx];
				// If the col idx(CSR)/ row idx(CSC) is smaller than start skip
				if(j_idx<j_idx_start) continue;
				// If the col idx(CSR)/ row idx(CSC) matches the range, then add the nnz
				else if(j_idx<j_idx_end)
					nnz_tile_count++;
				// If the col idx(CSR)/ row idx(CSC) is larger than the max, get out.
				//  You are done soldier Svejk
				else
					return nnz_tile_count;
			}
			return nnz_tile_count;
	}
	else if(mat_name == 'O'){
			for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++){
				if(o_csr_tiled->csr_tiles[i_idx][j_idx].nnz)
					nnz_tile_count++;
			}
			return nnz_tile_count;
	}
	else { printf("Unknown variable is requested!\n"); exit(1);}

	return 0;
}



// Return the COO size of range of tiles
// For A and B they are calculated in the initialization phase
// and for O it is calculated after each multiply and accumulate
int Matrix::accumulateCOOSize(char mat_name, int i_idx, int j_idx_start, int j_idx_end){
	int size= 0 ;
	switch (mat_name){
		case 'A':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += a_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
				return size;
		case 'B':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += b_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
				return size;
		case 'O':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += o_csr_tiled->csr_tiles[i_idx][j_idx].cooSize;
				return size;
		case 'L':
				for(int j_idx = j_idx_start; j_idx<j_idx_end; j_idx++)
					size += o_csr_tiled_log->csr_tiles[i_idx][j_idx].cooSize;
				return size;

		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	return 0;
}


// Gets the CSR information and calculates the CSF size
int Matrix::calculateCSFSize(int matrix_nnz, int matrix_row_size, int * matrix_pos){
	// CSF size calculation
	int csfSize = 0;
	// If it has no nnz, size is zero
	if(matrix_nnz){
		// nnr: number of non-zero rows
		int nnr = 0;
		// Find the number of nnr
		for (int i = 1; i<=matrix_row_size; i++){
			if (matrix_pos[i] > matrix_pos[i-1]) nnr++;
		}
		// CSF has this construction:
		//  POS_0 [0,NNR] -> 2 * posSize
		//  IDX_0 [idx_1, idx_2, ..., idx_nnr] -> nnr * idxSize
		//	POS_1 [pos_1 (idx_1 start), pos_2 (idx_1 end/idx_2 end),
		//	  ..., pos_nnr (idx_nnr-1 end/ idx_nnr start),
		//	  pos_nnr+1 (idx_nnr end)] -> (nnr+1) * posSize
		//	IDX_1 [idx_0, ..., idx_nnz] -> nnz * idxSize
		//	VALS  [val_0 ,..., val_nnz] -> nnz * vals
		int nnz = matrix_nnz;
		csfSize = params->getPosSize() * (3 + nnr) +
							params->getIdxSize() * (nnr + nnz) +
							params->getDataSize() * (nnz);
	}
	else{csfSize = 0;}

	return csfSize;
}

// Gets a CSR matrix and calculates the CSF size
void Matrix::calculateCSFSize_csr(CSR_format * matrix_csr){

	// nnr: number of non-zero rows
	int nnr = 0;
	// CSF size calculation
	// If it has no nnz, size is zero
	if(matrix_csr->nnz){
		// Find the number of nnr
		for (int i = 1; i<=matrix_csr->row_size; i++){
			if (matrix_csr->pos[i] > matrix_csr->pos[i-1]) nnr++;
		}
		// CSF has this construction:
		//  POS_0 [0,NNR] -> 2 * posSize
		//  IDX_0 [idx_1, idx_2, ..., idx_nnr] -> nnr * idxSize
		//	POS_1 [pos_1 (idx_1 start), pos_2 (idx_1 end/idx_2 end),
		//	  ..., pos_nnr (idx_nnr-1 end/ idx_nnr start),
		//	  pos_nnr+1 (idx_nnr end)] -> (nnr+1) * posSize
		//	IDX_1 [idx_0, ..., idx_nnz] -> nnz * idxSize
		//	VALS  [val_0 ,..., val_nnz] -> nnz * vals
		int nnz = matrix_csr->nnz;

		matrix_csr->csfSize = params->getPosSize() * (3 + nnr) +
													params->getIdxSize() * (nnr + nnz) +
													params->getDataSize() * (nnz);
	}
	else{matrix_csr->csfSize = 0;}

	matrix_csr->nnr = nnr;

	return;
}

// Gets a CSR matrix and calculates the CSR size
void Matrix::calculateCSRSize_csr(CSR_format * matrix_csr){
	// CSR size calculation
	if(matrix_csr->nnz){

		// CSR has this construction:
		//  POS  [row_0, row_1, ... row_n-1, row_n] -> (row_size+1) * posSize
		//	IDX  [idx_0, ..., idx_nnz] -> nnz * idxSize
		//	VALS [val_0 ,..., val_nnz] -> nnz * vals
		int csr_size = matrix_csr->nnz * (params->getDataSize() +params->getIdxSize()) +
				(matrix_csr->row_size+1) * params->getPosSize();
		matrix_csr->csrSize = csr_size;
	}
	else{matrix_csr->csrSize = 0;}

	return;
}

// Gets a CSR matrix and calculates the COO size
void Matrix::calculateCOOSize_csr(CSR_format * matrix_csr){
	// CSR size calculation
	if(matrix_csr->nnz){

		// COO has this construction:
		//  ROWS [row_0, ..., row_nnz] -> nnz * idxSize
		//	COLS [col_0, ..., col_nnz] -> nnz * idxSize
		//	VALS [val_0, ..., val_nnz] -> nnz * vals
		int coo_size = matrix_csr->nnz *
			(params->getDataSize() + 2*params->getIdxSize());
		matrix_csr->cooSize = coo_size;
	}
	else{matrix_csr->cooSize = 0;}

	return;
}



#define CALL_AND_CHECK_STATUS(function, error_message) do { \
	if(function != SPARSE_STATUS_SUCCESS)											\
	{																													\
		printf(error_message); fflush(0);												\
	}																													\
} while(0)

void Matrix::print_csr_sparse_d(const sparse_matrix_t csrA){

	//Read Matrix Data and Print it
	int row, col;
	sparse_index_base_t indextype;
	int * bi, *ei;
	int * j;
	double *rv;
	sparse_status_t status = mkl_sparse_d_export_csr(csrA, &indextype, &row, &col, &bi, &ei, &j, &rv);
	if (status==SPARSE_STATUS_SUCCESS)
	{
		printf("SparseMatrix(%d x %d) [base:%d]\n", row, col, indextype);
		for (int r = 0; r<row; ++r)
		{
			for (int idx = bi[r]; idx<ei[r]; ++idx)
			{
				printf("  <%d, %d> \t %f\n", r, j[idx], rv[idx]);
			}
		}
	}
	return;
}

// Since B and O are dense we do not need to do much!
void Matrix::CSRTimesDense(int i, int j, int k, int * cycles_comp, uint32_t & pproductSize){

	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i][j]);
	int b_cols = std::min(params->getNumDenseCols(), params->getTileSize());
	// Number of nnz entries in A basic tile define how many cycles
	//	will be spent for intersection (i.e., computation / in SpMM case)
	*cycles_comp = a_csr->nnz * b_cols;
	pproductSize = a_csr->nnr * b_cols;

	return;
}


// Please note that this fucntion only applies output products to the LOG datastructure
//	and does not do anything to the actual output datastructure
// CSRTimesCSR function is the core computation function, it is where the real
//	matrix multiplication takes place.
// The ID of the A & B basic tiles are given to the function.
//	It does the computation, updates the output, and output log matrices and finally
//	reports the number of cycles it took for the multiplication
void Matrix::CSRTimesCSROnlyLogUpdated(int i, int j, int k){

	float alpha = 1.0;

	sparse_index_base_t indexing;
	MKL_INT rows_d, cols_d;
	MKL_INT *columns_D = NULL, *pointerE_D = NULL, *pointerB_D = NULL;
	double *value_D = NULL;

	sparse_matrix_t csrA = NULL, csrB = NULL, csrD = NULL;

	double  *values_A = NULL, *values_B = NULL;
	MKL_INT *columns_A = NULL, *columns_B = NULL;
	MKL_INT *rowIndex_A = NULL, *rowIndex_B = NULL;

	// Two inputs (A & B), output, and output log tiles
	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i][j]);
	CSR_format * b_csr = &(b_csr_tiled->csr_tiles[j][k]);
	CSR_format * o_log_csr =
		&(o_csr_tiled_log->csr_tiles[i-row_start_log][k-col_start_log]);

	// #non-zero in the output log tile before computation
	int nnz_log_tile_pre = o_log_csr->nnz;
	int nnz_log_tile_post = nnz_log_tile_pre;

	// if either of them is all zero then just skip; there is no computation to do
	if ((a_csr->nnz == 0) | (b_csr->nnz == 0))
		return;

	// Basically the numbers should be aligned for AVX512
	// I tried it in the allocation part and it does not work

	// alligned malloc fixes the problem
	values_A = (double *)mkl_malloc(sizeof(double) * a_csr->nnz, ALIGN);
	columns_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * a_csr->nnz, ALIGN);
	rowIndex_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size + 1), ALIGN);

	values_B = (double *)mkl_malloc(sizeof(double) * b_csr->nnz, ALIGN);
	columns_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * b_csr->nnz, ALIGN);
	rowIndex_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (b_csr->row_size + 1), ALIGN);

	memcpy(values_A, a_csr->vals, sizeof(double) *a_csr->nnz);
	memcpy(columns_A, a_csr->idx, sizeof(MKL_INT) *a_csr->nnz);
	memcpy(rowIndex_A, a_csr->pos, sizeof(MKL_INT) * (a_csr->row_size+1));

	memcpy(values_B, b_csr->vals, sizeof(double) *b_csr->nnz);
	memcpy(columns_B, b_csr->idx, sizeof(MKL_INT) *b_csr->nnz);
	memcpy(rowIndex_B, b_csr->pos, sizeof(MKL_INT) * (b_csr->row_size+1));

	// convert A and B tiles to MKL readable formats
	mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, a_csr->row_size,
			a_csr->col_size, rowIndex_A, (rowIndex_A)+1,
			columns_A, values_A );
	mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, b_csr->row_size,
			b_csr->col_size, rowIndex_B, (rowIndex_B)+1,
			columns_B, values_B );

	// Do the actual MKL sparse multiplication
	CALL_AND_CHECK_STATUS(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrD),
			"Error after MKL_SPARSE_SPMM \n");
	// Order outputs; MKL does not do it
	CALL_AND_CHECK_STATUS(mkl_sparse_order(csrD), "Error after mkl_sparse_order \n");

	// Export the mult output to see if it has produced any output or all zeros
	mkl_sparse_d_export_csr(csrD, &indexing, &rows_d, &cols_d, &pointerB_D,
			&pointerE_D, &columns_D, &value_D);

	// If mult has zero nnz, then we do not need to touch csrC (o_csr)
	if(pointerB_D[rows_d]>0){
		// The following if else statement is for the output log
		//	It is exactly as above's if else statement, but updates the output log instead
		// if both mult and o_log_csr have nnz, then add them up
		if(o_log_csr->nnz >0){
			// C & E matrix variables
			sparse_matrix_t csrC = NULL, csrE = NULL;
			double  *values_C = NULL;
			MKL_INT *columns_C = NULL;
			MKL_INT *rowIndex_C = NULL;
			MKL_INT rows_e, cols_e;
			MKL_INT *columns_E = NULL, *pointerE_E = NULL, *pointerB_E = NULL;
			double *value_E = NULL;

			// convert C tile to MKL readable formats
			mkl_sparse_d_create_csr( &csrC, SPARSE_INDEX_BASE_ZERO, o_log_csr->row_size,
					o_log_csr->col_size, o_log_csr->pos, (o_log_csr->pos)+1,
					o_log_csr->idx, o_log_csr->vals );

			// Add stored psums to current psums
			CALL_AND_CHECK_STATUS(mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, csrC, alpha, csrD, &csrE),
					"Error after MKL_SPARSE_D_ADD \n");

			// Export add result
			mkl_sparse_d_export_csr(csrE, &indexing, &rows_e, &cols_e, &pointerB_E,
					&pointerE_E, &columns_E, &value_E);

			// free the memory used for previous psums
			mkl_free(o_log_csr->idx);
			mkl_free(o_log_csr->vals);
			// update output values
			o_log_csr->nnz = pointerB_E[rows_e];
			o_log_csr->row_size = rows_e;
			o_log_csr->col_size = cols_e;
			// dynamically allocate memory for new psum output
			o_log_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_log_csr->nnz), ALIGN);
			o_log_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_log_csr->nnz), ALIGN);

			// copy data to o_log_csr
			memcpy(o_log_csr->pos, pointerB_E, sizeof(MKL_INT) * ((o_log_csr->row_size)+1));
			memcpy(o_log_csr->idx, columns_E, sizeof(MKL_INT)  * o_log_csr->nnz);
			memcpy(o_log_csr->vals, value_E, sizeof(double) * o_log_csr->nnz);

			// free memory used for csrC and csrE
			if( mkl_sparse_destroy( csrC ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}
			mkl_free(values_C); mkl_free(columns_C); mkl_free(rowIndex_C);

			if( mkl_sparse_destroy( csrE ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

		}
		else{
			// update output values
			o_log_csr->nnz = pointerB_D[rows_d];
			o_log_csr->row_size = rows_d;
			o_log_csr->col_size = cols_d;
			// dynamically allocate memory for new psum output
			o_log_csr->pos = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size+1), ALIGN);
			//o_log_csr->idx  = new int[o_log_csr->nnz];
			//o_log_csr->vals = new double[o_log_csr->nnz];
			o_log_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_log_csr->nnz), ALIGN);
			o_log_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_log_csr->nnz), ALIGN);

			// copy data to o_log_csr
			memcpy(o_log_csr->pos, pointerB_D, sizeof(MKL_INT) * ((o_log_csr->row_size)+1));
			memcpy(o_log_csr->idx, columns_D, sizeof(MKL_INT) * o_log_csr->nnz);
			memcpy(o_log_csr->vals, value_D, sizeof(double ) * o_log_csr->nnz);
		}
	}

	// We are done. free csrA, csrB, and csrD
	if( mkl_sparse_destroy( csrA ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrA \n");fflush(0);}
	if( mkl_sparse_destroy( csrB ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrB \n");fflush(0);}
	if( mkl_sparse_destroy( csrD ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

	mkl_free(values_A); mkl_free(columns_A); mkl_free(rowIndex_A);
	mkl_free(values_B); mkl_free(columns_B); mkl_free(rowIndex_B);

	// Update basic tile of output log size
	calculateCSFSize_csr(o_log_csr);
	calculateCOOSize_csr(o_log_csr);

	// #Non-zero after computation
	nnz_log_tile_post = o_log_csr->nnz;
	// Update total #non-zeros in the output log matrix
	//	This will make finding the COO size super simple!
	o_csr_tiled_log->nnz =
		o_csr_tiled_log->nnz + nnz_log_tile_post - nnz_log_tile_pre;

	return;
}



// CSRTimesCSR function is the core computation function, it is where the real
//	matrix multiplication takes place.
// The ID of the A & B basic tiles are given to the function.
//	It does the computation, updates the output, and output log matrices and finally
//	reports the number of cycles it took for the multiplication
void Matrix::CSRTimesCSR(int i, int j, int k, int * cycles_comp, uint32_t & pproductSize){

	uint64_t comp_nnz = 0, num_nnz = 0;
	*cycles_comp = 0;

	float alpha = 1.0;

	sparse_index_base_t indexing;
	MKL_INT rows_d, cols_d;
	MKL_INT *columns_D = NULL, *pointerE_D = NULL, *pointerB_D = NULL;
	double *value_D = NULL;

	sparse_matrix_t csrA = NULL, csrB = NULL, csrD = NULL;

	double  *values_A = NULL, *values_B = NULL;
	MKL_INT *columns_A = NULL, *columns_B = NULL;
	MKL_INT *rowIndex_A = NULL, *rowIndex_B = NULL;

	// Two inputs (A & B), output, and output log tiles
	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i][j]);
	CSR_format * b_csr = &(b_csr_tiled->csr_tiles[j][k]);
	CSR_format * o_csr = &(o_csr_tiled->csr_tiles[i][k]);
	CSR_format * o_log_csr =
		&(o_csr_tiled_log->csr_tiles[i-row_start_log][k-col_start_log]);


	// #non-zero in the output log tile before computation
	int nnz_log_tile_pre = o_log_csr->nnz;
	int nnz_log_tile_post = nnz_log_tile_pre;

	// if either of them is all zero then just skip; there is no computation to do
	if ((a_csr->nnz == 0) | (b_csr->nnz == 0))
		return;

	// Basically the numbers should be aligned for AVX512
	// I tried it in the allocation part and it does not work

	// alligned malloc fixes the problem
	values_A = (double *)mkl_malloc(sizeof(double) * a_csr->nnz, ALIGN);
	columns_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * a_csr->nnz, ALIGN);
	rowIndex_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size + 1), ALIGN);

	values_B = (double *)mkl_malloc(sizeof(double) * b_csr->nnz, ALIGN);
	columns_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * b_csr->nnz, ALIGN);
	rowIndex_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (b_csr->row_size + 1), ALIGN);

	memcpy(values_A, a_csr->vals, sizeof(double) *a_csr->nnz);
	memcpy(columns_A, a_csr->idx, sizeof(MKL_INT) *a_csr->nnz);
	memcpy(rowIndex_A, a_csr->pos, sizeof(MKL_INT) * (a_csr->row_size+1));

	memcpy(values_B, b_csr->vals, sizeof(double) *b_csr->nnz);
	memcpy(columns_B, b_csr->idx, sizeof(MKL_INT) *b_csr->nnz);
	memcpy(rowIndex_B, b_csr->pos, sizeof(MKL_INT) * (b_csr->row_size+1));

	// convert A and B tiles to MKL readable formats
	mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, a_csr->row_size,
			a_csr->col_size, rowIndex_A, (rowIndex_A)+1,
			columns_A, values_A );
	mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, b_csr->row_size,
			b_csr->col_size, rowIndex_B, (rowIndex_B)+1,
			columns_B, values_B );

	// Do the actual MKL sparse multiplication
	CALL_AND_CHECK_STATUS(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrD),
			"Error after MKL_SPARSE_SPMM \n");
	// Order outputs; MKL does not do it
	CALL_AND_CHECK_STATUS(mkl_sparse_order(csrD), "Error after mkl_sparse_order \n");

	// Export the mult output to see if it has produced any output or all zeros
	mkl_sparse_d_export_csr(csrD, &indexing, &rows_d, &cols_d, &pointerB_D,
			&pointerE_D, &columns_D, &value_D);

	// number of nnz produced during the calculation for idealModel;
	//   Does not affect skipModel
	num_nnz = pointerB_D[rows_d];
//	if(params->getIntersectModel() == intersect::idealModel){
		comp_nnz = std::accumulate(value_D, value_D + num_nnz,  comp_nnz);
//	}
	// Since we have an output stationary dataflow in the bottom DOT
	//	this will return the correct value.
	pproductSize = num_nnz *(params->getDataSize() + 2*params->getIdxSize());
	// If mult has zero nnz, then we do not need to touch csrC (o_csr)
	if(pointerB_D[rows_d]>0){
		// if both mult and o_csr have nnz, then add them up
		// NOTE: The first if else statement is for the actual output matrix
		//	The second if e;se statement is for the temporary output log (outerSPACE)
		if(o_csr->nnz >0){
			// C & E matrix variables
			sparse_matrix_t csrC = NULL, csrE = NULL;
			double  *values_C = NULL;
			MKL_INT *columns_C = NULL;
			MKL_INT *rowIndex_C = NULL;
			MKL_INT rows_e, cols_e;
			MKL_INT *columns_E = NULL, *pointerE_E = NULL, *pointerB_E = NULL;
			double *value_E = NULL;

			// convert C tile to MKL readable formats
			mkl_sparse_d_create_csr( &csrC, SPARSE_INDEX_BASE_ZERO, o_csr->row_size,
					o_csr->col_size, o_csr->pos, (o_csr->pos)+1,
					o_csr->idx, o_csr->vals );

			// Add stored psums to current psums
			CALL_AND_CHECK_STATUS(mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, csrC, alpha, csrD, &csrE),
					"Error after MKL_SPARSE_D_ADD \n");

			// Export add result
			mkl_sparse_d_export_csr(csrE, &indexing, &rows_e, &cols_e, &pointerB_E,
					&pointerE_E, &columns_E, &value_E);

			// free the memory used for previous psums
			mkl_free(o_csr->idx);
			mkl_free(o_csr->vals);
			//delete [] o_csr->idx;
			//delete [] o_csr->vals;
			// update output values
			o_csr->nnz = pointerB_E[rows_e];
			o_csr->row_size = rows_e;
			o_csr->col_size = cols_e;

			// dynamically allocate memory for new psum output
			o_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_csr->nnz), ALIGN);
			o_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_csr->nnz), ALIGN);

			//o_csr->idx  = new int[o_csr->nnz];
			//o_csr->vals = new double[o_csr->nnz];

			// copy data to o_csr
			memcpy(o_csr->pos, pointerB_E, sizeof(MKL_INT) * ((o_csr->row_size)+1));
			memcpy(o_csr->idx, columns_E, sizeof(MKL_INT)  * o_csr->nnz);
			memcpy(o_csr->vals, value_E, sizeof(double) * o_csr->nnz);

			// free memory used for csrC and csrE
			if( mkl_sparse_destroy( csrC ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}
			mkl_free(values_C); mkl_free(columns_C); mkl_free(rowIndex_C);

			if( mkl_sparse_destroy( csrE ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

		}
		// If o_csr has no nnz and we need to copy csrD to o_csr
		else{
			// update output values
			o_csr->nnz = pointerB_D[rows_d];
			o_csr->row_size = rows_d;
			o_csr->col_size = cols_d;

			// dynamically allocate memory for new psum output
			o_csr->pos = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_csr->row_size+1), ALIGN);
			o_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_csr->nnz), ALIGN);
			o_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_csr->nnz), ALIGN);

			//o_csr->idx  = new int[o_csr->nnz];
			//o_csr->vals = new double[o_csr->nnz];

			// copy data to o_csr
			memcpy(o_csr->pos, pointerB_D, sizeof(MKL_INT) * ((o_csr->row_size)+1));
			memcpy(o_csr->idx, columns_D, sizeof(MKL_INT) * o_csr->nnz);
			memcpy(o_csr->vals, value_D, sizeof(double ) * o_csr->nnz);
		}

		// The following if else statement is for the output log
		//	It is exactly as above's if else statement, but updates the output log instead
		// if both mult and o_log_csr have nnz, then add them up
		if(o_log_csr->nnz >0){
			// C & E matrix variables
			sparse_matrix_t csrC = NULL, csrE = NULL;
			double  *values_C = NULL;
			MKL_INT *columns_C = NULL;
			MKL_INT *rowIndex_C = NULL;
			MKL_INT rows_e, cols_e;
			MKL_INT *columns_E = NULL, *pointerE_E = NULL, *pointerB_E = NULL;
			double *value_E = NULL;

			// convert C tile to MKL readable formats
			mkl_sparse_d_create_csr( &csrC, SPARSE_INDEX_BASE_ZERO, o_log_csr->row_size,
					o_log_csr->col_size, o_log_csr->pos, (o_log_csr->pos)+1,
					o_log_csr->idx, o_log_csr->vals );

			// Add stored psums to current psums
			CALL_AND_CHECK_STATUS(mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, csrC, alpha, csrD, &csrE),
					"Error after MKL_SPARSE_D_ADD \n");

			// Export add result
			mkl_sparse_d_export_csr(csrE, &indexing, &rows_e, &cols_e, &pointerB_E,
					&pointerE_E, &columns_E, &value_E);

			// free the memory used for previous psums
			mkl_free(o_log_csr->idx);
			mkl_free(o_log_csr->vals);
			//delete [] o_log_csr->idx;
			//delete [] o_log_csr->vals;
			// update output values
			o_log_csr->nnz = pointerB_E[rows_e];
			o_log_csr->row_size = rows_e;
			o_log_csr->col_size = cols_e;
/*
			if((rows_e!=32) | (cols_e!=32)){
				printf("ridi!\n");
				exit(1);
			}
*/
			// dynamically allocate memory for new psum output
			o_log_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_log_csr->nnz), ALIGN);
			o_log_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_log_csr->nnz), ALIGN);

			//o_log_csr->idx  = new int[o_log_csr->nnz];
			//o_log_csr->vals = new double[o_log_csr->nnz];

			// copy data to o_log_csr
			memcpy(o_log_csr->pos, pointerB_E, sizeof(MKL_INT) * ((o_log_csr->row_size)+1));
			memcpy(o_log_csr->idx, columns_E, sizeof(MKL_INT)  * o_log_csr->nnz);
			memcpy(o_log_csr->vals, value_E, sizeof(double) * o_log_csr->nnz);

			// free memory used for csrC and csrE
			if( mkl_sparse_destroy( csrC ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}
			mkl_free(values_C); mkl_free(columns_C); mkl_free(rowIndex_C);

			if( mkl_sparse_destroy( csrE ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

		}
		// If o_log_csr has no nnz and we need to copy csrD to o_log_csr
		else{
			// update output values
			o_log_csr->nnz = pointerB_D[rows_d];
			o_log_csr->row_size = rows_d;
			o_log_csr->col_size = cols_d;
/*
			if((rows_d!=32) | (cols_d!=32)){
				printf("ridi!\n");
				exit(1);
			}
			*/
			// dynamically allocate memory for new psum output
			o_log_csr->pos = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size+1), ALIGN);
			//o_log_csr->idx  = new int[o_log_csr->nnz];
			//o_log_csr->vals = new double[o_log_csr->nnz];
			o_log_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_log_csr->nnz), ALIGN);
			o_log_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_log_csr->nnz), ALIGN);

			// copy data to o_log_csr
			memcpy(o_log_csr->pos, pointerB_D, sizeof(MKL_INT) * ((o_log_csr->row_size)+1));
			memcpy(o_log_csr->idx, columns_D, sizeof(MKL_INT) * o_log_csr->nnz);
			memcpy(o_log_csr->vals, value_D, sizeof(double ) * o_log_csr->nnz);
		}

	}

	// We are done. free csrA, csrB, and csrD
	if( mkl_sparse_destroy( csrA ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrA \n");fflush(0);}
	if( mkl_sparse_destroy( csrB ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrB \n");fflush(0);}
	if( mkl_sparse_destroy( csrD ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

	mkl_free(values_A); mkl_free(columns_A); mkl_free(rowIndex_A);
	mkl_free(values_B); mkl_free(columns_B); mkl_free(rowIndex_B);

	// Calculate the new csf size of o_csr
	calculateCSFSize_csr(o_csr);
	calculateCOOSize_csr(o_csr);
	calculateCSRSize_csr(o_csr);
	// Update basic tile of output log size
	calculateCOOSize_csr(o_log_csr);
	// #Non-zero after computation
	nnz_log_tile_post = o_log_csr->nnz;
	// Update total #non-zeros in the output log matrix
	//	This will make finding the COO size super simple!
	o_csr_tiled_log->nnz =
		o_csr_tiled_log->nnz + nnz_log_tile_post - nnz_log_tile_pre;

	// Find the skip model cycles with 32-CAMs
	if (params->getIntersectModel() == intersect::skipModel){
		// 32-CAM based model
		int keyI = i * a_csr_tiled->col_size * b_csr_tiled->col_size;
		int keyJ = j * b_csr_tiled->col_size;
		int key = keyI + keyJ + k;

		auto dict_inquiry = skipCycles_map.find(key);
		*cycles_comp = dict_inquiry->second;
	}
	else if(params->getIntersectModel() == intersect::parBidirecSkipModel){
		// Parallel bidirectional 32-register based model
		int keyI = i * a_csr_tiled->col_size * b_csr_tiled->col_size;
		int keyJ = j * b_csr_tiled->col_size;
		int key = keyI + keyJ + k;

		auto dict_inquiry = skipCycles_map.find(key);

		*cycles_comp = std::max((int)comp_nnz, dict_inquiry->second);
	}

	else if(params->getIntersectModel() == intersect::idealModel){
		// Ideal case
		*cycles_comp = comp_nnz;
	}
	else if(params->getIntersectModel() == intersect::instantModel){
		*cycles_comp = 0;
	}
	else{
		printf("Skip model not supported!\n");
		exit(1);
	}

	//printf("%d\t%d\n", *cycles_comp, a_csr->csfSize + b_csr->csfSize);

	return;
}


// Please note that now, bytes_wr shows the amount of partial products produced
void Matrix::CSRTimesCSR(int i, int j, int k, int * cycles_comp,
		uint64_t * bytes_rd, uint64_t * bytes_wr){

	uint64_t comp_nnz = 0, num_nnz = 0;
	//uint64_t end_nnz = 0;

	float alpha = 1.0;

	sparse_index_base_t indexing;
	MKL_INT rows_e, cols_e;
	MKL_INT *columns_E = NULL, *pointerE_E = NULL, *pointerB_E = NULL;
	double *value_E = NULL;

	MKL_INT rows_d, cols_d;
	MKL_INT *columns_D = NULL, *pointerE_D = NULL, *pointerB_D = NULL;
	double *value_D = NULL;

	sparse_matrix_t csrA = NULL, csrB = NULL, csrC = NULL, csrD = NULL, csrE = NULL;

	double  *values_A = NULL, *values_B = NULL, *values_C = NULL;
	MKL_INT *columns_A = NULL, *columns_B = NULL, *columns_C = NULL;
	MKL_INT *rowIndex_A = NULL, *rowIndex_B = NULL, *rowIndex_C = NULL;

	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i][j]);
	CSR_format * b_csr = &(b_csr_tiled->csr_tiles[j][k]);
	CSR_format * o_csr = &(o_csr_tiled->csr_tiles[i][k]);

	*bytes_rd = 0;
	*bytes_wr = 0;
	*cycles_comp = 0;
	// if either of them is all zero then just skip; there is no computation to do
	if ((a_csr->nnz == 0) | (b_csr->nnz == 0))
		return;

	*bytes_rd = getCOOSize('O', i, k);

	// Basically the numbers should be aligned for AVX512
	// I tried it in the allocation part and it does not work

	// alligned malloc fixes the problem
	values_A = (double *)mkl_malloc(sizeof(double) * a_csr->nnz, ALIGN);
	columns_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * a_csr->nnz, ALIGN);
	rowIndex_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size + 1), ALIGN);

	values_B = (double *)mkl_malloc(sizeof(double) * b_csr->nnz, ALIGN);
	columns_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * b_csr->nnz, ALIGN);
	rowIndex_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (b_csr->row_size + 1), ALIGN);

	memcpy(values_A, a_csr->vals, sizeof(double) *a_csr->nnz);
	memcpy(columns_A, a_csr->idx, sizeof(MKL_INT) *a_csr->nnz);
	memcpy(rowIndex_A, a_csr->pos, sizeof(MKL_INT) * (a_csr->row_size+1));

	memcpy(values_B, b_csr->vals, sizeof(double) *b_csr->nnz);
	memcpy(columns_B, b_csr->idx, sizeof(MKL_INT) *b_csr->nnz);
	memcpy(rowIndex_B, b_csr->pos, sizeof(MKL_INT) * (b_csr->row_size+1));

	// For the ideal intersect we need to find the total number of multiplications
	//   This is an easy solution to get it wo/ overhead but limit it to only ideal
	//   For the correction check, use skipModel to get correct outputs
	/*
	if(params->getIntersectModel() == intersect::idealModel){
		std::fill(values_A, values_A+a_csr->nnz, 1.0);
		std::fill(values_B, values_B+b_csr->nnz, 1.0);
	}
*/
	// convert A and B tiles to MKL readable formats
	mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, a_csr->row_size,
			a_csr->col_size, rowIndex_A, (rowIndex_A)+1,
			columns_A, values_A );
	mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, b_csr->row_size,
			b_csr->col_size, rowIndex_B, (rowIndex_B)+1,
			columns_B, values_B );

	// Do the actual MKL sparse multiplication
	CALL_AND_CHECK_STATUS(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrD),
			"Error after MKL_SPARSE_SPMM \n");
	// Order outputs; MKL does not do it
	CALL_AND_CHECK_STATUS(mkl_sparse_order(csrD), "Error after mkl_sparse_order \n");

	// Export the mult output to see if it has produced any output or all zeros
	mkl_sparse_d_export_csr(csrD, &indexing, &rows_d, &cols_d, &pointerB_D,
			&pointerE_D, &columns_D, &value_D);

	// number of nnz produced during the calculation for idealModel;
	//   Does not affect skipModel
	//if(params->getIntersectModel() == intersect::idealModel){
		num_nnz = pointerB_D[rows_d];
		comp_nnz = std::accumulate(value_D, value_D + num_nnz,  comp_nnz);
	//}
	*bytes_wr = num_nnz *(params->getDataSize() + 2*params->getIdxSize());

	// If mult has zero nnz, then we do not need to touch csrC (o_csr)
	if(pointerB_D[rows_d]>0){
		// if both mult and o_csr have nnz, then add them up
		if(o_csr->nnz >0){
			// convert C tile to MKL readable formats
			mkl_sparse_d_create_csr( &csrC, SPARSE_INDEX_BASE_ZERO, o_csr->row_size,
					o_csr->col_size, o_csr->pos, (o_csr->pos)+1,
					o_csr->idx, o_csr->vals );

			// Add stored psums to current psums
			CALL_AND_CHECK_STATUS(mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, csrC, alpha, csrD, &csrE),
					"Error after MKL_SPARSE_D_ADD \n");

			// Export add result
			mkl_sparse_d_export_csr(csrE, &indexing, &rows_e, &cols_e, &pointerB_E,
					&pointerE_E, &columns_E, &value_E);

			// free the memory used for previous psums
			delete [] o_csr->idx;
			delete [] o_csr->vals;
			// update output values
			o_csr->nnz = pointerB_E[rows_e];
			o_csr->row_size = rows_e;
			o_csr->col_size = cols_e;

			// dynamically allocate memory for new psum output
			o_csr->idx  = new int[o_csr->nnz];
			o_csr->vals = new double[o_csr->nnz];

			// copy data to o_csr
			memcpy(o_csr->pos, pointerB_E, sizeof(MKL_INT) * ((o_csr->row_size)+1));
			memcpy(o_csr->idx, columns_E, sizeof(MKL_INT)  * o_csr->nnz);
			memcpy(o_csr->vals, value_E, sizeof(double) * o_csr->nnz);

			// free memory used for csrC and csrE
			if( mkl_sparse_destroy( csrC ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}
			mkl_free(values_C); mkl_free(columns_C); mkl_free(rowIndex_C);

			if( mkl_sparse_destroy( csrE ) != SPARSE_STATUS_SUCCESS){
				printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

		}
		// If o_csr has no nnz and we need to copy csrD to o_csr
		else{
			// update output values
			o_csr->nnz = pointerB_D[rows_d];
			o_csr->row_size = rows_d;
			o_csr->col_size = cols_d;

			// dynamically allocate memory for new psum output
			o_csr->pos = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size+1), ALIGN);
			o_csr->idx  = new int[o_csr->nnz];
			o_csr->vals = new double[o_csr->nnz];

			// copy data to o_csr
			memcpy(o_csr->pos, pointerB_D, sizeof(MKL_INT) * ((o_csr->row_size)+1));
			memcpy(o_csr->idx, columns_D, sizeof(MKL_INT) * o_csr->nnz);
			memcpy(o_csr->vals, value_D, sizeof(double ) * o_csr->nnz);
		}
	}

	// We are done. free csrA, csrB, and csrD
	if( mkl_sparse_destroy( csrA ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrA \n");fflush(0);}
	if( mkl_sparse_destroy( csrB ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrB \n");fflush(0);}
	if( mkl_sparse_destroy( csrD ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

	mkl_free(values_A); mkl_free(columns_A); mkl_free(rowIndex_A);
	mkl_free(values_B); mkl_free(columns_B); mkl_free(rowIndex_B);

	// Calculate the new csf size of o_csr
	calculateCSFSize_csr(o_csr);
	calculateCOOSize_csr(o_csr);
	calculateCSRSize_csr(o_csr);

	// TODO: COO specific. Add CSR and CSF support
	// Stats update
	// I am moving all the stats gathering in the scheduler except for cleanRdWr
	// stats->Accumulate_o_write(getCOOSize('O',i,k));
	//*bytes_compute += getCOOSize('O', i, k);
	//*bytes_wr = getCOOSize('O', i, k);

	/*
	// This is not the correct way of getting clean RdWr, it should be done in the scheduler
	end_nnz = o_csr->nnz;
	uint64_t cleanRdW = end_nnz - comp_nnz;

	cleanRdW = cleanRdW * (params->getIdxSize() + params->getPosSize() + params->getDataSize());
	stats->Accumulate_cleanRdWr(cleanRdW);
	*/
	// I am moving all the stats gathering in the scheduler except for cleanRdWr
	//stats->Accumulate_o_size(getCOOSize('O', i, k));

	// Find the skip model cycles with 32-CAMs
	if (params->getIntersectModel() == intersect::skipModel){
		// 32-CAM based model
		int keyI = i * a_csr_tiled->col_size * b_csr_tiled->col_size;
		int keyJ = j * b_csr_tiled->col_size;
		int key = keyI + keyJ + k;

		auto dict_inquiry = skipCycles_map.find(key);
		*cycles_comp = dict_inquiry->second;
	}
	else if(params->getIntersectModel() == intersect::parBidirecSkipModel){
		// Parallel bidirectional 32-register based model
		int keyI = i * a_csr_tiled->col_size * b_csr_tiled->col_size;
		int keyJ = j * b_csr_tiled->col_size;
		int key = keyI + keyJ + k;

		auto dict_inquiry = skipCycles_map.find(key);

		*cycles_comp = std::max((int)comp_nnz, dict_inquiry->second);
	}
	else if(params->getIntersectModel() == intersect::idealModel){
		// Ideal case
		*cycles_comp = comp_nnz;
	}
	else if(params->getIntersectModel() == intersect::instantModel){
		*cycles_comp = 0;
	}
	else{
		printf("Skip model not supported!\n");
		exit(1);
	}
	//skipModelCyclesCalc(i, j, k, cycles_comp);

	// Find the cycles with naive intersect unit
	//noSkipModelCyclesCalc(i, j, k, cycles_comp);

	return;
}


// Calculated how many cycles will spend in N-CAM model to multiply two basic tiles
//	Skip cycles of intersect(A(i,j), B(j,k))
void Matrix::parBidirecSkipModelCyclesCalc(int i, int j, int k, int & cycles_comp){
	int init_sum = 0;
	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i][j]);
	CSR_format * b_csc = &(b_csc_tiled->csr_tiles[j][k]);

	int num_cams = params->getNumCAMs();
	//printf("\n************************\n");
	//printf("(i:%d j:%d k:%d): \n", i, j, k);

	int a_rows = a_csr->row_size;
	int b_rows = b_csc->row_size;

	for(int i_start = 0; i_start < a_rows;/* EMPTY*/){
		int i_end = i_start;
		while(((a_csr->pos[i_end+1] - a_csr->pos[i_start]) < num_cams)
				& (i_end < a_rows )){i_end++;}
		int a_group_len = i_end - i_start;
		//printf("([%d, %d]), ", i_start, i_end);

		for(int j_start = 0; j_start < b_rows; /* EMPTY*/){
			int j_end = j_start;
			while(((b_csc->pos[j_end+1] - b_csc->pos[j_start]) < num_cams)
					& (j_end < b_rows )){j_end++; }
			int b_group_len = j_end - j_start;

			int min_len = std::min(a_group_len, b_group_len);
			int max_len = std::max(a_group_len, b_group_len);
			//printf("([%d, %d],[%d, %d])\n", i_start, i_end,	j_start, j_end);

			/*
			printf("(a_len, b_len) = (%d [%d, %d] : %d , %d [%d, %d] : %d)\n",
					a_group_len,i_start, i_end,(a_csr->pos[i_end] - a_csr->pos[i_start]),
					b_group_len, j_start, j_end,(b_csc->pos[j_end] - b_csc->pos[j_start]) );
					*/
			for(int t_long_idx = 0; t_long_idx < max_len; t_long_idx++){
				int batch_time = 0;
				for(int t_short_idx = 0; t_short_idx < min_len; t_short_idx++){
					if(a_group_len <= b_group_len){
						batch_time = std::max(batch_time,
								skipModelFiber(a_csr, b_csc,
									(i_start + t_short_idx),
									(j_start + (t_short_idx + t_long_idx)%b_group_len )));
/*
						printf("(a: %d, b: %d), ",
								(i_start + t_short_idx),
								(j_start + (t_short_idx + t_long_idx)%b_group_len ));
*/
					}
					else{
						batch_time = std::max(batch_time,
								skipModelFiber(a_csr, b_csc,
									(i_start + (t_short_idx + t_long_idx)%a_group_len),
									(j_start + t_short_idx) ) );
/*
						printf("(a: %d, b: %d), ",
									(i_start + (t_short_idx + t_long_idx)%a_group_len),
									(j_start + t_short_idx) );
*/
					}
				} // for t_short_idx
				init_sum += batch_time;
			} // for t_long_idx
			//printf("\n");
			j_start = j_end;
		} // for j_start
		i_start = i_end;
	} // for i_start
	cycles_comp = init_sum;
}


// Calculated how many cycles will spend in N-CAM model to multiply two basic tiles
//	Skip cycles of intersect(A(i,j), B(j,k))
void Matrix::skipModelCyclesCalc(int i, int j, int k, int * cycles_comp){

	int init_sum = 0;
	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i][j]);
	CSR_format * b_csc = &(b_csc_tiled->csr_tiles[j][k]);

	int a_rows = a_csr->row_size;
	int b_rows = b_csc->row_size;

	// Do not parallelize it! Since the task is small, it is inefficient
	//   to do it in parallel! Yes, I tried.
	// Create an array of all fiber intersects; will be accumulated at the end
	//int * partial_cycles = new int[a_rows*b_rows];
	for(int i_idx = 0; i_idx < a_rows; i_idx++){
		for(int j_idx = 0; j_idx < b_rows; j_idx++){
			//init_sum += skipModelFiber(a_csr, b_csc, i_idx, j_idx);
			if(params->getIntersectParallelism() == 1)
				init_sum += skipModelFiber(a_csr, b_csc, i_idx, j_idx);
			else
				init_sum += handleParallelIntersect(a_csr, b_csc, i_idx, j_idx);
			//partial_cycles[(i_idx*b_rows)+j_idx] = skipModelFiber(a_csr, b_csc, i_idx, j_idx);
		}
	}
	*cycles_comp = init_sum;
	//*cycles_comp = std::accumulate(partial_cycles, partial_cycles+(a_rows*b_rows), init_sum);
	//delete [] partial_cycles;
}

int Matrix::handleParallelIntersect(CSR_format *a_csr, CSR_format *b_csc,
		int i_idx, int j_idx){

	int parallelism = params->getIntersectParallelism();

	CSR_format * longer_fiber_tensor, * shorter_fiber_tensor;
	int idx_longer, idx_shorter;
	int nnz_a = a_csr->pos[i_idx+1] - a_csr->pos[i_idx];
	int nnz_b = b_csc->pos[j_idx+1] - b_csc->pos[j_idx];

	// TODO: try other way around too :D
	if(nnz_a >= nnz_b){
		longer_fiber_tensor = a_csr; shorter_fiber_tensor = b_csc;
		idx_longer = i_idx;	idx_shorter = j_idx;
	}
	else{
		longer_fiber_tensor = b_csc; shorter_fiber_tensor = a_csr;
		idx_longer = j_idx;	idx_shorter = i_idx;
	}
	int start_l = longer_fiber_tensor->pos[idx_longer];
	int end_l = longer_fiber_tensor->pos[idx_longer+1];

	int start_s = shorter_fiber_tensor->pos[idx_shorter];
	int end_s = shorter_fiber_tensor->pos[idx_shorter+1];

	int maccs1, maccs2, time1, time2;

	// Just assume 2 way parallelism for now
	//for(int i=0;i<parallelism;i++){
	int start_l1 = start_l, end_l2 = end_l;
	int end_l1 = start_l + ((end_l - start_l) / parallelism);
	int start_l2 = end_l1;
	time1 = skipModelPartialFiber(longer_fiber_tensor, shorter_fiber_tensor, idx_longer, idx_shorter,
			start_l1, end_l1, start_s, end_s, maccs1);
	time2 = skipModelPartialFiber(longer_fiber_tensor, shorter_fiber_tensor, idx_longer, idx_shorter,
			start_l2, end_l2, start_s, end_s, maccs2);
	//}

	//if((time1!=0) & (time2!=0))
	//printf("%d %d %d %d\n", time1, time2, maccs1, maccs2);
	int skipTime = time1 + std::max(maccs2, std::max(time2-time1, 0));

	return skipTime;
}

int Matrix::skipModelPartialFiber(CSR_format *a_csr, CSR_format *b_csc, int i_idx, int j_idx,
		int start_a, int end_a, int start_b, int end_b, int &effectual_MACCs){

	/*
	int start_a = a_csr->pos[i_idx];
	int end_a = a_csr->pos[i_idx+1];

	int start_b = b_csc->pos[j_idx];
	int end_b = b_csc->pos[j_idx+1];
	*/

	int curr_b = start_b;
	int curr_a = start_a;

	effectual_MACCs = 0;
	//early termination if any of the rows empty
	if ((start_a==end_a) | (start_b == end_b)) return 0;

	// Find the number of effectual MACCs for parallelism computation
	std::vector<int> v(params->getTileSize());
	std::vector<int>::iterator it;
	it = std::set_intersection(&a_csr->idx[start_a], &a_csr->idx[end_a],
			&b_csc->idx[start_b], &b_csc->idx[end_b], v.begin() );
	effectual_MACCs = std::distance(v.begin(), it);

	int cycles_a = 0, cycles_b = 0;
	int num_cams = params->getNumCAMs();

	if(num_cams == 0)
		printf("There is a problem");
	int stride_a = (end_a - start_a) / num_cams;
	stride_a = ((end_a - start_a) % num_cams == 0 )? stride_a : stride_a+1;

	int stride_b = (end_b - start_b) / num_cams;
	stride_b = ((end_b - start_b) % num_cams == 0 )? stride_b : stride_b+1;

	int reg_a = start_a;
	int reg_b = start_b;

	int progress_pos_a = 0;
	int progress_pos_b = 0;

	int coord_a = 0, coord_b = 0;

	if ((stride_a == 1) & (stride_b == 1)){
		// fast path; each look-up will take 1 cycle
		// functionality check : passed!
		coord_a = 0; coord_b = 0;
		while((curr_a<end_a) & (curr_b<end_b)){
			// Next coord or feedback from the other side
			coord_a = std::max(a_csr->idx[curr_a], coord_a);
			coord_b = std::max(b_csc->idx[curr_b], coord_b);

			// Find the next position; either the position of the curr coord i
			//	or the closest coord's pos after it
			curr_a = nextPos(a_csr, start_a, end_a, coord_a, &progress_pos_a);
			curr_b = nextPos(b_csc, start_b, end_b, coord_b, &progress_pos_b);
			// Go to next pos
			curr_a++; curr_b++;
			// increase  cycles for each scan unit
			cycles_a++; cycles_b++;
			// pass the coord to the other scan/serach unit
			coord_a = std::max(coord_a , coord_b);
			coord_b = coord_a;
		}
	}
	else {
		// Queue to hold the cycle and coord of the other scan/search unit
		// pairq_a gets the values from A and delivers them to B
		std::queue< std::pair<int,int> > pairq_a;
		// pairq_b gets the values from B and delivers them to A
		std::queue< std::pair<int,int> > pairq_b;

		while((curr_a<end_a) | (curr_b<end_b)){
			// Get current coords
			if (curr_a >= end_a){
				coord_a = EOS_MSG; curr_a = end_a;
			}
			else
				coord_a = a_csr->idx[curr_a];
			if (curr_b >= end_b){
				coord_b = EOS_MSG; curr_b = end_b;
			}
			else
				coord_b = b_csc->idx[curr_b];

			// Comparing coord_b against the feedback of queue A
			if(!pairq_a.empty()){
				std::pair<int,int> temp_a = pairq_a.front();
				// if timing matches, consume the front and use coord
				//	if it is higher than B's current coord
				if  (temp_a.first <= cycles_b ){
					pairq_a.pop();
					if (temp_a.second > coord_b){
						coord_b = temp_a.second;
					}
				}
			}
			// Comparing coord_a against the feedback of queue B
			if(!pairq_b.empty()){
				std::pair<int,int>  temp_b = pairq_b.front();
				if  (temp_b.first <= cycles_a ){
					pairq_b.pop();
					if (temp_b.second > coord_a){
						coord_a = temp_b.second;
					}
				}
			}

			// These guards are for when one of the scanners have reached EOS
			//		but the other scanner is still producing addresses
			if (curr_a != end_a){
				// Get next position based on coord_a
				curr_a = nextPos(a_csr, start_a, end_a, coord_a, &progress_pos_a);
				// What is the CAM search cycles (parallel) without linear search (w/o help of register)
				int cam_search_a    = (coord_a == a_csr->idx[curr_a])? (curr_a-start_a) % stride_a : stride_a-1 ;
				// CAM has not found the value; Do a linear search between two CAM pos'
				int linear_search_a = (curr_a - reg_a);
				cycles_a += (1 + std::min(linear_search_a, cam_search_a));
				// If we reached EOS, update coord
				if (curr_a == end_a){
					coord_a = EOS_MSG;
				}
				else{
					coord_a = a_csr->idx[curr_a];
				}
				// send coord to scanner B
				pairq_a.push(std::make_pair(cycles_a,  coord_a));
				// increment pos to next pos
				curr_a++;
				// Update the linear serach helper register
				reg_a = curr_a;
			}
			if (curr_b != end_b){
				// Get next position based on coord_b
				curr_b = nextPos(b_csc, start_b, end_b, coord_b, &progress_pos_b);
				// What is the CAM search cycles (parallel) without linear search (w/o help of register)
				int cam_search_b    = (coord_b == b_csc->idx[curr_b])? (curr_b-start_b) % stride_b : stride_b-1 ;
				// CAM has not found the value; Do a linear search between two CAM pos'
				int linear_search_b = (curr_b - reg_b);
				cycles_b += (1 + std::min(linear_search_b, cam_search_b));
				// If we reached EOS, update coord
				if (curr_b == end_b){
					coord_b = EOS_MSG;
				}
				else{
					coord_b = b_csc->idx[curr_b];
				}
				// send coord to scanner A
				pairq_b.push(std::make_pair(cycles_b,  coord_b));
				// increment pos to next pos
				curr_b++;
				// Update the linear serach helper register
				reg_b = curr_b;
			}
		}
	}
	return std::max(cycles_a, cycles_b);
}

int Matrix::nextPos(CSR_format * a_csr, int pos_start, int pos_end,
		int coord, int *start_pos){

	int start_a = std::max(*start_pos, pos_start);
	int end_a = std::max(*start_pos, pos_end);
	// INT_MAX is eos, early exit
	if (coord == EOS_MSG)
		return end_a;
	// search in the row to find the pos
	for(int i=start_a; i<end_a; i++){
		*start_pos = i;
		if (a_csr->idx[i]>=coord)
			return i;
	}
	return end_a;
}

// Calculates the intersection cycle for intersecting two fibers
//  (fiber i_idx of A and fiber j_idx of B)
int Matrix::skipModelFiber(CSR_format *a_csr, CSR_format *b_csc, int i_idx, int j_idx){

	int start_a = a_csr->pos[i_idx];
	int curr_a = start_a;
	int end_a = a_csr->pos[i_idx+1];

	int start_b = b_csc->pos[j_idx];
	int curr_b = start_b;
	int end_b = b_csc->pos[j_idx+1];

	//early termination if any of the rows empty
	if ((start_a==end_a) | (start_b == end_b)) return 0;

	int cycles_a = 0, cycles_b = 0;
	int num_cams = params->getNumCAMs();

	if(num_cams == 0)
		printf("There is a problem");
	int stride_a = (end_a - start_a) / num_cams;
	stride_a = ((end_a - start_a) % num_cams == 0 )? stride_a : stride_a+1;

	int stride_b = (end_b - start_b) / num_cams;
	stride_b = ((end_b - start_b) % num_cams == 0 )? stride_b : stride_b+1;

	//printf("%d(%d,%d) %d(%d,%d)\n",stride_a, start_a, end_a, stride_b, start_b, end_b); fflush(stdout);
	int reg_a = start_a;
	int reg_b = start_b;

	int progress_pos_a = 0;
	int progress_pos_b = 0;

	int coord_a = 0, coord_b = 0;

	if ((stride_a == 1) & (stride_b == 1)){
		// fast path; each look-up will take 1 cycle
		// functionality check : passed!
		coord_a = 0; coord_b = 0;
		while((curr_a<end_a) & (curr_b<end_b)){
			// Next coord or feedback from the other side
			coord_a = std::max(a_csr->idx[curr_a], coord_a);
			coord_b = std::max(b_csc->idx[curr_b], coord_b);

			// Find the next position; either the position of the curr coord i
			//	or the closest coord's pos after it
			curr_a = nextPos(a_csr, i_idx, coord_a, &progress_pos_a);
			curr_b = nextPos(b_csc, j_idx, coord_b, &progress_pos_b);
			// Go to next pos
			curr_a++; curr_b++;
			// increase  cycles for each scan unit
			cycles_a++; cycles_b++;
			// pass the coord to the other scan/serach unit
			coord_a = std::max(coord_a , coord_b);
			coord_b = coord_a;
			//printf("%d . %d - ",coord_a, coord_b);

		}
	}
	else {
		// slow path

		/*
		printf("a\n");
		for(int i = start_a; i<end_a; i++)
			printf("%d ",a_csr->idx[i]);
		printf("\nb\n");
		for(int i = start_b; i<end_b; i++)
			printf("%d ",b_csc->idx[i]);
		printf("\n");
		*/

		// Queue to hold the cycle and coord of the other scan/search unit
		// pairq_a gets the values from A and delivers them to B
		std::queue< std::pair<int,int> > pairq_a;
		// pairq_b gets the values from B and delivers them to A
		std::queue< std::pair<int,int> > pairq_b;

		while((curr_a<end_a) | (curr_b<end_b)){

			// Get current coords
			if (curr_a >= end_a){
				coord_a = EOS_MSG; curr_a = end_a;
			}
			else
				coord_a = a_csr->idx[curr_a];
			if (curr_b >= end_b){
				coord_b = EOS_MSG; curr_b = end_b;
			}
			else
				coord_b = b_csc->idx[curr_b];

			// Comparing coord_b against the feedback of queue A
			if(!pairq_a.empty()){
				std::pair<int,int> temp_a = pairq_a.front();
				// if timing matches, consume the front and use coord
				//	if it is higher than B's current coord
				if  (temp_a.first <= cycles_b ){
					pairq_a.pop();
					if (temp_a.second > coord_b){
						coord_b = temp_a.second;
					}
				}
			}
			// Comparing coord_a against the feedback of queue B
			if(!pairq_b.empty()){
				std::pair<int,int>  temp_b = pairq_b.front();
				if  (temp_b.first <= cycles_a ){
					pairq_b.pop();
					if (temp_b.second > coord_a){
						coord_a = temp_b.second;
					}
				}
			}

			// These guards are for when one of the scanners have reached EOS
			//		but the other scanner is still producing addresses
			if (curr_a != end_a){
				// Get next position based on coord_a
				curr_a = nextPos(a_csr, i_idx, coord_a, &progress_pos_a);
				// What is the CAM search cycles (parallel) without linear search (w/o help of register)
				int cam_search_a    = (coord_a == a_csr->idx[curr_a])? (curr_a-start_a) % stride_a : stride_a-1 ;
				// CAM has not found the value; Do a linear search between two CAM pos'
				int linear_search_a = (curr_a - reg_a);
				cycles_a += (1 + std::min(linear_search_a, cam_search_a));
				// If we reached EOS, update coord
				if (curr_a == end_a){
					coord_a = EOS_MSG;
				}
				else{
					coord_a = a_csr->idx[curr_a];
				}
				// send coord to scanner B
				pairq_a.push(std::make_pair(cycles_a,  coord_a));
				// increment pos to next pos
				curr_a++;
				// Update the linear serach helper register
				reg_a = curr_a;
			}
			if (curr_b != end_b){
				// Get next position based on coord_b
				curr_b = nextPos(b_csc, j_idx, coord_b, &progress_pos_b);
				// What is the CAM search cycles (parallel) without linear search (w/o help of register)
				int cam_search_b    = (coord_b == b_csc->idx[curr_b])? (curr_b-start_b) % stride_b : stride_b-1 ;
				// CAM has not found the value; Do a linear search between two CAM pos'
				int linear_search_b = (curr_b - reg_b);
				cycles_b += (1 + std::min(linear_search_b, cam_search_b));
				// If we reached EOS, update coord
				if (curr_b == end_b){
					coord_b = EOS_MSG;
				}
				else{
					coord_b = b_csc->idx[curr_b];
				}
				// send coord to scanner A
				pairq_b.push(std::make_pair(cycles_b,  coord_b));
				// increment pos to next pos
				curr_b++;
				// Update the linear serach helper register
				reg_b = curr_b;
			}
			//printf("%d (%d). %d(%d) - ",coord_a, cycles_a, coord_b, cycles_b);
		}
		//printf("a: %d, b: %d, (%d,%d): %d, - %d, %d\n", end_a-start_a, end_b-start_b, cycles_a, cycles_b, std::max(cycles_a, cycles_b), stride_a, stride_b);
		//exit(1);
	}

	//printf("a: %d, b: %d, (%d,%d): %d, - %d, %d\n", end_a-start_a, end_b-start_b, cycles_a, cycles_b, std::max(cycles_a, cycles_b), stride_a, stride_b);
	//exit(1);

	return std::max(cycles_a, cycles_b);
}

int Matrix::nextPos(CSR_format * a_csr, int i_idx, int coord, int *start_pos){

	int start_a = std::max(*start_pos, a_csr->pos[i_idx]);
	int end_a = std::max(*start_pos, a_csr->pos[i_idx+1]);
	// INT_MAX is eos, early exit
	if (coord == EOS_MSG)
		return end_a;
	// search in the row to find the pos
	for(int i=start_a; i<end_a; i++){
		*start_pos = i;
		if (a_csr->idx[i]>=coord)
			return i;
	}

	return end_a;
}

void Matrix::PrintMatrixSizes(){

	uint64_t a_size_csr = 0, a_size_csf = 0,
		o_size_coo = 0, o_size_csr = 0, o_size_csf = 0;

	for(int i=0; i<a_csr_tiled->row_size; i++){
		for(int j=0; j<a_csr_tiled->col_size; j++){
			a_size_csr += a_csr_tiled->csr_tiles[i][j].csrSize;
			a_size_csf += a_csr_tiled->csr_tiles[i][j].csfSize;
		}
	}
	for(int i=0; i<o_csr_tiled->row_size; i++){
		for(int k=0; k<o_csr_tiled->col_size; k++){
			o_size_csr += o_csr_tiled->csr_tiles[i][k].csrSize;
			o_size_csf += o_csr_tiled->csr_tiles[i][k].csfSize;
			o_size_coo += o_csr_tiled->csr_tiles[i][k].cooSize;
		}
	}

	printf("a csf: %lu\na csr: %lu\no coo: %lu\no csf: %lu\no csr: %lu\n",
			a_size_csf, a_size_csr, o_size_coo, o_size_csf, o_size_csr);
	return;
}


void Matrix::TableDebug(){

	uint64_t cycles=0;
	for(auto it = skipCycles_map.begin(); it != skipCycles_map.end(); ++it){
		cycles+= (uint64_t)it->second;
	}
	printf("Sequential #cycles w/o BW limit: %lu \n", cycles);

	return;
}


void Matrix::InputTilingDebug(){
	std::cout<<"Here!\n";

	// checking if input sizes make sense
	/*
	for(int i_idx = 0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<a_csr_tiled->col_size; j_idx++){
			if ( a_csr_tiled->csr_tiles[i_idx][j_idx].nnz != 0 )
			printf("(%d,%d): %d (nnz), %d(COO), %d(CSR), %d(CSF)\n",i_idx, j_idx, a_csr_tiled->csr_tiles[i_idx][j_idx].nnz, 
					getCOOSize('A', i_idx, j_idx), getCSRSize('A', i_idx, j_idx), getCSFSize('A', i_idx, j_idx));
		}
	}
	*/


	// checking output pre-tiling matches nnz
	/*
	for(int i_idx = 0; i_idx<o_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<o_csr_tiled->col_size; j_idx++){
			printf("(%d,%d): %d\n",i_idx, j_idx, o_csr_tiled->csr_tiles[i_idx][j_idx].nnz);
			for(int t_idx = 0; t_idx<=o_csr_tiled->csr_tiles[i_idx][j_idx].row_size; t_idx++)
				printf("%d ", o_csr_tiled->csr_tiles[i_idx][j_idx].pos[t_idx]);
			printf("\n");
		}
	}
	*/

	// checking if POS matches nnz
	/*
	for(int i_idx = 0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<a_csr_tiled->col_size; j_idx++){
			printf("(%d,%d): %d, %d\n",i_idx, j_idx, a_csr_tiled->csr_tiles[i_idx][j_idx].nnz,  a_csr_tiled->csr_tiles[i_idx][j_idx].pos[128]);
		}
	}
	*/

	// Checking number of nnz in each tile
	// Passed
	/*for(int i_idx = 0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<a_csr_tiled->col_size; j_idx++){
			printf("(%d,%d): %d\n",i_idx, j_idx, a_csr_tiled->csr_tiles[i_idx][j_idx].nnz);
		}
	}
	*/

	// Checking contents of each tile to match with Python code
	// Passed

	/*
	for(int i_idx = 0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<a_csr_tiled->col_size; j_idx++){
			if(a_csr_tiled->csr_tiles[i_idx][j_idx].nnz){
				for(int i = 0; i<a_csr_tiled->csr_tiles[i_idx][j_idx].nnz; i++){
					printf("(%d)\n", a_csr_tiled->csr_tiles[i_idx][j_idx].idx[i]);
				}
			}
		}
	}
	*/
	return;
}

void Matrix::PrintNNZTilesAndFiberHist(){
	int matrix_dim = params->getTileSize();
	uint64_t * nnr_buckets = new uint64_t[matrix_dim+1];
	std::fill(nnr_buckets, nnr_buckets+matrix_dim+1, 0);
	uint64_t total_fibers = a_csr_tiled->row_size * a_csr_tiled->col_size
			* params->getTileSize();

	for(int i_idx=0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx=a_csr_outdims->pos[i_idx];j_idx<a_csr_outdims->pos[i_idx+1]; j_idx++){
			int * matrix_pos = a_csr_tiled->csr_tiles[i_idx][a_csr_outdims->idx[j_idx]].pos;
			for(int t=1; t<= matrix_dim; t++){
				nnr_buckets[matrix_pos[t]-matrix_pos[t-1]]++;
			}
		}
	}
	nnr_buckets[0] = total_fibers - std::accumulate(nnr_buckets+1, nnr_buckets+matrix_dim, 0);

	//printf("nnr distribution report:\n");
	for(int t=0; t<matrix_dim; t++){
		//printf("%d: %lu\n",t, nnr_buckets[t]);
		printf("%lu\n",nnr_buckets[t]);
	}
	delete [] nnr_buckets;
}
/*
void Matrix::PrintTileSizeHist(){

	int matrix_dim = params->getTileSize();
	int max_size = matrix_dim * matrix_dim * params->getDataSize();
	int count_buckets = std::log2(double(max_size));
	uint64_t * size_buckets = new uint64_t[count_buckets+1];
	std::fill(size_buckets, size_buckets+count_buckets+1, 0);
	int nnz_tiles = 0;

	//int single_bucket_size = max_size / (matrix_dim+1);

	for(int i_idx=0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx=a_csr_outdims->pos[i_idx];j_idx<a_csr_outdims->pos[i_idx+1]; j_idx++){
			CSR_format a_tile = a_csr_tiled->csr_tiles[i_idx][a_csr_outdims->idx[j_idx]];
			// ceil(a_tile->csfSize / (matrix_dim+1))
			int bucket_id = (int)std::log2((double)a_tile.csfSize)+1;
			size_buckets[bucket_id]++;
			nnz_tiles++;
		}
	}
	size_buckets[0] = (a_csr_tiled->row_size) * (a_csr_tiled->col_size) - nnz_tiles;

	for(int t=0; t<count_buckets; t++){
		//printf("%d: %lu\n",t, nnr_buckets[t]);
		printf("%lu\n",size_buckets[t]);
	}

	exit(1);
	delete [] size_buckets;
}
*/

void Matrix::PrintTileSizeHist(){

	int matrix_dim = params->getTileSize();
	int bucket_count = matrix_dim*4 + 1;
	uint64_t * size_buckets = new uint64_t[bucket_count];
	int max_size = matrix_dim * matrix_dim * (params->getDataSize() + params->getIdxSize()) + (matrix_dim+3) *8;
	std::fill(size_buckets, size_buckets+bucket_count, 0);
	int nnz_tiles = 0;

	int single_bucket_size = max_size / bucket_count;

	for(int i_idx=0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx=a_csr_outdims->pos[i_idx];j_idx<a_csr_outdims->pos[i_idx+1]; j_idx++){
			CSR_format a_tile = a_csr_tiled->csr_tiles[i_idx][a_csr_outdims->idx[j_idx]];
			// ceil(a_tile->csfSize / (matrix_dim+1))
			if ((a_tile.csfSize + single_bucket_size)/ single_bucket_size > bucket_count)
				printf("problem!%d, %d, %d, %d, %d\n",a_tile.nnz, a_tile.nnr, a_tile.csfSize, bucket_count, (a_tile.csfSize + single_bucket_size)/ single_bucket_size );
			size_buckets[ (a_tile.csfSize + single_bucket_size)/ single_bucket_size]++;
			nnz_tiles++;
		}
	}
	size_buckets[0] = (a_csr_tiled->row_size) * (a_csr_tiled->col_size) - nnz_tiles;

	for(int t=0; t<bucket_count; t++){
		//printf("%d: %lu\n",t, nnr_buckets[t]);
		printf("%lu\n",size_buckets[t]);
	}

	exit(1);
	delete [] size_buckets;
}




void Matrix::PrintNNZTiles(){

	printf("Entered here!\n");
	for(int i_idx=0; i_idx<a_csr_tiled->row_size; i_idx++){
		for(int j_idx=a_csr_outdims->pos[i_idx];j_idx<a_csr_outdims->pos[i_idx+1]; j_idx++){
			printf("(%d, %d) ",i_idx, a_csr_outdims->idx[j_idx]);
		}
		//printf("%d - %d -- ",a_csr_outdims->pos[i_idx], a_csr_outdims->pos[i_idx+1]);
	}

	return;
}

void Matrix::OutputResultDebug(){

	for(int i_idx = 0; i_idx<o_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<o_csr_tiled->col_size; j_idx++){
			printf("(%d,%d): %d , %d bytes\n",i_idx, j_idx, o_csr_tiled->csr_tiles[i_idx][j_idx].nnz, getCOOSize('O', i_idx, j_idx));
		}
	}

	/*
	for(int i_idx = 0; i_idx<o_csr_tiled->row_size; i_idx++){
		for(int j_idx = 0; j_idx<o_csr_tiled->col_size; j_idx++){
			printf("(%d,%d): %d\n",i_idx, j_idx, o_csr_tiled->csr_tiles[i_idx][j_idx].nnz);
		}
	}
	*/
}


// outerSpace related functions

// Rows in the A non-tiled input matrix
int Matrix::GetARowSize(){
	return a_csr->row_size;
}
// Cols in the A non-tiled input matrix
int Matrix::GetAColSize(){
	return a_csr->col_size;
}
// Cols in the B non-tiled input matrix
int Matrix::GetBColSize(){
	return b_csr->col_size;
}

// Return the #non-zero entries in the non-tiled output matrix
uint64_t Matrix::GetNotTiledOutputNNZCount(){
	return o_csr->nnz;
}


// Multiplies a_csr by b_csr and returns the output matrix size;
//	This is a basic MKL based csr matrix multiplication
// DOES THE ACTUAL MULTIPLICATION
uint64_t Matrix::CalculateNotTiledMatrixProduct(){
	float alpha = 1.0;

	sparse_index_base_t indexing;
	MKL_INT rows_d, cols_d;
	MKL_INT *columns_D = NULL, *pointerE_D = NULL, *pointerB_D = NULL;
	double *value_D = NULL;

	sparse_matrix_t csrA = NULL, csrB = NULL, csrD = NULL;

	double  *values_A = NULL, *values_B = NULL;
	MKL_INT *columns_A = NULL, *columns_B = NULL;
	MKL_INT *rowIndex_A = NULL, *rowIndex_B = NULL;

	// Two inputs (A & B), output, and output log tiles
	CSR_format * a_csr = this->a_csr;
	CSR_format * b_csr = this->b_csr;
	CSR_format * o_csr = this->o_csr;

	// if either of them is all zero then just skip; there is no computation to do
	if ((a_csr->nnz == 0) | (b_csr->nnz == 0))
		return 0;

	// Basically the numbers should be aligned for AVX512
	// I tried it in the allocation part and it does not work

	// alligned malloc fixes the problem
	values_A = (double *)mkl_malloc(sizeof(double) * a_csr->nnz, ALIGN);
	columns_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * a_csr->nnz, ALIGN);
	rowIndex_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (a_csr->row_size + 1), ALIGN);

	values_B = (double *)mkl_malloc(sizeof(double) * b_csr->nnz, ALIGN);
	columns_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * b_csr->nnz, ALIGN);
	rowIndex_B = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (b_csr->row_size + 1), ALIGN);

	memcpy(values_A, a_csr->vals, sizeof(double) *a_csr->nnz);
	memcpy(columns_A, a_csr->idx, sizeof(MKL_INT) *a_csr->nnz);
	memcpy(rowIndex_A, a_csr->pos, sizeof(MKL_INT) * (a_csr->row_size+1));

	memcpy(values_B, b_csr->vals, sizeof(double) *b_csr->nnz);
	memcpy(columns_B, b_csr->idx, sizeof(MKL_INT) *b_csr->nnz);
	memcpy(rowIndex_B, b_csr->pos, sizeof(MKL_INT) * (b_csr->row_size+1));

	// convert A and B tiles to MKL readable formats
	mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, a_csr->row_size,
			a_csr->col_size, rowIndex_A, (rowIndex_A)+1,
			columns_A, values_A );
	mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, b_csr->row_size,
			b_csr->col_size, rowIndex_B, (rowIndex_B)+1,
			columns_B, values_B );

	// Do the actual MKL sparse multiplication
	CALL_AND_CHECK_STATUS(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrD),
			"Error after MKL_SPARSE_SPMM \n");
	// Order outputs; MKL does not do it
	CALL_AND_CHECK_STATUS(mkl_sparse_order(csrD), "Error after mkl_sparse_order \n");

	// Export the mult output to see if it has produced any output or all zeros
	mkl_sparse_d_export_csr(csrD, &indexing, &rows_d, &cols_d, &pointerB_D,
			&pointerE_D, &columns_D, &value_D);

	/*
	 * Output extract
	*/
	// update output values
	o_csr->nnz = pointerB_D[rows_d];
	o_csr->row_size = rows_d;
	o_csr->col_size = cols_d;

	// dynamically allocate memory for new psum output
	o_csr->pos = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_csr->row_size+1), ALIGN);
	o_csr->idx = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (o_csr->nnz), ALIGN);
	o_csr->vals = (double *)mkl_malloc(sizeof(double) * (o_csr->nnz), ALIGN);
	// copy data to o_csr
	memcpy(o_csr->pos, pointerB_D, sizeof(MKL_INT) * ((o_csr->row_size)+1));
	memcpy(o_csr->idx, columns_D, sizeof(MKL_INT) * o_csr->nnz);
	memcpy(o_csr->vals, value_D, sizeof(double ) * o_csr->nnz);

	uint64_t output_nnz = pointerB_D[rows_d];

	// We are done. free csrA, csrB, and csrD
	if( mkl_sparse_destroy( csrA ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrA \n");fflush(0);}
	if( mkl_sparse_destroy( csrB ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrB \n");fflush(0);}
	if( mkl_sparse_destroy( csrD ) != SPARSE_STATUS_SUCCESS){
		printf(" Error after MKL_SPARSE_DESTROY, csrD \n");fflush(0);}

	// Free A and B related datastructures
	mkl_free(values_A); mkl_free(columns_A); mkl_free(rowIndex_A);
	mkl_free(values_B); mkl_free(columns_B); mkl_free(rowIndex_B);

	return output_nnz * (params->getDataSize() + 2*params->getIdxSize());
};

// Gets the A column/B row id and calculates the o_traffic and #MACCs
// Please note that this function does not do the actual multiplication
// And also this is for not tiled (no temporal tiling -- Joel's terminology)
void Matrix::OuterProduct(int j_idx, uint32_t &a_traffic_per_pe,
		uint32_t &b_traffic, uint32_t & o_traffic_per_pe, uint32_t & macc_count_per_pe, uint32_t &a_elements){

	a_elements = a_csc->pos[j_idx+1] - a_csc->pos[j_idx];
	int b_elements = b_csr->pos[j_idx+1] - b_csr->pos[j_idx];
	// There will be a unique output element per a_element and b element
	macc_count_per_pe = b_elements;
	// Read traffic for reading A scalar
	a_traffic_per_pe = params->getDataSize() + params->getIdxSize();
	// Read traffic for reading B scalar
	b_traffic = b_elements * (params->getDataSize() + params->getIdxSize())
	 	+ params->getPosSize();
	// Write traffic for output log (later going to read it in the second phase)
	o_traffic_per_pe = macc_count_per_pe * (params->getDataSize() + params->getIdxSize())
		+ params->getPosSize();

	return;
}

//Gets the [i, j , k] and A column/B row id of the specific basic tiles and calculates
// the o_traffic and #MACCs! Please note that this function does not do the actual
// multiplication and also this is for not tiled (no temporal tiling--Joel's terminology)
void Matrix::OuterProduct(int i_idx, int j_idx, int k_idx, int j_micro_tile_idx,
		uint64_t & a_traffic, uint64_t & b_traffic, uint64_t & o_traffic, uint64_t & macc_count){

	CSR_format * a_csc = &(a_csc_tiled->csr_tiles[i_idx][j_idx]);
	CSR_format * b_csr = &(b_csr_tiled->csr_tiles[j_idx][k_idx]);

	int a_elements = a_csc->pos[j_micro_tile_idx+1] - a_csc->pos[j_micro_tile_idx];
	int b_elements = b_csr->pos[j_micro_tile_idx+1] - b_csr->pos[j_micro_tile_idx];

	// There will be a unique output element per a_element and b element
	macc_count = a_elements * b_elements;
	// Read traffic for reading A column
	a_traffic = a_elements * (params->getDataSize() + 2*params->getIdxSize());
	// Read traffic for reading B row
	b_traffic = b_elements * (params->getDataSize() + 2*params->getIdxSize());
	// Write traffic for output log (later going to read it in the second phase)
	o_traffic = macc_count * (params->getDataSize() + 2*params->getIdxSize());

	return;
}

//Gets the [i, j , k] and A row / B column id of the specific basic tiles and calculates
// the o_traffic and #MACCs! Please note that this function does not do the actual
// multiplication and also this is for not tiled (no temporal tiling--Joel's terminology)
//Each PE takes one scalar of A and multiplies to a row of B micro tile
void Matrix::OuterProduct(int i_idx, int j_idx, int k_idx, int j_micro_tile_idx,
		uint64_t & a_traffic_per_pe, uint64_t & b_traffic_per_pe, uint64_t & o_traffic_per_pe,
		uint64_t & macc_count_per_pe, int & a_elements){

	CSR_format * a_csc = &(a_csc_tiled->csr_tiles[i_idx][j_idx]);
	CSR_format * b_csr = &(b_csr_tiled->csr_tiles[j_idx][k_idx]);

	a_elements = a_csc->pos[j_micro_tile_idx+1] - a_csc->pos[j_micro_tile_idx];

	int b_elements = b_csr->pos[j_micro_tile_idx+1] - b_csr->pos[j_micro_tile_idx];

	// There will be a unique output element per a_element and b element
	macc_count_per_pe = b_elements;

	// Read traffic for reading A column; Since OuterSpace uses CR format for A
	a_traffic_per_pe = params->getDataSize() + params->getIdxSize();

	// Read traffic for reading B row; since OuterSpace uses CC format for B
	b_traffic_per_pe = b_elements * (params->getDataSize() + params->getIdxSize())
	 	+ params->getPosSize();

	// Write traffic for output log (later going to read it in the second phase)
	// OuterSpace uses CR format for Ourput and partial products
	o_traffic_per_pe = macc_count_per_pe * (params->getDataSize() + params->getIdxSize())
		+ params->getPosSize();

	return;
}




//Row-wise matrix multiplication
//Similar to the OuterProduct implementations
//input: one microtile ROW of A, and one microtile of B
//Update: This is within a microtile
//        Since we are using MACC units, we can assume that as each
//        output row per scalar is produced, it is held within local registers and/or
//        in the add unit of the MACC unit. 
//        This way, each row of this microtile of A produces a single output microtile row
void Matrix::RowWiseProduct(int i_idx, int j_idx, int k_idx, int i_micro_tile_idx,
		uint64_t & a_traffic, uint64_t & b_traffic, uint64_t & o_traffic, uint64_t & macc_count){

	CSR_format * a_csr = &(a_csr_tiled->csr_tiles[i_idx][j_idx]);
	CSR_format * b_csr = &(b_csr_tiled->csr_tiles[j_idx][k_idx]);
        CSR_format * o_csr = &(o_csr_tiled->csr_tiles[i_idx][k_idx]);

    macc_count = 0;
    a_traffic  = 0;
    b_traffic  = 0;
    o_traffic  = 0;

    //How many elements of A in this row of the microtile?
	int a_elements = a_csr->pos[i_micro_tile_idx+1] - a_csr->pos[i_micro_tile_idx];
    int a_pos_start = a_csr->pos[i_micro_tile_idx];
    int a_pos_end   = a_csr->pos[i_micro_tile_idx+1];
    int o_elements = o_csr->pos[i_micro_tile_idx+1] - o_csr->pos[i_micro_tile_idx];  

        // Write traffic for output log (later going to read it in the second phase)
    /*if (params->getOFormat() == CSX::COO) {
        o_traffic = o_elements * (params->getDataSize() + 2*params->getIdxSize());
    } else { //It's in CSF
        o_traffic = o_elements * (params->getDataSize() + params->getIdxSize())
                     + params->getPosSize();
    }*/ 
    //printf("(%d, %d, %d), %d, o_elements\n", i_idx, j_idx, k_idx, i_micro_tile_idx);
 
    //For each a_element, we process a different B row. Loop through each a element
    for (int a_pos=a_pos_start; a_pos < a_pos_end; a_pos++) {
        int j_micro_tile_idx = a_csr->idx[a_pos];
        //printf("Toluwa DEBUG: In RowWiseProduct -- %d, j_micro_tile_idx is %d\n", i_micro_tile_idx, j_micro_tile_idx);
        //For each a element, we get a j index.
        //Get the number of b elements for this A scalar 
	    int b_elements = b_csr->pos[j_micro_tile_idx+1] - b_csr->pos[j_micro_tile_idx];

        //There is multiplication of this element of a to all the b elements for this 
        //j value, b_elements times 
        macc_count += b_elements; //=> one microtile ROW of O, can't reduce, needs a local buffer

        //read traffic for this element of a to row of B
        if (params->getAFormat() == CSX::COO) {  
            a_traffic += (params->getDataSize() + 2*params->getIdxSize());
	    } else { //It's in CSF
            a_traffic += params->getDataSize() + params->getIdxSize();
        }

        // Read traffic for reading this B row
        if (params->getBFormat() == CSX::COO) {
	        b_traffic += b_elements * (params->getDataSize() + 2*params->getIdxSize());
	    } else { //It's in CSF
            b_traffic += b_elements * (params->getDataSize() + params->getIdxSize())
                         + params->getPosSize();
        }      

    }

    //We brought in this entire row of A, add it to the traffic
    if (params->getAFormat() == CSX::CSF) {
        a_traffic += params->getPosSize();
    }

    // Write traffic for output log (later going to read it in the second phase)
    // We account for this in ScheduleBasicMultTile function 
    /*if (params->getOFormat() == CSX::COO) {
	    o_traffic = o_elements * (params->getDataSize() + 2*params->getIdxSize());
    } else { //It's in CSF
        o_traffic = o_elements * (params->getDataSize() + params->getIdxSize())
                     + params->getPosSize();
    }*/


	return;
}


// Gets a COO format input and converts it to CSR/CSC format
void Matrix::convertCOOToCSR(CSR_format * matrix_csr,
		COO_format * matrix_coo, CSX inp_format){

	int row_size = matrix_coo->row_size;
	int col_size = matrix_coo->col_size;
	int nnz_count = matrix_coo->nnz;

	matrix_csr->row_size= row_size;
	matrix_csr->col_size= col_size;
	matrix_csr->nnz = nnz_count;
	if(inp_format == CSX::CSR)
		matrix_csr->pos = new int[row_size+1];
	else if (inp_format == CSX::CSC)
		matrix_csr->pos = new int[col_size+1];
	else{
		printf("Only CSR and CSC are supported");
		exit(1);
	}
	matrix_csr->idx = new int[nnz_count];
	matrix_csr->vals = new double[nnz_count];

	// Get the CSR/SCS tile of the COO input
	if (inp_format == CSX::CSR){
		coo_tocsr(matrix_coo->row_size, matrix_coo->col_size, matrix_coo->nnz,
			matrix_coo->rows, matrix_coo->cols, matrix_coo->vals,
			matrix_csr->pos, matrix_csr->idx,	matrix_csr->vals);
	}
	else if(inp_format == CSX::CSC){
		coo_tocsc(matrix_coo->row_size, matrix_coo->col_size, matrix_coo->nnz,
			matrix_coo->rows, matrix_coo->cols, matrix_coo->vals,
			matrix_csr->pos, matrix_csr->idx,	matrix_csr->vals);
	}
	else{
		printf("Only CSR and CSC are supported");
		exit(1);
	}

	// Calculate the tile size for different formats
	//	for Inputs this reduces the computation in later steps
	//	although that's super trivial for COO and CSR.
	calculateCSFSize_csr(matrix_csr);
	calculateCSRSize_csr(matrix_csr);
	calculateCOOSize_csr(matrix_csr);

	return;
}

// Gets the A row/O row id and calculates A, B, & O_traffics and #MACCs
// Please note that this function does not do the actual multiplication
void Matrix::GustavsonProduct(int i_idx, uint64_t &a_traffic,
		uint64_t &b_traffic, uint64_t & o_traffic, uint64_t & macc_count){

	a_traffic = 0; b_traffic = 0; o_traffic = 0; macc_count = 0;
	int * a_start = &a_csr->idx[a_csr->pos[i_idx]];
	int * a_end = &a_csr->idx[a_csr->pos[i_idx+1]];
	int a_elements = a_csr->pos[i_idx+1] - a_csr->pos[i_idx];
	int o_elements = o_csr->pos[i_idx+1] - o_csr->pos[i_idx];

    //For each column in this row of A...
	for(int * idx_ptr = a_start; idx_ptr < a_end; idx_ptr++){
        //Get the corresonding row of B
		int j_idx = *idx_ptr;

		int b_elements = b_csr->pos[j_idx+1] - b_csr->pos[j_idx];
		b_traffic += b_elements*(params->getDataSize() + params->getIdxSize()) + params->getPosSize();
    
        //Each time we bring in a row of B, we multiply and add to the previous output product
		macc_count += b_elements;
	}

	// Read traffic for reading A column
	a_traffic = a_elements * (params->getDataSize() + params->getIdxSize()) + params->getPosSize();

	// Write traffic for output log (later going to read it in the second phase)
    //Assuming that we do reduction as soon as we're done multiplying
	if (params->getOFormat() == CSX::COO) {
		o_traffic = o_elements * (params->getDataSize() + 2*params->getIdxSize());
	} else {
		o_traffic = o_elements * (params->getDataSize() + params->getIdxSize()) + params->getPosSize();
	}

    return;
}	

// Taken from https://github.com/scipy/scipy/blob/3b36a57/scipy/sparse/sparsetools/coo.h
/*
 * Compute B = A for COO matrix A, CSR matrix B
 *
 *
 * Input Arguments:
 *   I  n_row      - number of rows in A
 *   I  n_col      - number of columns in A
 *   I  nnz        - number of nonzeros in A
 *   I  Ai[nnz(A)] - row indices
 *   I  Aj[nnz(A)] - column indices
 *   T  Ax[nnz(A)] - nonzeros
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated
 *
 * Note:
 *   Input:  row and column indices *are not* assumed to be ordered
 *
 *   Note: duplicate entries are carried over to the CSR represention
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 *
 */
template <class I, class T>
void Matrix::coo_tocsr(const I n_row,
		const I n_col,
		const I nnz,
		const I Ai[],
		const I Aj[],
		const T Ax[],
		I Bp[],
		I Bj[],
		T Bx[])
{
	//compute number of non-zero entries per row of A
	std::fill(Bp, Bp + n_row, 0);

	for (I n = 0; n < nnz; n++){
		Bp[Ai[n]]++;
	}

	//cumsum the nnz per row to get Bp[]
	for(I i = 0, cumsum = 0; i < n_row; i++){
		I temp = Bp[i];
		Bp[i] = cumsum;
		cumsum += temp;
		//printf("%d ",Bp[i]);
	}
	Bp[n_row] = nnz;
	//printf("%d ",Bp[n_row]);
	//printf("\n");

	//write Aj,Ax into Bj,Bx
	for(I n = 0; n < nnz; n++){
		I row  = Ai[n];
		I dest = Bp[row];

		Bj[dest] = Aj[n];
		Bx[dest] = Ax[n];

		Bp[row]++;
	}

	for(I i = 0, last = 0; i <= n_row; i++){
		I temp = Bp[i];
		Bp[i]  = last;
		last   = temp;
	}

	//now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

	template<class I, class T>
void Matrix::coo_tocsc(const I n_row,
		const I n_col,
		const I nnz,
		const I Ai[],
		const I Aj[],
		const T Ax[],
		I Bp[],
		I Bi[],
		T Bx[])
{ coo_tocsr<I,T>(n_col, n_row, nnz, Aj, Ai, Ax, Bp, Bi, Bx); }

/*
 * Compute B += A for COO matrix A, dense matrix B
 *
 * Input Arguments:
 *   I  n_row           - number of rows in A
 *   I  n_col           - number of columns in A
 *   I  nnz             - number of nonzeros in A
 *   I  Ai[nnz(A)]      - row indices
 *   I  Aj[nnz(A)]      - column indices
 *   T  Ax[nnz(A)]      - nonzeros
 *   T  Bx[n_row*n_col] - dense matrix
 *
 */
	template <class I, class T>
void Matrix::coo_todense(const I n_row,
		const I n_col,
		const I nnz,
		const I Ai[],
		const I Aj[],
		const T Ax[],
		T Bx[],
		int fortran)
{
	if (!fortran) {
		for(I n = 0; n < nnz; n++){
			Bx[ n_col * Ai[n] + Aj[n] ] += Ax[n];
		}
	}
	else {
		for(I n = 0; n < nnz; n++){
			Bx[ n_row * Aj[n] + Ai[n] ] += Ax[n];
		}
	}
}

template <class I>
void Matrix::coo_tocsr_nodata(const I n_row,
		const I n_col,
		const I nnz,
		const I Ai[],
		const I Aj[],
		I Bp[],
		I Bj[])
{
	//compute number of non-zero entries per row of A
	std::fill(Bp, Bp + n_row, 0);

	for (I n = 0; n < nnz; n++){
		Bp[Ai[n]]++;
	}

	//cumsum the nnz per row to get Bp[]
	for(I i = 0, cumsum = 0; i < n_row; i++){
		I temp = Bp[i];
		Bp[i] = cumsum;
		cumsum += temp;
		//printf("%d ",Bp[i]);
	}
	Bp[n_row] = nnz;
	//printf("%d ",Bp[n_row]);
	//printf("\n");

	//write Aj,Ax into Bj,Bx
	for(I n = 0; n < nnz; n++){
		I row  = Ai[n];
		I dest = Bp[row];

		Bj[dest] = Aj[n];
		//Bx[dest] = Ax[n];

		Bp[row]++;
	}

	for(I i = 0, last = 0; i <= n_row; i++){
		I temp = Bp[i];
		Bp[i]  = last;
		last   = temp;
	}

	//now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

	template<class I>
void Matrix::coo_tocsc_nodata(const I n_row,
		const I n_col,
		const I nnz,
		const I Ai[],
		const I Aj[],
		I Bp[],
		I Bi[])
{ coo_tocsr_nodata<I>(n_col, n_row, nnz, Aj, Ai, Bp, Bi); }


#endif
