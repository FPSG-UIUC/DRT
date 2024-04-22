MKLROOT=~/intel/compilers_and_libraries_2020.1.217/linux/mkl
#CXX=g++ -std=c++11 -fsanitize=address -Wall -g -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -m64 -I${MKLROOT}/include
#CXX=g++ -std=c++11 -g -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -m64 -I${MKLROOT}/include
CXX=g++ -std=c++11 -O3 -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -m64 -I${MKLROOT}/include
#CXX=g++ -std=c++11 -O3 -pg -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -m64 -I${MKLROOT}/include

CC=g++
#FLAGS=-O3 -fopenmp
LIBS= -fopenmp
CFALGS=-I. -m64 -I${MKLROOT}/include

DEPS=matrix.h mmio.h scheduler_3.h scheduler_4.h scheduler_5.h scheduler_6.h scheduler_7.h llb_mem.h sol_analytical_models.h
OBJ_TEST=matrix.o mmio.o scheduler_3.o scheduler_4.o scheduler_5.o scheduler_6.o scheduler_7.o scheduler_8.o llb_mem.o sol_analytical_models.o test.o
OBJ_SPMM_TACTILE=matrix.o mmio.o scheduler_SpMM_8.o llb_mem.o sol_analytical_models.o SpMM_TACTile.o
OBJ_SPMM_EXTENSOR=matrix.o mmio.o scheduler_SpMM_7.o llb_mem.o sol_analytical_models.o SpMM_ExTensor.o
OBJ_SPMSPM_EXTENSOR=matrix.o mmio.o scheduler_7.o llb_mem.o sol_analytical_models.o SpMSpM_ExTensor.o
OBJ_SPMSPM_TACTILE=matrix.o mmio.o scheduler_8.o llb_mem.o sol_analytical_models.o SpMSpM_TACTile.o
OBJ_SPMSPM_TACTILE_TEST=matrix.o mmio.o scheduler_8.o llb_mem.o sol_analytical_models.o SpMSpM_TACTile.o
OBJ_SPMSPM_TACTILE_TWO_INP=matrix.o mmio.o scheduler_8.o llb_mem.o sol_analytical_models.o SpMSpM_TACTile_twoInp.o
OBJ_SPMSPM_OUTERSPACE=matrix.o mmio.o scheduler_9.o llb_mem.o sol_analytical_models.o SpMSpM_OuterSpace.o
OBJ_SPMSPM_OUTERSPACE_DRT=matrix.o mmio.o scheduler_9_drt.o llb_mem.o sol_analytical_models.o SpMSpM_OuterSpace_drt.o
OBJ_SPMSPM_MATRAPTOR=matrix.o mmio.o scheduler_10.o llb_mem.o sol_analytical_models.o SpMSpM_MatRaptor.o
OBJ_SPMSPM_MATRAPTOR_DRT=matrix.o mmio.o scheduler_10_drt.o llb_mem.o sol_analytical_models.o SpMSpM_MatRaptor_drt.o
OBJ_SPMSPM_TACTILE_DIAG_DRT=matrix.o mmio.o scheduler_8_diagonal_drt.o llb_mem.o sol_analytical_models.o SpMSpM_TACTile_twoInp.o

%.o : %.cpp $(DEPS)
	$(CXX) $(FLAGS) -c -o $@ $< $(CFLAGS) $(LIBS)

%.o : %.c $(DEPS)
	$(CXX) $(FLAGS) -c -o $@ $< $(CFLAGS) $(LIBS)

test: $(OBJ_TEST)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMM_TACTile: $(OBJ_SPMM_TACTILE)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMM_ExTensor: $(OBJ_SPMM_EXTENSOR)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_ExTensor: $(OBJ_SPMSPM_EXTENSOR)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_TACTile: $(OBJ_SPMSPM_TACTILE)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_TACTile_test: $(OBJ_SPMSPM_TACTILE_TEST)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_TACTile_twoInp: $(OBJ_SPMSPM_TACTILE_TWO_INP)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_OuterSpace: $(OBJ_SPMSPM_OUTERSPACE)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_OuterSpace_drt: $(OBJ_SPMSPM_OUTERSPACE_DRT)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_MatRaptor: $(OBJ_SPMSPM_MATRAPTOR)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_MatRaptor_drt: $(OBJ_SPMSPM_MATRAPTOR_DRT)
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

SpMSpM_TACTile_diag_drt: $(OBJ_SPMSPM_TACTILE_DIAG_DRT)
	DIAGONAL_DRT=1
	$(CXX) $(FLAGS) -o $@ $^ $(CFLAGS)

all: test SpMM_ExTensor SpMM_TACTile SpMSpM_ExTensor SpMSpM_TACTile SpMSpM_TACTile_twoInp SpMSpM_OuterSpace SpMSpM_MatRaptor SpMSpM_MatRaptor_drt SpMSpM_TACTile_diag_drt

.PHONY: clean

clean:
	rm -f *.o test SpMM_ExTensor SpMM_TACTile SpMSpM_ExTensor SpMSpM_TACTile SpMSpM_TACTile_twoInp SpMSpM_OuterSpace SpMSpM_MatRaptor_drt
