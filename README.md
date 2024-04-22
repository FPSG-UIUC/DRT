# DRT
Simulators for dynamic reflexive tiling and several accelerators.

Setup:
1. In `src/makefile` set the variable `MKLROOT` to your MKL library location
2. `src/makefile` contains the various targets. The relevant ones are:
    - SpMSpM_ExTensor: executable for the ExTensor simulation
    - SpMSpM_TACTile_twoInp: executable for the TACTile simulation
        - has options for static tiling and DRT tiling
    - SpMSpM_OuterSpace: executable for the idealized OuterSPACE untiled baseline
    - SpMSpM_OuterSpace_drt: executable for the idealized OuterSPACE baseline + DRT (with SUC options)
    - SpMSpM_MatRaptor: executable for the idealized MatRaptor untiled baseline
    - SpMSpM_MatRaptor_drt: executable for the idealized MatRaptor baseline + DRT (with SUC options)
3. To run the TACTile simulation (this assumes A is an IxJ matrix, B is a JxK matrix, and C is an IxK matrix.)
    ```src/SpMSpM_TACTile_twoInp -inp1=<path to first matrix> -inp2=<path to second matrix> \
     --tiledim=<micro tile dimension> \
     --staticdist=<rr|nnz|oracle> \ # static distributor for the PEs 
     --intersect=<skip|parbi> \ #skip --> ExTensor skip-based intersection, parbi --> TACTile cfi/parallel intersection unit \
	 --tiling=<static|dynamic> \
	 --aperc=0.05  # LLB percentage assigned for tensor a\
	 --bperc=0.35  # LLB percentage assigned for tensor b \
	 --operc=0.6   # LLB percentage assigned for tensor o \
     --llbsize=30  # LLB total size (in MB)\
	 --constreuse=128 # # of micro tiles in a macro tile along the K dimension \
	 --topbw=68.25   # Top DOT (DRAM) bandwidth in GB/s \
	 --middlebw=2048 # Middle DOT NoC Bandwidth in GB/s \
	 --itop= # num. of microtiles in a macrotile along the I dimension for SUC tiling \
	 --jtop= # num. of microtiles in a macrotile along the J dimension for SUC tiling \
	 --ktop= # num. of microtiles in a macrotile along the K dimension for SUC tiling
    ```
    - An example command run: 
        - `src/SpMSpM_TACTile_twoInp --inp1=./data/test_amazon0302.mtx --inp2=./data/test_amazon0302.mtx --tiledim=32 --staticdist=rr --intersect=parbi --tiling=dynamic | tee out.txt`
        - 30MB is the default LLB size
        - llb default partitioning: A = 5%, B=50%, C=45%

4. To run the ExTensor simulation:

5. To run the idealized OuterSPACE simulation:

6. To run the idealized MatRaptor simulation:

7. If you are using slurm, you can use the python scripts in the `run_...` directories to launch similar runs as those found in the paper. Please replace the DATADIR, OUTDIR, and EXECDIR with your local file paths (to your dataset directory of mtx files, the output directory where you want results, and the directory of your executable, respectively)

*** 
The simulator spits out text similar to the following (we only include relevant output below):
```
runtime: 0.002372, cycles: 2371708, busy_cycles: 110825158
Top DOT NoC:
 total_top: 0.159724 GBs, a_top: 53040696, b_top: 30309984, o_r_top: 0, o_w_top: 88151644
Middle DOT NoC:
 total_mid: 0.254072 GBs, a_mid: 53040696, b_mid: 134560180, o_r_mid: 0, o_w_mid: 85207216
...
A_csf: 30309984, B_csf: 30309984, O_csf: 88151644, O_COO: 62339776, bandwidth: 68.250000 GB/s
```
- runtime: 0.002372, cycles: 2371708, busy_cycles: 110825158 
    - runtime in seconds of the accelerator on that workload 
    - cycles: # of cycles it would take the accelerator to execute 
    - busy cycles: # of total cycles across all PEs (that is, the cycles of each PE added up) 
- total_top: total number of GB transferred to and from the DRAM (in GB)
    - a_top: total number of bytes transferred for the $A$ matrix  (in DRAM)
    - b_top: total number of bytes transferred for the $B$ matrix (in DRAM)
    - o_r_top: total number of output tensor bytes read from the DRAM
    - o_w_top: total number of output tensor bytes written to the DRAM
- total_mid: total number of GB transferred to and from the LLB (in GB)
    - a_mid: total number of bytes transferred for the $A$ matrix (in LLB)
    - b_mid: total number of bytes transferred for the $B$ matrix (in LLB)
    - o_r_mid: total number of bytes read in the LLB for the output tensor
    - o_w_mid: total number of bytes written in the LLB for the output tensor
- A_csf, B_csf, O_csf, O_COO: the size, in bytes, of each of the tensors for that particular data format. 
- bandwidth: the assumed DRAM bandwidth in GB/s
***

`src`:
This contains the source files for the simulator.
- `scheduler_7.cpp`: the main code for ExTensor modeling
- `scheduler_8.cpp`: the main code for TACTile modeling
- `scheduler_8_diagonal_drt.cpp`: source code for a diagonal variant of DRT in TACTile
- `scheduler_9.cpp`: idealized model of OuterSPACE (no tiling)
- `scheduler_9_drt.cpp`: idealized model of OuterSPACE + DRT
- `llb_mem.cpp`: tracking memory transfers to and form the LLB
