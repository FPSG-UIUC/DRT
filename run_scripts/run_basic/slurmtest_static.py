#!/usr/bin/env python3

import os
import os.path
import sys
import subprocess
import socket
import math
import csv
import datetime

#####################################################
### ========== CHANGE THESE FOR SWEEPS ========== ###
#####################################################

# Top-level folder to store outputs in
# NOTE: must use full path
OUTDIR = os.getcwd() + "/results_basic"

DATADIR = ""
EXECDIR = ""


#CPU_ARGS += " --qos=medium"
#CPU_ARGS += " --mem=250G"

###########################################
### ========== RUN THE TESTS ========== ###
###########################################

# Check to make sure we have the spmspm_tactile executable
if not os.path.exists(EXECDIR + "/SpMSpM_TACTile_twoInp"):
    print("Error: SpMSpM_TACTile_twoInp binary not found", file=sys.stderr)
    exit(1)

subprocess.run("mkdir -p " + OUTDIR, shell=True)

gpu_args = []

now = datetime.datetime.now()
print(now.year, now.month, now.day, now.hour, now.minute, now.second)

TEST_LOG_DIR = os.getcwd()+"/output/"
TEST_LOG_DIR += str(now.year)+"_"+str(now.month)
TEST_LOG_DIR += "_"+str(now.day)+"_"+str(now.hour)+"_"
TEST_LOG_DIR += str(now.minute)+"_"+str(now.second)
print("TEST_LOG_DIR is", TEST_LOG_DIR)

# CPU Runs
# Build up a list of working directories and commands
tiledim=32
staticdist="rr"
intersect="parbi" #ideal, skip, parbi
abo_perc=[0.01, 0.3, 0.69]
llbpartition="const" #const, min, avg, prev, ideal
constreuse=128
llbsize=30 #in MB
middlebw=2048 #in GB/s

florida_i_list = [512, 512, 192, 256, 128, 192, 128, 256, 256, 256, 256, 512]
florida_j_list = [8192, 8192, 4096, 1536, 2048, 1536, 1536, 512, 2048, 4096, 1536, 384]
florida_k_list = [256, 192, 128, 32, 64, 128, 128, 64, 128, 128, 32, 128]
florida_matrices = ["mac_econ_fwd500.mtx", "shipsec1.mtx", "pwtk.mtx", 
        "consph.mtx", "cant.mtx", "rma10.mtx", "pdb1HYS.mtx", "bcsstk17.mtx", 
        "cop20k_A.mtx", "scircuit.mtx", "enron.mtx", "mc2depi.mtx"]

snap_i_list = [256, 256, 192, 520000, 32, 192, 256, 256]
snap_j_list = [1024, 1536, 512, 520000, 1536, 2048, 6144, 4096]
snap_k_list = [32, 128, 32, 520000, 32, 32, 128, 192]
snap_matrices = ["sx-mathoverflow.mtx", "cit-HepPh.mtx", "soc-Epinions1.mtx", 
        "p2p-Gnutella31.mtx", "soc-sign-epinions.mtx", 
        "sx-askubuntu.mtx", "email-EuAll.mtx", "amazon0302.mtx"]

florida_paths = ["".join([DATADIR+"/floridaMatrices/", x]) 
                    for x in os.listdir(DATADIR+"/floridaMatrices")
                ]
snap_paths    = ["".join([DATADIR+"/SNAP/", x]) 
                    for x in os.listdir(DATADIR+"/SNAP")
                ]

matrix_paths = florida_paths + snap_paths

print(matrix_paths)


for index, matrix in enumerate(florida_matrices):
    print(index)
    for dataset in matrix_paths:
        if dataset.endswith(".mtx") and matrix in dataset:
            graph = os.path.split(dataset)
            graph = os.path.splitext(graph[1])[0]
            cmd = EXECDIR + "/SpMSpM_TACTile_twoInp_orig "
            cmd  += "--inp1="+dataset+" " #DATADIR+"/"+graph + ".mtx "
            cmd  += "--inp2="+dataset+" " #DATADIR+"/"+graph + ".mtx "
            cmd  += "--tiledim=32"+" --staticdist=rr"+" --intersect=parbi "
            cmd  += "--itop="+str(florida_i_list[index])+" "
            cmd  += "--jtop="+str(florida_j_list[index])+" "
            cmd  += "--ktop="+str(florida_k_list[index])+" "
            cmd  += "--tiling=static "
        
            test_folder = TEST_LOG_DIR + "/res_tactile_suc_asquared_reg_"+graph+"_cfi"+\
                                            "_reg" + str(constreuse)
            cmd += " > " + test_folder + "/res_tactile_suc_asquared_reg_"+graph+"_cfi"+\
                                            "_reg" + str(constreuse) + ".txt;"
            subprocess.run("mkdir -p " + test_folder, shell=True)
        
            gpu_args.append([test_folder, cmd])


for index, matrix in enumerate(snap_matrices):
    for dataset in matrix_paths:
        if dataset.endswith(".mtx") and matrix in dataset:
            graph = os.path.split(dataset)
            graph = os.path.splitext(graph[1])[0]
            cmd = EXECDIR + "/SpMSpM_TACTile_twoInp_orig "
            cmd  += "--inp1="+dataset+" " #DATADIR+"/"+graph + ".mtx "
            cmd  += "--inp2="+dataset+" " #DATADIR+"/"+graph + ".mtx "
            cmd  += "--tiledim=32"+" --staticdist=rr"+" --intersect=parbi "
            cmd  += "--itop="+str(snap_i_list[index])+" "
            cmd  += "--jtop="+str(snap_j_list[index])+" "
            cmd  += "--ktop="+str(snap_k_list[index])+" "
            cmd  += "--tiling=static "
        
            test_folder = TEST_LOG_DIR + "/res_tactile_suc_asquared_reg_"+graph+"_cfi"+\
                                            "_reg" + str(constreuse)
            cmd += " > " + test_folder + "/res_tactile_suc_asquared_reg_"+graph+"_cfi"+\
                                            "_reg" + str(constreuse) + ".txt;"
            subprocess.run("mkdir -p " + test_folder, shell=True)
        
            gpu_args.append([test_folder, cmd])

print("*************************************************")
print("We have ", len(gpu_args), "total jobs")
print("*************************************************")

# Save the working directories and slurm command to a csv file
with open(OUTDIR + "/gpu_jobs_synth.csv", mode='w') as gpu_jobs_synth:
    csv_writer = csv.writer(gpu_jobs_synth)
    csv_writer.writerows(gpu_args)

GPU_CHUNKS = 12

# Split this into a standalone function eventually
gpu_items_per_job = [0] * GPU_CHUNKS
gpu_items_per_job = [x + int(len(gpu_args) / GPU_CHUNKS) for x in gpu_items_per_job]
print("We have gpu_items_per_job as ", gpu_items_per_job)

gpu_chunk_remainder = len(gpu_args) % GPU_CHUNKS
for idx, val in enumerate(gpu_items_per_job):
    if idx < gpu_chunk_remainder:
        gpu_items_per_job[idx] += 1

# Prefix sum operation
gpu_items_per_job_scan = [0] * (GPU_CHUNKS+1)
for i in range(1, len(gpu_items_per_job_scan)):
    gpu_items_per_job_scan[i] += gpu_items_per_job[i-1] + gpu_items_per_job_scan[i-1]

#sbatch_cmd = "sbatch -c 10 --mem=64G --wait ./slurmchunk.py"
sbatch_cmd = "sbatch -c 5 --wait ./slurmchunk.py"

procs = []

for chunk_idx in range(0, GPU_CHUNKS):
    chunk_args = gpu_args[chunk_idx]
    # Creates args in the form (dir, cmd, dir, cmd, ...)
    chunk_cmd = sbatch_cmd + " " + str(gpu_items_per_job_scan[chunk_idx]) + " " + str(gpu_items_per_job_scan[chunk_idx+1]) + " " + OUTDIR + "/gpu_jobs_synth.csv" 
    p = subprocess.Popen(chunk_cmd, shell=True)
    procs.append(p)

for proc in procs:
    proc.wait()
