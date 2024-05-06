#!/usr/bin/env python3
# Originally written by Jonathan Wapman

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
OUTDIR = os.getcwd() + "/results_aat"

## Update this!!
DATADIR = ""
EXECDIR = ""


CPU_ARGS = " -c 10"
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

# GPU RUNS:
# Build up a list of working directories and commands
for i in [32]:#[8,16,32]:
    for j in [32, 48]:#[24,32,48]:
        for k in [32, 48]:#[24,32,48]:
            for dataset in os.listdir(DATADIR):
                if dataset.endswith(".mtx") and ("transpose" not in dataset):
                    graph = os.path.split(dataset)
                    graph = os.path.splitext(graph[1])[0]
                    cmd = EXECDIR + "/SpMSpM_TACTile_twoInp "
                    cmd +=  "--inp1="+DATADIR+"/"+graph + ".mtx "
                    cmd +=  "--inp2="+DATADIR+"/"+graph + "_transpose.mtx "
                    cmd +=  "--tiledim=32 --staticdist=rr --intersect=parbi "
                    cmd +=  "--itop="+str(i)+" --jtop="+str(j)+" --ktop="+str(k)+" --tiling=static "
                    #cmd += "128 32 0.05 0.45 0.5 cfi rr; "
                    test_folder = TEST_LOG_DIR + "/res_tactile_suc_aat_"+graph+"_cfi_rr_i"+str(i)+"_j"+str(j)+"_k"+str(k)
                    cmd += " > " + test_folder + "/res_tactile_suc_aat_"+graph+"_cfi_rr_i"+str(i)+"_j"+str(j)+"_k"+str(k)+".txt;"

                    subprocess.run("mkdir -p " + test_folder, shell=True)

                    gpu_args.append([test_folder, cmd])

                    cmd += EXECDIR + "/SpMSpM_TACTile_twoInp "
                    cmd +=  "--inp1="+DATADIR+"/"+graph+"_transpose.mtx "
                    cmd +=  "--inp2="+DATADIR+"/"+graph+".mtx "
                    cmd +=  "--tiledim=32 --staticdist=rr --intersect=parbi "
                    cmd +=  "--itop="+str(i)+" --jtop="+str(j)+" --ktop="+str(k)+" --tiling=static; "
                    test_folder = TEST_LOG_DIR + "/res_tactile_suc_aat_"+graph+"_transpose_cfi_rr_i"+str(i)+"_j"+str(j)+"_k"+str(k)
                    cmd += " > " + test_folder + "/res_tactile_suc_aat_"+graph+"_transpose_cfi_rr_i"+str(i)+"_j"+str(j)+"_k"+str(k)+".txt;"

                    dataset_name = os.path.split(dataset)
                    dataset_name = os.path.splitext(dataset_name[1])[0]

                    subprocess.run("mkdir -p " + test_folder, shell=True)

                    gpu_args.append([test_folder, cmd])



# Save the working directories and slurm command to a csv file
with open(OUTDIR + "/gpu_jobs_synth.csv", mode='w') as gpu_jobs_synth:
    csv_writer = csv.writer(gpu_jobs_synth)
    csv_writer.writerows(gpu_args)

GPU_CHUNKS = 10

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
sbatch_cmd = "sbatch -c 10 --wait ./slurmchunk.py"

procs = []

for chunk_idx in range(0, GPU_CHUNKS):
    chunk_args = gpu_args[chunk_idx]
    # Creates args in the form (dir, cmd, dir, cmd, ...)
    chunk_cmd = sbatch_cmd + " " + str(gpu_items_per_job_scan[chunk_idx]) + " " + str(gpu_items_per_job_scan[chunk_idx+1]) + " " + OUTDIR + "/gpu_jobs_synth.csv" 
    p = subprocess.Popen(chunk_cmd, shell=True)
    procs.append(p)

for proc in procs:
    proc.wait()
