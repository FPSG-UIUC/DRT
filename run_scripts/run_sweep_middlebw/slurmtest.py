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
OUTDIR = os.getcwd() + "/results_middlebw_sweep_asquared"

DATADIR = ""
EXECDIR = "../../src"

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


florida_paths = ["".join([DATADIR+"/floridaMatrices/", x]) 
                    for x in os.listdir(DATADIR+"/floridaMatrices")
                ]
snap_paths    = ["".join([DATADIR+"/SNAP/", x]) 
                    for x in os.listdir(DATADIR+"/SNAP")
                ]

matrix_paths = florida_paths + snap_paths

print(matrix_paths)


#loop through the llb sizes per percentage variant...

for abo_perc in [ [0.01, 0.3, 0.69] ]:
    for middlebw in [2048, 4096, 8192, 16384, 1024]:
        abo_string = "buf_"+str(abo_perc[0])+"_"+str(abo_perc[1])+"_"+str(abo_perc[2])
        llb_string = "llbsize_"+str(llbsize)
        
        for dataset in matrix_paths:
            if dataset.endswith(".mtx"):
                graph = os.path.split(dataset)
                graph = os.path.splitext(graph[1])[0]
                cmd   = EXECDIR + "/SpMSpM_TACTile_twoInp "
                cmd  += "--inp1="+dataset+" " #DATADIR+"/"+graph + ".mtx "
                cmd  += "--inp2="+dataset+" " #DATADIR+"/"+graph + ".mtx "
                cmd  += "--tiledim="+str(tiledim)+" --staticdist="+staticdist+" --intersect="+intersect+" "
                cmd  += "--tiling=dynamic "
                cmd  += "--aperc=" + str(abo_perc[0]) + " "
                cmd  += "--bperc=" + str(abo_perc[1]) + " "
                cmd  += "--operc=" + str(abo_perc[2]) + " "
                cmd  += "--llbpartition=" + llbpartition + " "
                cmd  += "--constreuse=" + str(constreuse) + " "
                cmd  += "--llbsize=" + str(llbsize) + " "
                cmd  += "--middlebw="+ str(middlebw) + " "
        
                test_folder = TEST_LOG_DIR + "/res_tactile_asquared_middlebw_"+graph+"_cfi_rr_"+\
                                             abo_string + "_middlebw_" + str(middlebw)
                cmd += " > " + test_folder + "/res_tactile_asquared_middlebw_"+graph+"_cfi_rr_"+\
                                             abo_string + "_middlebw_" + str(middlebw) + ".txt;"
                #print(cmd)
        
                subprocess.run("mkdir -p " + test_folder, shell=True)
        
                gpu_args.append([test_folder, cmd])

print("*************************************************")
print("We have ", len(gpu_args), "total jobs")
print("*************************************************")

# Save the working directories and slurm command to a csv file
with open(OUTDIR + "/gpu_jobs_synth.csv", mode='w') as gpu_jobs_synth:
    csv_writer = csv.writer(gpu_jobs_synth)
    csv_writer.writerows(gpu_args)

GPU_CHUNKS = 5

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
