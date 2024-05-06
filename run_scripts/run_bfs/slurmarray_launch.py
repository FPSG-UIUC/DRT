#!/usr/bin/env python3

import os
import os.path
import sys
import subprocess
import socket
import math
import csv
import datetime
import argparse

#####################################################
### ========== CHANGE THESE FOR SWEEPS ========== ###
#####################################################
##########################
#Read in the parameters
##########################
parser = argparse.ArgumentParser(description='Run the ExTensor Sweep')
parser.add_argument("sweep", help="file containing sweep parameters", type=str)
args = parser.parse_args()

print(args)


# Top-level folder to store outputs in
# NOTE: must use full path
OUTDIR = os.getcwd() + "/results_bfs"

DATADIR = ""
SNAPDIR = DATADIR + "/SNAP"
FLORIDADIR = DATADIR + "/floridaMatrices"
SEED_DIR = ""
EXECDIR = ""

CPU_ARGS = "--mem=50G"


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

########################
#Read in the CSV file
########################
sweep_list = []
with open(args.sweep) as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        sweep_list.append(row)
print(sweep_list)


#########################
#Create a dictionary
#Graph to i, j, k values
#########################
graph_dict = {}

for graph_item in sweep_list:
    graph = graph_item["graph"]
    if graph not in graph_dict:
        graph_dict[graph] = ([],[],[])
         
    graph_dict[graph][0].append(float(graph_item["i"]))
    graph_dict[graph][1].append(float(graph_item["j"]))
    graph_dict[graph][2].append(float(graph_item["k"]))

print(graph_dict)
print(len(graph_dict))


# GPU RUNS:
# Build up a list of working directories and commands
for graph, sizes in graph_dict.items():
    for i in sizes[0]:
        for j in sizes[1]:
            for k in sizes[2]:
                iter_dir = SEED_DIR+"/"+graph
                for seed_file in os.listdir(iter_dir):
                    if ((".mtx" in seed_file ) and ("aspect7" in seed_file)):
                        seed_path = iter_dir +"/" + seed_file
                        seed_name = os.path.splitext(seed_file)[0]

                        #is it in SNAP or FLORIDA
                        snap_mtx = SNAPDIR+"/"+graph + ".mtx"
                        florida_mtx = FLORIDADIR+"/"+graph + ".mtx"
                        inp2_cmd = " "

                        if (graph in snap_mtx):
                            inp2_cmd = "--inp2="+SNAPDIR+"/"+graph + ".mtx "
                        elif (graph in florida_mtx):
                            inp2_cmd = "--inp2="+FLORIDADIR+"/"+graph + ".mtx "
                        else:
                            print("ERROR!", graph, snap_mtx, florida_mtx)
                            exit(1)

                        cmd = EXECDIR + "/SpMSpM_TACTile_twoInp "
                        cmd +=  "--inp1="+seed_path+" "
                        cmd +=  inp2_cmd
                        cmd +=  "--tiledim=32 --staticdist=rr --intersect=parbi "
                        cmd +=  "--itop="+str(i)+" --jtop="+str(j)+" --ktop="+str(k)+" --tiling=static "
                        test_folder = TEST_LOG_DIR + "/res_tactile_suc_aat_"+graph+"_cfi_rr_i"+str(i)+"_j"+str(j)+"_k"+str(k)
                        cmd += " >> " + test_folder + "/res_tactile_suc_aat_"+seed_name+"_cfi_rr_i"+str(i)+"_j"+str(j)+"_k"+str(k)+".txt; "

                        if (not os.path.exists(test_folder)):
                            subprocess.run("mkdir -p " + test_folder, shell=True)

                        gpu_args.append([test_folder, cmd])

print("hello world")
# Save the working directories and slurm command to a csv file
with open(OUTDIR + "/gpu_jobs_synth.csv", mode='w') as gpu_jobs_synth:
    csv_writer = csv.writer(gpu_jobs_synth)
    csv_writer.writerows(gpu_args)

#We now have a series of commands + an output command
#There is a max array size, so lets stay below that per chunk
CHUNK_SIZE = 1000
GPU_CHUNKS = int(len(gpu_args) / CHUNK_SIZE) + 1
print("gpu chunks", GPU_CHUNKS)
gpu_items_per_job = [0] * GPU_CHUNKS
gpu_items_per_job = [x + int(len(gpu_args) / GPU_CHUNKS) for x in gpu_items_per_job]
print("We have gpu_items_per_job as ", gpu_items_per_job, "with this many chunks", GPU_CHUNKS)
print(CHUNK_SIZE, len(gpu_args), int(len(gpu_args) / GPU_CHUNKS), int(len(gpu_args) / CHUNK_SIZE))
#Give each bin an extra item (for the left over jobs)
gpu_chunk_remainder = len(gpu_args) % GPU_CHUNKS
print("chunk remainder is", gpu_chunk_remainder)
for idx, val in enumerate(gpu_items_per_job):
    if idx < gpu_chunk_remainder:
        gpu_items_per_job[idx] += 1

# Prefix sum operation
gpu_items_per_job_scan = [0] * (GPU_CHUNKS+1)
for i in range(1, len(gpu_items_per_job_scan)):
    gpu_items_per_job_scan[i] += gpu_items_per_job[i-1] + gpu_items_per_job_scan[i-1]


procs = []
for chunk_idx in range(0, GPU_CHUNKS):
    chunk_args = gpu_args[chunk_idx]
    array_params = str(gpu_items_per_job_scan[chunk_idx]) + "-" + str(gpu_items_per_job_scan[chunk_idx+1]) 
    end_line = gpu_items_per_job_scan[chunk_idx+1] - gpu_items_per_job_scan[chunk_idx]
    #launch_cmd = "sbatch -p wario -c 10 --array [" + array_params + "]%4 ./slurmarray_job.py "  + OUTDIR + "/gpu_jobs_synth.csv"
    launch_cmd = "sbatch --wait -p wario -c 10 --array [0-" + str(end_line)+"]%4 ./slurmarray_job.py "  +\
           str(gpu_items_per_job_scan[chunk_idx]) + " " + str(gpu_items_per_job_scan[chunk_idx+1]) + " " + OUTDIR + "/gpu_jobs_synth.csv"

    print("Launching this command", launch_cmd)

    #chunk_cmd = sbatch_cmd + " " + str(gpu_items_per_job_scan[chunk_idx]) + " " + str(gpu_items_per_job_scan[chunk_idx+1]) + " " + OUTDIR + "/gpu_jobs_synth.csv" 
    p = subprocess.Popen(launch_cmd, shell=True)
    procs.append(p)

for proc in procs:
    proc.wait()