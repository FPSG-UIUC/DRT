#!/usr/bin/env python3
import os
import os.path
import sys
import subprocess
import socket
import math
import csv
import datetime
import htcondor
import classad

#####################################################
### ========== CHANGE THESE FOR SWEEPS ========== ###
#####################################################

# Top-level folder to store outputs in
# NOTE: must use full path
OUTDIR = os.getcwd() + "/results_mt_sizes_sweep_asquared"
CONDORFILE = os.getcwd() + "/jobs.condor"
DATADIR = ""
EXECDIR = os.getcwd() + "/../../src"

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
    for tiledim in [32, 4, 8, 16, 64, 128, 256, 512, 1]:
        abo_string = "buf_"+str(abo_perc[0])+"_"+str(abo_perc[1])+"_"+str(abo_perc[2])
        llb_string = "llbsize_"+str(llbsize)
        
        for dataset in matrix_paths:
            if dataset.endswith(".mtx") and "Stanford" not in dataset and "Slash" not in dataset:
                graph = os.path.split(dataset)
                graph = os.path.splitext(graph[1])[0]

                cmd_exec   = EXECDIR + "/SpMSpM_TACTile_twoInp "
                
                cmd_args   = "--inp1="+dataset+" " #DATADIR+"/"+graph + ".mtx "
                cmd_args  += "--inp2="+dataset+" " #DATADIR+"/"+graph + ".mtx "
                cmd_args  += "--tiledim="+str(tiledim)+" --staticdist="+staticdist+" --intersect="+intersect+" "
                cmd_args  += "--tiling=dynamic "
                cmd_args  += "--aperc=" + str(abo_perc[0]) + " "
                cmd_args  += "--bperc=" + str(abo_perc[1]) + " "
                cmd_args  += "--operc=" + str(abo_perc[2]) + " "
                cmd_args  += "--llbpartition=" + llbpartition + " "
                cmd_args  += "--constreuse=" + str(constreuse) + " "
                cmd_args  += "--llbsize=" + str(llbsize) + " "
                cmd_args  += "--middlebw="+ str(middlebw) + " "
        
                test_folder = TEST_LOG_DIR + "/res_tactile_asquared_mt_sweep_"+graph+"_cfi_rr_"+\
                                             abo_string + "_mt" + str(tiledim)
                output_file = test_folder + "/res_tactile_asquared_mt_sweep_"+graph+"_cfi_rr_"+\
                                             abo_string + "_mt" + str(tiledim) + ".txt"
                #print(cmd)
        
                subprocess.run("mkdir -p " + test_folder, shell=True)
        
                gpu_args.append([test_folder, cmd_exec, cmd_args, output_file])

print("*************************************************")
print("We have ", len(gpu_args), "total jobs")
print("*************************************************")

# Save the working directories and slurm command to a csv file
with open(OUTDIR + "/gpu_jobs_synth.csv", mode='w') as gpu_jobs_synth:
    csv_writer = csv.writer(gpu_jobs_synth)
    csv_writer.writerows(gpu_args)


with open(CONDORFILE, "w") as fh:

    #let's launch our jobs!
    schedd = htcondor.Schedd()
    cnt = 0
    for run in gpu_args:
        sweep_job = "" 
    
        if "mt1" in run[2] or "mt2" in run[2]:
            sweep_job = htcondor.Submit({
                "executable"    : str(run[1]),
                "arguments"     : str(run[2]),
                "output"        : str(run[3]),
                "error"         : run[0]+"/condor-$(ProcID).error",
                "log"           : run[0]+"/condor-$(ProcID).log",
                "getenv"        : "true",
                "request_cpus"  : "10",
                "notification"  : "Always",
                "notify_user"   : "",
                "request_memory": "200GB"
            })
    
    
        else:
            sweep_job = htcondor.Submit({
                "executable"    : str(run[1]),
                "arguments"     : str(run[2]),
                "output"        : str(run[3]),
                "error"         : run[0]+"/condor-$(ProcID).error",
                "log"           : run[0]+"/condor-$(ProcID).log",
                "getenv"        : "true",
                "request_cpus"  : "10",
                "notification"  : "Always",
                "notify_user"   : "",
                "request_memory": "50GB"
            })
    
    
        print("Submitted ", sweep_job)
        fh.write(str(sweep_job))
        fh.write("queue\n\n")
    
        cnt = cnt + 1
    

