#!/usr/bin/env python3
#Originally written by Jonathan Wapman

import sys
import os
import subprocess
import csv

# Usage: ./slurmchunk.py start_idx stop_idx csv_file.csv
# Index usage: [start_idx, stop_idx]
start_idx = int(sys.argv[1]) # Inclusive
stop_idx = int(sys.argv[2]) # Exclusive
csv_file = sys.argv[3]

run_dir = os.getcwd()
procs = []

job_data = []
with open(csv_file) as f:
    reader = csv.reader(f)
    job_data = list(reader)

for job_idx in range(start_idx, stop_idx):
    slurm_dir = job_data[job_idx][0]
    slurm_cmd = job_data[job_idx][1]

    os.chdir(slurm_dir)

    print("In directory ", slurm_dir)

    # Only issue the command if a "DONE.log" file is not found
    if not os.path.exists(slurm_dir + "/DONE.log"):
        cmd = "srun -c 5 -o " + slurm_dir + "/slurm.log python " + run_dir + "/testitem.py " + "\"" + slurm_cmd + "\""
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        procs.append(p)

for proc in procs:
    proc.wait()
    # Issue the slurm command from here using srun, detach, and wait
    # May need to use wrapper script
