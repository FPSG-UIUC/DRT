#!/usr/bin/env python3
# Todemuyiwa

import os
import sys
import csv
import subprocess

# Get the jobid
start = int(sys.argv[1])
end = int(sys.argv[2])
csv_file = sys.argv[3]
jobid = int(os.getenv('SLURM_ARRAY_TASK_ID')) + start
print("Job id is: ", jobid)

run_dir = os.getcwd()

job_data = []
with open(csv_file) as f:
    reader = csv.reader(f)
    job_data = list(reader)

slurm_dir = job_data[jobid][0]
exec_cmd = job_data[jobid][1]

os.chdir(slurm_dir)
print("In directory ", slurm_dir)

# create a job step
cmd = "srun --mem 25G -c 10 -o " + slurm_dir + "/slurm.log python " + run_dir + "/testitem.py " + "\"" + exec_cmd + "\""
print(cmd)
p = subprocess.Popen(cmd, shell=True)
p.wait()
