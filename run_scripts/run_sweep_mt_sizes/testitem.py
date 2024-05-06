#!/usr/bin/env python3
#Originally written by Jonathan Wapman
#lowest-level scipt to run slurm jobs

import sys
import subprocess

# Simple wrapper program to run the command given as one or more args
# After finishing the command, creates a "DONE.log" file in the working
# directory, used as a signal to an sbatch script that the
# work in a given subfolder finished successfully

slurm_cmd = sys.argv[1:]

print("The slurm_cmd is ", slurm_cmd)

subprocess.run(slurm_cmd, shell=True)

donefile = open("DONE.log", "w")
print("Opening DONEFILE")
donefile.close()
