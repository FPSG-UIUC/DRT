#Script to run ExTensor Sweep in parallel
from subprocess import Popen
import subprocess
import argparse
import csv
import threading, queue

##########################
#Read in the parameters
##########################
parser = argparse.ArgumentParser(description='Run the ExTensor Sweep')
parser.add_argument("exe", help="executable", type=str)
parser.add_argument("out_dir", help="output directory", type=str)
parser.add_argument("sweep", help="file containing sweep parameters", type=str)
args = parser.parse_args()

print(args)

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


##########################################
#Time to run ExTensor sweep in parallel!
##########################################
out_list = []
count = 0

#See https://docs.python.org/3/library/queue.html
q = queue.Queue()

def worker():
    while True:
        cmd_item = q.get()
        print("Starting work: ", cmd_item)
        cmd_str = cmd_item[0]
        out_name = cmd_item[1]
        f_h = open(out_name, "w")
        p = Popen(cmd_str, stdout=f_h, stderr=subprocess.STDOUT, shell=True)
        p.wait()


for graph, sizes in graph_dict.items():
    for i in sizes[0]:
        for j in sizes[1]:
            for k in sizes[2]:
                cmd_str = ' '.join(["/bin/bash run_extensor_bfs_ijk.sh", graph, args.exe, str(i), str(j), str(k)])
                print(cmd_str)
                ijk = str(i) + "_" + str(j) + "_" + str(k)
                f_name = args.out_dir+"/res_extensor_"+graph+"_"+ijk+".txt"

                if (f_name not in out_list): #don't run duplicates
                    out_list.append(f_name)
                    q.put([cmd_str, f_name])
                    #f_h = open(f_name, "w")
                    #p = Popen(cmd_str, stdout=f_h, stderr=subprocess.STDOUT, shell=True)
                    #count += 1
                    #print("Count is", count)
                    #print("")

threads = [threading.Thread(target=worker) for _i in range(4)] #We are limited by memory

for thread in threads:
    thread.start()

q.join()
