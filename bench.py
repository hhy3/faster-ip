import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

Ns = [int(1.6**i) for i in range(2, 30)]
dim = 960

os.system("rm -r tmp")
os.system("mkdir tmp")

for N in Ns:
    print(f"N = {N}")
    os.system(
        "g++ -O3 -march=native simd.cc -o simd -std=c++20 -lbenchmark_main -lbenchmark -lpthread")
    os.system(
        f"simd {N} {dim} --benchmark_out=tmp/{N}.json --benchmark_out_format=json")

mp = defaultdict(list)
for N in Ns:
    json_file = f'tmp/{N}.json'
    with open(json_file, 'r') as fp:
        stat = json.load(fp)
        for benchmark in stat['benchmarks']:
            name = benchmark['name']
            cpu_time = benchmark['cpu_time']
            mp[name].append(cpu_time)

l1_kb = 32
l2_kb = 1024
l3_mb = 33

l1_size = l1_kb * 1024
l2_size = l2_kb * 1024
l3_size = l3_mb * 1024 * 1024
n_l1 = l1_size / dim / 4
n_l2 = l2_size / dim / 4
n_l3 = l3_size / dim / 4

ymax = 500
plt.figure(figsize=(16, 10), dpi=120)
for name, times in mp.items():
    plt.plot(Ns, times, label=name)
plt.title(f"{dim}-dimensional vector inner product calculation latency")
plt.xlabel('N')
plt.ylabel('latency(ns)')
plt.xscale('log')
plt.ylim(0, ymax)
plt.vlines(x=n_l3, ymin=0, ymax=ymax, color='black', linestyle='--')
plt.vlines(x=n_l3 * 2, ymin=0, ymax=ymax, color='red', linestyle='--')
plt.text(n_l3 * 1.1, ymax / 2, f"{l3_mb}MB",
         verticalalignment='center', fontsize=10)
plt.vlines(x=n_l2, ymin=0, ymax=ymax, color='black', linestyle='--')
plt.vlines(x=n_l2 * 2, ymin=0, ymax=ymax, color='red', linestyle='--')
plt.text(n_l2 * 1.1, ymax / 2, f"{l2_kb}KB",
         verticalalignment='center', fontsize=10)
plt.vlines(x=n_l1, ymin=0, ymax=ymax, color='black', linestyle='--')
plt.vlines(x=n_l1 * 2, ymin=0, ymax=ymax, color='red', linestyle='--')
plt.text(n_l1 * 1.1, ymax / 2, f"{l1_kb}KB",
         verticalalignment='center', fontsize=10)
plt.legend()
plt.savefig("figures/output.png")
