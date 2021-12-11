import numpy as np
import matplotlib.pyplot as plt
import random
import math
from numba import njit, prange, cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from guppy import hpy
import time
import sys
hp = hpy()



def single_boundary_base(n, o_x=0, o_y=0, radius=200, step_limit = 5):
    x=np.zeros(n)
    y=np.zeros(n)
    for i in range(1,n):
        theta=2 * np.pi * random.random()
        step=round(random.uniform(0, step_limit),2)
        x[i] = x[i-1]+ step*np.cos(theta)
        y[i] = y[i-1]+ step*np.sin(theta)
        distance = (x[i] - o_x)**2 + (y[i] - o_y)**2
        if distance > radius ** 2:
            return x[0:i], y[0:i]
    return x, y



def main_base(num_packets, steps):
    packets_x = []
    packets_y = []
    for i in prange(num_packets):
        x,y = single_boundary_base(steps)
        packets_x.append(x)
        packets_y.append(y)
    return packets_x, packets_y



@njit()
def single_boundary_numba(n, o_x=0, o_y=0, radius=200, step_limit = 5):
    x=np.zeros(n)
    y=np.zeros(n)
    for i in range(1,n):
        theta=2 * np.pi * random.random()
        step=round(random.uniform(0, step_limit),2)
        x[i] = x[i-1]+ step*np.cos(theta)
        y[i] = y[i-1]+ step*np.sin(theta)
        distance = (x[i] - o_x)**2 + (y[i] - o_y)**2
        if distance > radius ** 2:
            return x[0:i], y[0:i]
    return x, y



@njit()
def main_numba(num_packets, steps):
    packets_x = []
    packets_y = []
    for i in prange(num_packets):
        x,y = single_boundary_numba(steps)
        packets_x.append(x)
        packets_y.append(y)
    return packets_x, packets_y



@cuda.jit
def single_boundary_cuda(n, num_packets, o_x, o_y, radius, step_limit, packets_x, packets_y, rng_states):
    idx = cuda.grid(1)
    if idx >= num_packets:
        return
    x=packets_x[idx]
    y=packets_y[idx]
    x[0] = 0.0
    y[0] = 0.0

    for i in range(1,n):
        theta = 2 * np.pi * xoroshiro128p_uniform_float32(rng_states, idx)
        step = xoroshiro128p_uniform_float32(rng_states, idx) * step_limit
        x[i] = x[i-1]+ step * math.cos(theta)
        y[i] = y[i-1]+ step * math.sin(theta)
        distance = (x[i] - o_x)**2 + (y[i] - o_y)**2
        
        if distance > radius ** 2:
            break



def main_cuda(num_packets, steps):
    packets_x = np.full((num_packets, steps), -np.Inf, dtype=np.float32)
    packets_y = np.full((num_packets, steps), -np.Inf, dtype=np.float32)
    threads_per_block = 64
    blocks = math.ceil(num_packets / threads_per_block)
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.random())
    single_boundary_cuda[blocks,threads_per_block](steps, num_packets, 0, 0, 200, 5, packets_x, packets_y, rng_states)

    return packets_x, packets_y



def plot_packets(packets_x, packets_y):
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '18'
    for i in range(len(packets_x)):
        x = packets_x[i][packets_x[i] != -np.Inf]
        y = packets_y[i][packets_y[i] != -np.Inf]
        plt.plot(x,y)
        if i == 0:
            plt.plot(x[0],y[0],color='black',marker='*', markersize=10, label="Start point")
            plt.plot(x[-1],y[-1],color='red',marker='o', label = "End point")
        else:
            plt.plot(x[0],y[0],color='black',marker='*', markersize=10)
            plt.plot(x[-1],y[-1],color='red',marker='o')
    plt.legend(loc = 'upper right')
    plt.savefig('result.pdf')
    return



if __name__ == "__main__":
    runtype = sys.argv[1]
    nump = int(sys.argv[2])

    if len(sys.argv) != 3:
        print("Usage: runtype(base/numba/cuda) #packets)")
        exit()

    if runtype.lower() == "base":
        begin = time.time()
        hp.setrelheap()
        x,y = main_base(nump, 10000)
        print(hp.heap())
        end = time.time()
    elif runtype.lower() == "cuda":
        begin = time.time()
        hp.setrelheap()
        x,y = main_cuda(nump, 10000)
        print(hp.heap())
        end = time.time()
    elif runtype.lower() == "numba":
        begin = time.time()
        hp.setrelheap()
        x,y = main_numba(nump, 10000)
        print(hp.heap())
        end = time.time()


    print(f'Total time: {end-begin}')
