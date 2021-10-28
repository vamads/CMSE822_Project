Group Members: Vicente Amado Olivo and Yaqi Jie

# Parallelizing Monte Carlo Methods for the TARDIS SN Package
![status](./blabla/banner.png)
## Abstract:
We will be using various parallelization techniques to speed up the Monte Carlo simulations in the TARDIS SN package. The TARDIS SN package is an open-source Monte Carlo radiative-transfer spectral synthesis code for 1D models of exploding stars (supernova ejecta)[1]. The Monte Carlo method tracks the random walk of a photon leaving the supernova ejecta. Both the initialization of the photons and their random walk through the supernova ejecta are modeled through probabilistic processes. In the TARDIS SN package the Monte Carlo method has been parallelized using numba (as TARDIS is written in python), however, Alerstam et. al. found that using GPU programming can speed up the computation substantially in a previous Monte Carlo study [2]. In this project, we will implement a simple Monte Carlo method and compare several techniques in python using jit and CUDA supported by numba[3]. We will develop a benchmark for the TARDIS SN collaborators to use when implementing the more complex radiative-transfer Monte Carlo simulations. 

## Parallelization Strategies: 
We will compare the computation time for using the following different strategies on the same Monte Carlo methods. 
- Develop base Monte Carlo methods without using any speed-up strategies;
- Modify our base code using jit class supported by [Numba](http://numba.pydata.org/numba-doc/latest/index.html), which  is a compiler that gives users the power to speed up array-oriented and math-heavy python code to perform at a similar speed as  C++.
- Implement the same method using [Numba CUDA](http://numba.pydata.org/numba-doc/latest/cuda/), a python wrapper for cuda that supports GPU programming by compiling python code into CUDA kernels.

## Benchmark and optimization: 
We will measure our success by comparing the speed of the Monte Carlo methods. The various parallelization techniques will be timed and compared to see which is the most efficient technique. We have found various references that we will use to measure against our own study. Erik Alerstam used an NVIDIA GeForce 8800GT GPU to run Monte Carlo simulations of photon migration [2]. In this study, Erik found that “The simulation times were 7.9 s for the GPU (a NVIDIA 8800GT) and 8513s for the CPU (an Intel Pentium 4 HT 3.4 GHz), i.e the GPU proves to be 1080X faster” [2]. Rego and Brandao gave a benchmark between pure Python, Cython, and Python with numba in kinetic Monte Carlo study. They found that Cython and Python with numba are both much faster, and Python with numba can reduce 99% of the computational time relative to the pure Python[4].

## If time permits:
Time permitting, we would like to extend the study to more complex radiative transfer models. We would run the same study for more complex astrophysics simulations to see how that may affect the speed, while keeping the parallelization techniques constant. Additionally, we would like to implement various other parallelization techniques, such as: threading, OpenMP, or MPI.

## Resources: 
2080 RTX TI Nvidia GPU and the Moria server (Dr. Wolfgang Kerzendorf’s server)

## Reference:
[1] “Monte Carlo Radiative Transfer - Basic Principles — tardis.” https://tardis-sn.github.io/tardis/physics/montecarlo/basicprinciples.html (accessed Oct. 27, 2021).

[2] Erik Alerstam, Tomas Svensson, Stefan Andersson-Engels, "Parallel computing with graphics processing units for high-speed Monte Carlo simulation of photon migration," J. Biomed. Opt. 13(6) 060504 (1 November 2008) https://doi.org/10.1117/1.3041496

[3] S. K. Lam, A. Pitrou, and S. Seibert, “Numba: a LLVM-based Python JIT compiler,” in Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC - LLVM ’15, Austin, Texas, 2015, pp. 1–6. doi: 10.1145/2833157.2833162.

[4] A. S. C. Rego and A. L. T. Brandão, “General Method for Speeding Up Kinetic Monte Carlo Simulations,” Ind. Eng. Chem. Res., vol. 59, no. 19, pp. 9034–9042, May 2020, doi: 10.1021/acs.iecr.0c01069.