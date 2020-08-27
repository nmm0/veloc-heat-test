This project contains the [VeloC sample heat distribution program](https://veloc.readthedocs.io/en/latest/api.html#example).  It is modified to include [Kokkos](https://github.com/kokkos/kokkos/wiki) functionality to measure performance. 

There are 4 different perfomance evaluations.  Two measure VeloC's performance, and two measure Resilient Kokkos' performance.  Each has a version of the heat distribution code that does automatic checkpoints only, and another version that does automatic checkpoints in addition to restarts.

The master script (master.sh) and input configuration file (input.cfg) are what you need to create the batch files that you will use to run the evaluations.  Make sure to set all of the variables in 
input.cfg before running master.sh.  Make sure to also create a results directory for the results.  The batch file will run the filter.py file at the end of execution and dump the results of the experiment in results/average_time.txt.

To build, ensure you have at least cmake/3.14 and then set the flags for the other required (Kokkos, Resilient Kokkos, and VeloC) installations:

Kokkos_DIR - set to the directory of your Kokkos installation containing KokkosConfig.cmake  
resilience_DIR - set to the directory of your Resilient Kokkos installation containing resilienceConfig.cmake or resilience-config.cmake  
VeloC_DIR - set to the install directory of your VeloC installation  


Additionally, set the following:

VELOC_BAREBONE=ON  
CMAKE_CXX_FLAGS=-dynamic  
CMAKE_CXX_FLAGS=-lrt  
