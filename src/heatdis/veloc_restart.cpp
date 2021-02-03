#include <cxxopts/cxxopts.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <veloc.h>

#include "heatdis.hpp"

using namespace heatdis;

#ifdef HEATDIS_RESILIENCE
std::unique_ptr< KokkosResilience::ContextBase > ctx;
#endif
/*
    This sample application is based on the heat distribution code
    originally developed within the FTI project: github.com/leobago/fti
*/

int main(int argc, char *argv[]) {
  int rank, nbProcs, nbLines, M;
  double wtime, memSize, localerror, globalerror = 1;

  auto options = cxxopts::Options("heatdis", "Sample heat distribution code");
  options.add_options()
             ("s,size", "Problem size", cxxopts::value<std::size_t>()->default_value("100"))
             ("n,nsteps", "Number of timesteps", cxxopts::value<std::size_t>()->default_value("600"))
             ("p,precision", "Min precision", cxxopts::value<double>()->default_value("0.00001"))
             ("c,checkpoint-interval", "Checkpoint interval", cxxopts::value<int>()->default_value("100"))
             ("fail", "Fail iteration or negative for no fail", cxxopts::value<int>()->default_value("301"))
             ("fail-rank", "Rank to fail if failing", cxxopts::value<int>()->default_value("0"))
             ("config", "Config file", cxxopts::value<std::string>())
             ("scale", "Weak or strong scaling", cxxopts::value<std::string>())
             ;

  options.parse_positional({"config"});
  auto args = options.parse(argc, argv);

  std::size_t nsteps = args["nsteps"].as< std::size_t >();
  const auto precision = args["precision"].as< double >();
  const auto chk_interval = args["checkpoint-interval"].as< int >();
  const int fail_iter = args["fail"].as< int >();
  const int fail_rank = args["fail-rank"].as< int >();

  int strong, str_ret;

  std::string scale;
  scale = args["scale"].as< std::string >();
  if (scale == "strong") {
    strong = 1;
  } else {
    strong = 0;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t mem_size = args["size"].as< std::size_t >();

  if (mem_size == 0) {
    printf("Wrong memory size! See usage\n");
    exit(3);
  }
  if (VELOC_Init(MPI_COMM_WORLD, args["config"].as< std::string >().c_str()) != VELOC_SUCCESS) {
    printf("Error initializing VELOC! Aborting...\n");
    exit(2);
  }

  Kokkos::initialize(argc, argv);
  {
  
  if (!strong) {

    /* weak scaling */
    M = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double))); // two matrices needed
    nbLines = (M / nbProcs) + 3;
  } else {

    /* strong scaling */
    M = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double) * nbProcs)); // two matrices needed
    nbLines = (M / nbProcs) + 3;
  }

  Kokkos::View<double*> h_view("h", M * nbLines);
  Kokkos::View<double*> g_view("g", M * nbLines);

  initData(nbLines, M, rank, g_view);

  memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

  if (rank == 0)
    if (!strong) {
      printf("Local data size is %d x %d = %f MB (%lu).\n", M, nbLines, memSize, mem_size);
    } else {
      printf("Local data size is %d x %d = %f MB (%lu).\n", M, nbLines, memSize, mem_size / nbProcs);
    }
  if (rank == 0)
    printf("Target precision : %f \n", precision);
  if (rank == 0)
    printf("Maximum number of iterations : %lu \n", nsteps);

  int i;
  VELOC_Mem_protect(0, &i, 1, sizeof(int));
  VELOC_Mem_protect(1, h_view.data(), M * nbLines, sizeof(double));
  VELOC_Mem_protect(2, g_view.data(), M * nbLines, sizeof(double));

  wtime = MPI_Wtime();
  int v = VELOC_Restart_test("heatdis", 0);
  if (v > 0) {
    printf("Previous checkpoint found at iteration %d, initiating restart...\n", v);
    // v can be any version, independent of what VELOC_Restart_test is returning
    assert(VELOC_Restart("heatdis", v) == VELOC_SUCCESS);

    /* Added by Carson on 5/6/20 to test VeloC Restart */
    i++;
    //printf("Test v - %d\n", v);

  } else {
    i = 0;
  } 

  /* May need to be added to be consistent with Resilient Kokkos */
  //i++;
  if (rank == 0) {
    //printf("Initial i value: %d\n", i);
  }

  while(i < nsteps) {
    if (rank == 0) {
      //printf("Loop i value: %d\n", i);
    }
    localerror = doWork(nbProcs, rank, M, nbLines, g_view, h_view);

    if (((i % ITER_OUT) == 0) && (rank == 0)) {
      printf("Step : %d, error = %f\n", i, globalerror);
    }
    if ((i % REDUCED) == 0) {
      MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (globalerror < precision) {
      printf("PRECISION ERROR\n");
      break;
    }

    /* Iterate after checkpoint - need to checkpoint at 0 to be consistent
     * with Resilient Kokkos
     */
    //i++;

    if (!(i % chk_interval) && (i != nsteps) && (i != v)) {
      assert(VELOC_Checkpoint("heatdis", i) == VELOC_SUCCESS);
      if (rank == 0) {
        printf("checkpoint rank: %d ---- i: %d\n", rank, i);
      }
    }
    i++;

    /* Add code to kill MPI processes */
    // if (rank == 0 && i == 301 && v <= 0) {
    if ((fail_iter >= 0 ) && (rank == fail_rank) && (i == fail_iter) && (v <= 0)) {
      printf("Killing rank %d at i == %d. Program should terminate.\n", rank, i);
      MPI_Abort(MPI_COMM_WORLD, 400);
    }
  }
  if (rank == 0)
    printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

  }
  Kokkos::finalize();

  VELOC_Finalize(0); // no clean up
  MPI_Finalize();
  return 0;
}
