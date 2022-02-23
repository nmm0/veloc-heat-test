#include <cxxopts/cxxopts.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>

#include "heatdis.hpp"
//#include <resilience/CheckpointFilter.hpp>

using namespace heatdis;

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
             ("fail", "Fail iteration or negative for no fail", cxxopts::value<int>()->default_value("-1"))
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

  bool strong, str_ret;

  if (args.count("scale") > 0 && args["scale"].as< std::string >() == "strong") {
    strong = true;
  } else {
    strong = false;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t mem_size = args["size"].as< std::size_t >();

  if (mem_size == 0) {
    printf("Wrong memory size! See usage\n");
    exit(3);
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

  // printf("nbProcs: %d\n", nbProcs);
  // printf("size of double: %d\n", sizeof(double));

  if (rank == 0) {
    if ( !strong ) {
      printf( "Local data size is %d x %d = %f MB (%lu).\n", M, nbLines, memSize, mem_size );
    } else {
      printf( "Local data size is %d x %d = %f MB (%lu).\n", M, nbLines, memSize, mem_size / nbProcs );
    }
  }
  if (rank == 0)
    printf("Target precision : %f \n", precision);
  if (rank == 0)
    printf("Maximum number of iterations : %lu \n", nsteps);

  wtime = MPI_Wtime();
  int i = 0;
  int v = i;
  printf("i: %d --- v: %d\n", i, v);
  if (i < 0) {
    i = 0;
  }

  while(i < nsteps) {

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
    i++;

    // if (rank == 0 && i == 301 && v < 0) {
    if ((fail_iter >= 0 ) && (rank == fail_rank) && (i == fail_iter) && (v < 0)) {
      printf("Killing rank %d at i == %d.\n", rank, i);
      MPI_Abort(MPI_COMM_WORLD, 400);
    }
  }
  if (rank == 0)
    printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

  }
  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
