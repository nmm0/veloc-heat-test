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
             ("config", "Config file", cxxopts::value<std::string>())
             ("scale", "Weak or strong scaling", cxxopts::value<std::string>())
             ("views", "Number of Kokkos Views", cxxopts::value<std::size_t>()->default_value("1"))
             ;

  options.parse_positional({"config"});
  auto args = options.parse(argc, argv);

  std::size_t nsteps = args["nsteps"].as< std::size_t >();
  const auto precision = args["precision"].as< double >();
  const auto chk_interval = args["checkpoint-interval"].as< int >();
  const auto num_views =  args["views"].as< std::size_t >();

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t mem_size = args["size"].as< std::size_t >();

  int strong, str_ret;

  std::string scale;
  scale = args["scale"].as< std::string >();
  if (scale == "strong") {
    strong = 1;
  } else {
    strong = 0;
  }

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

  std::vector<Kokkos::View<double*>> MyViews(num_views);
  for( int i = 0 ; i < num_views; ++i )
  {
     Kokkos::resize( MyViews[i], M*nbLines);
     Kokkos::deep_copy(MyViews[i],1.0);
  }


 // initData(nbLines, M, rank, g_view);

  memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

  if (rank == 0) {
     if (!strong) {
        printf("Local data size is %d x %d = %f MB (%lu) %d Views.\n", M, nbLines, memSize, mem_size, num_views);
     } else {
        printf("Local data size is %d x %d = %f MB (%lu) %d Views.\n", M, nbLines, memSize, mem_size / nbProcs,
        num_views);
     }
     printf("Target precision : %f \n", precision);
     printf("Maximum number of iterations : %lu \n", nsteps);
     printf("Array size : %lu\n",  M*nbLines);
  }

  int i;
  VELOC_Mem_protect(0, &i, 1, sizeof(int));
  for( int jj = 0 jj < num_views; ++jj )
  {
     VELOC_Mem_protect(jj+1, MyViews[jj].data(), M * nbLines, sizeof(double));
  }
  wtime = MPI_Wtime();
  int v = VELOC_Restart_test("heatdis", 0);
  if (v > 0) {
    printf("Previous checkpoint found at iteration %d, initiating restart...\n", v);
    // v can be any version, independent of what VELOC_Restart_test is returning
    assert(VELOC_Restart("heatdis", v) == VELOC_SUCCESS);
  } else {
    i = 0;
  }

  while(i < nsteps) {
    // localerror = doWork(nbProcs, rank, M, nbLines, g_view, h_view);

    //if (((i % ITER_OUT) == 0) && (rank == 0)) {
    //  printf("Step : %d, error = %f\n", i, globalerror);
    //}
    //if ((i % REDUCED) == 0) {
    // MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    //}

    //if (globalerror < precision) {
    //  printf("PRECISION ERROR\n");
    //  break;
    //}

    /* Iterate after checkpoint - need to checkpoint at 0 to be consistent
     * with Resilient Kokkos
     */
    // i++;
    
    if (!(i % chk_interval) && (i != nsteps)) {
      assert(VELOC_Checkpoint("heatdis", i) == VELOC_SUCCESS);
      if (rank == 0) {
        printf("checkpoint rank: %d ---- i: %d\n", rank, i);
      }
    }
    i++;
  }
  if (rank == 0)
    printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

  }
  Kokkos::finalize();

  VELOC_Finalize(0); // no clean up
  MPI_Finalize();
  return 0;
}
