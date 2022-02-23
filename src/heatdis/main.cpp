#include <cxxopts/cxxopts.hpp>
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
  int nbLines, M;
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

  bool strong, str_ret;

  if (args.count("scale") > 0 && args["scale"].as< std::string >() == "strong") {
    strong = true;
  } else {
    strong = false;
  }

  std::size_t mem_size = args["size"].as< std::size_t >();

  if (mem_size == 0) {
    printf("Wrong memory size! See usage\n");
    exit(3);
  }

  Kokkos::initialize(argc, argv);
  {
  M = (int)sqrt((double)(mem_size * 1024.0 * 1024.0) / (2 * sizeof(double))); // two matrices needed
  nbLines = M + 3;

  Kokkos::View<double*> h_view("h", M * nbLines);
  Kokkos::View<double*> g_view("g", M * nbLines);

  initData(nbLines, M, 0, g_view);

  memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

  // printf("nbProcs: %d\n", nbProcs);
  // printf("size of double: %d\n", sizeof(double));

  printf( "Local data size is %d x %d = %f MB (%lu).\n", M, nbLines, memSize, mem_size );
  printf("Target precision : %f \n", precision);
  printf("Maximum number of iterations : %lu \n", nsteps);

  Kokkos::Timer timer;
  wtime = timer.seconds();
  int i = 0;
  int v = i;
  printf("i: %d --- v: %d\n", i, v);
  if (i < 0) {
    i = 0;
  }

  while(i < nsteps) {

    localerror = doWork(1, 0, M, nbLines, g_view, h_view);

    if (((i % ITER_OUT) == 0)) {
      printf("Step : %d, error = %f\n", i, globalerror);
    }
    globalerror = localerror;

    if (globalerror < precision) {
      printf("PRECISION ERROR\n");
      break;
    }
    i++;
  }
  printf("Execution finished in %lf seconds.\n", timer.seconds() - wtime);

  }
  Kokkos::finalize();
  return 0;
}
