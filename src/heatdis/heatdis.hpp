#ifndef INC_HEATDIS_HEATDIS_HPP
#define INC_HEATDIS_HEATDIS_HPP
 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <Kokkos_Core.hpp>

#define ITER_OUT    50
#define WORKTAG     5
#define REDUCED      1

#ifdef USE_RESILIENT_EXEC
#include <resilience/Resilience.hpp>
#include <resilience/openMP/ResHostSpace.hpp>
#include <resilience/openMP/ResOpenMP.hpp>
#include <resilience/openMP/OpenMPResSubscriber.hpp>
using view_type = Kokkos::View<double*, Kokkos::Experimental::SubscribableViewHooks<
                      KokkosResilience::ResilientDuplicatesSubscriber >>;
using range_policy = Kokkos::RangePolicy< KokkosResilience::ResOpenMP >;
#else
using view_type = Kokkos::View<double*>;
using range_policy = Kokkos::RangePolicy<>;
#endif

namespace heatdis
{
  void initData(int nbLines, int M, int rank, view_type h);

  double doWork(int numprocs, int rank, int M, int nbLines, view_type g, view_type h);
}

#endif  // INC_HEATDIS_HEATDIS_HPP
