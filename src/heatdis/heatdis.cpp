#include "heatdis.hpp"

namespace heatdis
{
  void initData(int nbLines, int M, int rank, Kokkos::View<double*> h) {

    using range_policy = Kokkos::RangePolicy<>;

    /* set all of the data to 0 */
    int len = nbLines * M;
    Kokkos::parallel_for("init_h", range_policy (0, len),
                         KOKKOS_LAMBDA (int i) {
      h(i) = 0;
    }
    );

    /* initialize some values on rank 0 to 100 */
    if (rank == 0) {
      int j = ceil(M * 0.9);
      Kokkos::parallel_for("init_rank0", range_policy((int)(M*0.1), j),
                           KOKKOS_LAMBDA (const int i) {
        h(i) = 100;
      }
      );
    }
  }

  double doWork(int numprocs, int rank, int M, int nbLines,
                      Kokkos::View<double*> g, Kokkos::View<double*> h) {
    double localerror;
    localerror = 0;

    using range_policy = Kokkos::RangePolicy<>;

    int len = nbLines * M;
    Kokkos::parallel_for("copy_g", range_policy(0, len),
                         KOKKOS_LAMBDA (int i) {
      h(i) = g(i);
    }
    );

    /* perform the computation */
    //mdrange_policy({1,0}, {nbLines-1, M})
    Kokkos::parallel_for("compute", range_policy(M, len - M),
                         [&localerror, &g, h, M] (int i) {
                           g(i) = 0.25 * (h(i - M) + h(i + M) + h(i-1) + h(i+1));
                           if(localerror < fabs(g(i) - h(i))) {
                             localerror = fabs(g(i) - h(i));
                           }
                         }
    );

    /* perform computation on right-most rank */
    if (rank == (numprocs-1)) {
      Kokkos::parallel_for("compute_right", range_policy(0,M),
                           [&g, nbLines, M] (const int j) {
                             g(((nbLines-1)*M)+j) = g(((nbLines-2)*M)+j);
                           }
      );

    }

    return localerror;
  }
}
