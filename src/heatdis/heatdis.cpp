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

    MPI_Request req1[2], req2[2];
    MPI_Status status1[2], status2[2];
    double localerror;
    localerror = 0;

    using range_policy = Kokkos::RangePolicy<>;

    int len = nbLines * M;
    Kokkos::parallel_for("copy_g", range_policy(0, len),
                         KOKKOS_LAMBDA (int i) {
      h(i) = g(i);
    }
    );

    double *g_raw = g.data();
    double *h_raw = h.data();

    /* Send and receive data from the left -- all ranks besides 0 because there are no ranks to its left */
    if (rank > 0) {
      MPI_Isend(g_raw+M, M, MPI_DOUBLE, rank-1, WORKTAG, MPI_COMM_WORLD, &req1[0]);
      MPI_Irecv(h_raw,   M, MPI_DOUBLE, rank-1, WORKTAG, MPI_COMM_WORLD, &req1[1]);
    }

    /* Send and receive data from the right -- all ranks besides numprocs - 1
     * because there are no ranks to its right */
    if (rank < numprocs - 1) {
      MPI_Isend(g_raw+((nbLines-2)*M), M, MPI_DOUBLE, rank+1, WORKTAG, MPI_COMM_WORLD, &req2[0]);
      MPI_Irecv(h_raw+((nbLines-1)*M), M, MPI_DOUBLE, rank+1, WORKTAG, MPI_COMM_WORLD, &req2[1]);
    }

    /* this should probably include ALL ranks 
     * (currently excludes leftmost and rightmost)
     */
    if (rank > 0) {
      MPI_Waitall(2,req1,status1);
    }
    if (rank < numprocs - 1) {
      MPI_Waitall(2,req2,status2);
    }

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
