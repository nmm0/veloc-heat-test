#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "heatdis.hpp"
//#include "include/veloc.h"

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <Kokkos_Core.hpp>
#include <veloc.h>

// this examples uses asserts so they need to be activated
#undef NDEBUG
#include <assert.h>

/*
    This sample application is based on the heat distribution code
    originally developed within the FTI project: github.com/leobago/fti
*/

static const unsigned int CKPT_FREQ = ITER_TIMES / 3;

void initKokkosData(int nbLines, int M, int rank, Kokkos::View<double*> h) {

    typedef Kokkos::RangePolicy<> range_policy;    
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;

    /* set all of the data to 0 */
    Kokkos::parallel_for("init_h", mdrange_policy({0,0}, {nbLines, M}), 
                         KOKKOS_LAMBDA (const int i, const int j) {
            h((i*M)+j) = 0;
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

double doKokkosWork(int numprocs, int rank, int M, int nbLines, 
                    Kokkos::View<double*> g, Kokkos::View<double*> h) {

    MPI_Request req1[2], req2[2];
    MPI_Status status1[2], status2[2];
    double localerror;
    localerror = 0;

    typedef Kokkos::RangePolicy<> range_policy;    
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;

    Kokkos::parallel_for("copy_g", mdrange_policy({0,0}, {nbLines, M}), 
                         KOKKOS_LAMBDA (const int i, const int j) {
            h((i*M)+j) = g((i*M)+j);  
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

    if (rank > 0) {
        MPI_Waitall(2,req1,status1);
    }
    if (rank < numprocs - 1) {
        MPI_Waitall(2,req2,status2);
    }

    /* perform the computation */
    Kokkos::parallel_for("compute", mdrange_policy({1,0}, {nbLines-1, M}),
                         [&localerror, &g, h, M] (const int i, const int j) {
            g((i*M)+j) = 0.25*(h(((i-1)*M)+j)+h(((i+1)*M)+j)+h((i*M)+j-1)+h((i*M)+j+1));
            if(localerror < fabs(g((i*M)+j) - h((i*M)+j))) {
                localerror = fabs(g((i*M)+j) - h((i*M)+j));
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

int main(int argc, char *argv[]) {
    int rank, nbProcs, nbLines, i, M, arg;
    double wtime, memSize, localerror, globalerror = 1;

    if (argc < 3) {
	printf("Usage: %s <mem_in_mb> <cfg_file>\n", argv[0]);
	exit(1);
    }

    MPI_Init(&argc, &argv);	
    MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (sscanf(argv[1], "%d", &arg) != 1) {
        printf("Wrong memory size! See usage\n");
	exit(3);
    }
    if (VELOC_Init(MPI_COMM_WORLD, argv[2]) != VELOC_SUCCESS) {
	printf("Error initializing VELOC! Aborting...\n");
	exit(2);
    }

    /* Initialize Kokkos - does it matter whether it's before or after VeloC initialization? */
    Kokkos::initialize(argc, argv);
    {

    M = (int)sqrt((double)(arg * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double))); // two matrices needed
    nbLines = (M / nbProcs) + 3;

    Kokkos::View<double*> h_view("h", M * nbLines);
    Kokkos::View<double*> g_view("g", M * nbLines);

    initKokkosData(nbLines, M, rank, g_view);

    memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

    if (rank == 0)
	printf("Local data size is %d x %d = %f MB (%d).\n", M, nbLines, memSize, arg);
    if (rank == 0)
	printf("Target precision : %f \n", PRECISION);
    if (rank == 0)
	printf("Maximum number of iterations : %d \n", ITER_TIMES);

    VELOC_Mem_protect(0, &i, 1, sizeof(int));
    VELOC_Mem_protect(1, h_view.data(), M * nbLines, sizeof(double));
    VELOC_Mem_protect(2, g_view.data(), M * nbLines, sizeof(double));

    wtime = MPI_Wtime();
    int v = VELOC_Restart_test("heatdis", 0);
    if (v > 0) {
	printf("Previous checkpoint found at iteration %d, initiating restart...\n", v);
	// v can be any version, independent of what VELOC_Restart_test is returning
	assert(VELOC_Restart("heatdis", v) == VELOC_SUCCESS);
    } else
	i = 0;
    while(i < ITER_TIMES) {
        localerror = doKokkosWork(nbProcs, rank, M, nbLines, g_view, h_view);
        if (((i % ITER_OUT) == 0) && (rank == 0))
	    printf("Step : %d, error = %f\n", i, globalerror);
        if ((i % REDUCED) == 0)
	    MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (globalerror < PRECISION)
	    break;
	i++;
	if (i % CKPT_FREQ == 0 && i != ITER_TIMES) {
	    assert(VELOC_Checkpoint("heatdis", i) == VELOC_SUCCESS);
            if (rank == 0) {
               printf("checkpoint rank: %d ---- i: %d\n", rank, i);
            }
        }
    }
    if (rank == 0)
	printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

    /* call to finalize Kokkos */
    }
    Kokkos::finalize();

    VELOC_Finalize(0); // no clean up
    MPI_Finalize();
    return 0;
}
