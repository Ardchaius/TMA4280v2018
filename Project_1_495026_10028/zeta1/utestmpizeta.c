#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status stat;

    if( size & (size - 1) ){

        if(rank == 0){
            printf("Please enter a power of 2 for the number of processors.\n");
        }
        MPI_Finalize();
        return 0;

    }

    int n = 3;
    double local_sum = 0;

    MPI_Bcast(&n,1,MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);


    for(int i = rank+1; i<=n; i += size){

            local_sum += 1/(pow(i,2));

    }

    double global_sum;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){

        double pi_estimate_zeta = global_sum;
        printf("(Estimate == Expected) = %d\n", (pi_estimate_zeta == (49.0/36.0)));
        printf("Estimate = %.17g\n", pi_estimate_zeta);
        printf("Expected = %.17g\n", (49.0/36.0));

    }


    MPI_Finalize();

    return 0;

}
