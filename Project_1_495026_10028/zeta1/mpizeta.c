#include <mpi.h>
#include <stdio.h>
#include <math.h>

const double pi = 3.14159265358979323846;

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

    int n;
    double local_sum = 0;
    double time_start;

    if(rank == 0){

        printf("Please enter a value for n: ");
        fflush(stdout);
        scanf("%d",&n);
        time_start = MPI_Wtime();

    }

    MPI_Bcast(&n,1,MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);


    for(int i = rank+1; i<=n; i += size){

            local_sum += 1/(pow(i,2));

    }

    double global_sum;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){

        global_sum = sqrt(6*global_sum);
        double duration = MPI_Wtime() - time_start;

        printf("Calcualted value of pi with Zeta method on %d processors: %.17g\n", size, global_sum);
        printf("The error with %d processors and %d steps was: %.17g\n", size, n, fabs(pi-global_sum));
        printf("Time taken for calculation: %.17g seconds.\n", duration);

    }


    MPI_Finalize();

    return 0;

}
