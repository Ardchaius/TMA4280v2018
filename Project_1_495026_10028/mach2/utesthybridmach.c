#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

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

    int n = 3;
    double local_sum[2] = {0,0};

    MPI_Bcast(&n,1,MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double sum5 = 0;
    double sum239 = 0;
    #pragma omp parallel for reduction (+: sum5, sum239)
    for(int i = rank+1;i<=n;i+=size)
    {

        sum5 += pow(-1,i-1)*pow(0.2,2*i-1)/(2*i-1);
        sum239 += pow(-1,i-1)*pow(239.0,-(2*i-1))/(2*i-1);

    }

    local_sum[0] = sum5;
    local_sum[1] = sum239;

    double global_sum[2] = {0,0};

    MPI_Reduce(&local_sum, &global_sum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double expected_arc5 = 9375.0/46875.0-125.0/46875.0+3.0/46875.0;

    double expected_arc239 = 48942129615.0/11697168977985.0-285605.0/11697168977985.0+3.0/11697168977985.0;

    if(rank == 0){

        printf("(tan(1/5) == Estimate tan(1/5)) =  %d\n", (global_sum[0] == expected_arc5));
        printf("(tan(1/239) == Estimate tan(1/239)) =  %d\n", (global_sum[1] == expected_arc239));

    }


    MPI_Finalize();

    return 0;

}
