#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

const double pi = 3.14159265358979323846;

int main(int argc, char **argv)
{

    FILE * fpMatlab;
    char buf[0x100];

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

    double timings[25];
    double errors[25];
    double time_start;
    double approx;

    int ompsize;

    #pragma omp parallel
    {

        ompsize = omp_get_num_threads();

    }


    if(rank == 0){

        snprintf(buf,sizeof(buf), "mach_Hybrid_data_%d_Procceses_%d_Threads.txt", size, ompsize);
        fpMatlab = fopen(buf,"w");
        fprintf(fpMatlab,"Processes\t Threads\t Steps\t\t Error\t\t\t Time\n");
        fflush(stdout);

    }

    MPI_Bcast(&n,1,MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 1; i<26;i++){

        if (rank == 0){

            time_start = MPI_Wtime();

        }

        n = pow(2,i);
        double local_sum[2] = {0,0};

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

        if(rank == 0){

            approx = 16*global_sum[0] - 4*global_sum[1];
            timings[i-1] = MPI_Wtime() - time_start;
            errors[i-1] = fabs(pi-approx);

        }

    }

    /*
    Switch the code below with the commented code in order to get a matlab friendly version of the data
    */

    if(rank == 0){

        for(int k = 1;k<26;k++){
            n = pow(2,k);
            if(k<20){
                fprintf(fpMatlab,"%d\t\t %d\t\t %d\t\t %.17g\t %.17g\n", size,ompsize, n, errors[k-1], timings[k-1]);
            }
            else{
                fprintf(fpMatlab,"%d\t\t %d\t\t %d\t %.17g\t %.17g\n", size, ompsize, n, errors[k-1], timings[k-1]);
            }
            fflush(stdout);

        }


    }

//    if(rank == 0){
//
//        fprintf(fpMatlab, "Errors: ");
//        for(int i = 1;i<26;i++){
//
//            fprintf(fpMatlab,"%.17g,",errors[i-1]);
//            fflush(stdout);
//
//        }
//
//        fprintf(fpMatlab, "\nTimings: ");
//        for(int i = 1;i<26;i++){
//
//            fprintf(fpMatlab,"%.17g,",timings[i-1]);
//            fflush(stdout);
//
//        }
//
//    }


    MPI_Finalize();

    return 0;

}
