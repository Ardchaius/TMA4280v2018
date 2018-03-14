#include <mpi.h>
#include <stdio.h>
#include <math.h>

const double pi = 3.14159265358979323846;

int main(int argc, char **argv)
{
    int rank, size;

    FILE * fp;
    char buf[0x100];

    fp = fopen ("zeta_MPI_data.txt","w");

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

    if(rank==0){
        snprintf(buf,sizeof(buf), "zeta_MPI_data_%d_procceses.txt", size);
        fp = fopen(buf,"w");
        fprintf(fp,"Processes\t Steps\t\t Error\t\t\t Time\n");
        fflush(stdout);
    }

    for(int i = 1; i<26;i++){

        n = pow(2,i);

        double local_sum = 0;
        double time_start;

        if(rank == 0){

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
            timings[i-1] = MPI_Wtime() - time_start;
            errors[i-1] = fabs(pi-global_sum);
            if(i<20){
                fprintf(fp,"%d\t\t %d\t\t %.17g\t %.17g\n", size, n, errors[i-1], timings[i-1]);
            }
            else{
                fprintf(fp,"%d\t\t %d\t %.17g\t %.17g\n", size, n, errors[i-1], timings[i-1]);
            }
            fflush(stdout);

        }

    }

    MPI_Finalize();

    fclose (fp);

    return 0;

}
