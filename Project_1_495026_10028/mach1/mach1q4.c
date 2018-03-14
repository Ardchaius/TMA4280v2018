#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<stdlib.h>
#define PI_VALUE 3.141592653589793238462643383279502
int main(int argc,char **argv)
{
int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status stat;
    int N = 16;
    int n = N / size;
    double local_sum = 0.;
	double * v = malloc((rank ? n : N)*sizeof(double));
	if(rank==0){
		for(int i = 1; i <= N; i++){
			v[i-1]=4*pow(-1.0,(i-1))*1.0/(pow(5,(2*i-1))*(2*i-1))-pow(-1.0,(i-1))*1.0/(pow(239,(2*i-1))*(2*i-1));
			}}
	MPI_Scatter(v,n,MPI_DOUBLE, v, n, MPI_DOUBLE, 0,MPI_COMM_WORLD);
	
     for(int i = 0; i < n; i++){ local_sum+= v[i]; }

     double global_sum = 0.;
     MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);
     if (rank == 0) printf("PI_N = %e\nerr = %e\n\n", 4.*global_sum, fabs(4.*global_sum -PI_VALUE));

free(v);
MPI_Finalize();
return 0;
}
