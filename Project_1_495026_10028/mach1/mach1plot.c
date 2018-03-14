#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<stdlib.h>

#define PI_VALUE 3.141592653589793238462643383279502

int main(int argc, char **argv)
{
  FILE * fp;
  fp = fopen ("plot.dat","w");
  int rank, size;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status stat;
  //Assert for power of 2 processors
  if (size & (size - 1))
  {
    if (rank == 0)
    {
      printf("Please enter a power of 2 for the number of processors.\n");
    }
    MPI_Finalize();
    return 0;
  }
	double timings;
  	double error;
  	int j;
  for (j = 3; j <= 15; j++)
  {
	double time_start;
	if(rank == 0){

            time_start = MPI_Wtime();

        }

    int N = pow(2, j);
    int n = N / size;
    if (n==0) continue;
    if (N % size) { perror("Not a multiple"); }

    double local_sum = 0.;

      int offset = rank * n;
      for (int i = offset + 1; i <= offset + n; i++)
      {
        local_sum += 4 * pow(-1.0, (i - 1)) * 1.0/ (pow(5, (2 * i - 1)) * (2 * i - 1))
            - pow(-1.0, (i - 1)) * 1.0
                / (pow(239, (2 * i - 1)) * (2 * i - 1)); 
        
      }
    double global_sum = 0.;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
          global_sum *= 4.0;
	timings = MPI_Wtime() - time_start;
      printf("Value of Pi=%e   at   N=2^%2d  \t", global_sum, j);
      error = fabs(global_sum - PI_VALUE);
      printf("Error[%d]=%e Time=%e\n", j - 3, error,timings);

      fprintf(fp,"%d\t\t %e\t %e\n",N,error,timings);
      fflush(stdout);
    }
  }
  fclose (fp);
  MPI_Finalize();
  return 0;
}
