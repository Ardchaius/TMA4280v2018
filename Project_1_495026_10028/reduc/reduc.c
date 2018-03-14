#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<stdlib.h>

#define PI_VALUE 3.141592653589793238462643383279502

int main(int argc, char **argv)
{
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
  	double time_start;
	if(rank == 0){

            time_start = MPI_Wtime();
        }

    int N = 16;
    int n = N / size;
    
    if (N % size) { perror("Not a multiple"); }

    double local_sum = 0.;

      int offset = rank * n;
      for (int i = offset + 1; i <= offset + n; i++)
      {
        local_sum += 4 * pow(-1.0, (i - 1)) * 1.0/ (pow(5, (2 * i - 1)) * (2 * i - 1))
            - pow(-1.0, (i - 1)) * 1.0
                / (pow(239, (2 * i - 1)) * (2 * i - 1));   
      }
    double sigma = local_sum;
    double sigmaq;

    for(int d = 0; d <=log2(size) - 1; d++){
        int q = rank^((int)pow(2,d));
        MPI_Send(&sigma, 1, MPI_DOUBLE, q, 1, MPI_COMM_WORLD);
        MPI_Recv(&sigmaq, 1, MPI_DOUBLE, q, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sigma += sigmaq;
    }

    double global_sum=0.; 

    global_sum = 4*sigma;

	timings = MPI_Wtime() - time_start;
	
    	error = fabs(global_sum - PI_VALUE);
      
     	printf("Process:%d\t Value of Pi:%e\t Error:%e\t Time:%e\n", rank, global_sum,error,timings);
      
  MPI_Finalize();
  return 0;
}
