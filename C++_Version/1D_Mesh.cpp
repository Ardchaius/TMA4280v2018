#include <stdio.h>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <iomanip>

using namespace std;

int main(int argc, char** argv) {

    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;

    if(rank == 0){

        printf("Please enter the number of cells: ");
        fflush(stdout);
        cin >> n;

    }

    MPI_Bcast(&n,1,MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> localPositions(n+1,0);
    vector<vector<int> > localTopology(n,vector<int>(2,0));

    double numberingStep = (n+1)/size;
    double meshStep = 1./n;

    int EPP = n/size; // Elements per processor

    if(rank == 0)
        for(int i = 1; i <= (rank+1)*EPP + 1; i++)
            localPositions[i-1] = (i-1)*meshStep;
    else
        for(int i = 1 + rank*EPP; i <= (rank+1)*EPP + 1; i++)
            localPositions[i-1] = (i-1)*meshStep;

    for(int i = rank*EPP + 1; i <= (rank+1)*EPP; i++){

        localTopology[i-1][0] = i;
        localTopology[i-1][1] = i+1;

    }

    for(int i = 0; i < n; i++)
        cout << i+1 << ": " << localTopology[i][0] << " " << localTopology[i][1] << endl;



    MPI_Finalize();

}
