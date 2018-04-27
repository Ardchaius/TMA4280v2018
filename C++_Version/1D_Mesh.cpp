#include <stdio.h>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <iomanip>
#include <eigen3>

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

//    vector<vector<double> > assemA(2,vector<double>(2,0));
    double assemA[2][2] = {};
    assemA[0][0] = 1./3.;
    assemA[0][1] = 1./6.;
    assemA[1][0] = 1./6.;
    assemA[1][1] = 1./3.;

//    vector<vector<double> > assemM(2,vector<double>(2,0));
    double assemM[2][2] = {};
    assemM[0][0] = 1.;
    assemM[0][1] = -1.;
    assemM[1][0] = -1.;
    assemM[1][1] = 1.;

//    vector<vector<double> > A(n+1, vector<double>(n+1,0));
//    vector<vector<double> > M(n+1, vector<double>(n+1,0));
    double A[n+1][n+1] = {};
    double M[n+1][n+1] = {};

    double globalA[n+1][n+1];
    double globalM[n+1][n+1];

    double jacobian;


    for(int i = rank*EPP; i < (rank+1)*EPP; i++){

        jacobian = abs(localPositions[localTopology[i][1]-1] - localPositions[localTopology[i][0]-1]);

        A[localTopology[i][0]-1][localTopology[i][0]-1] += jacobian * assemA[0][0];
        A[localTopology[i][0]-1][localTopology[i][1]-1] += jacobian * assemA[0][1];
        A[localTopology[i][1]-1][localTopology[i][0]-1] += jacobian * assemA[1][0];
        A[localTopology[i][1]-1][localTopology[i][1]-1] += jacobian * assemA[1][1];

        M[localTopology[i][0]-1][localTopology[i][0]-1] += jacobian * assemM[0][0];
        M[localTopology[i][0]-1][localTopology[i][1]-1] += jacobian * assemM[0][1];
        M[localTopology[i][1]-1][localTopology[i][0]-1] += jacobian * assemM[1][0];
        M[localTopology[i][1]-1][localTopology[i][1]-1] += jacobian * assemM[1][1];

    }

    MPI_Reduce(&A, &globalA, (n+1)*(n+1), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&M, &globalM, (n+1)*(n+1), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0){

        for(int i = 0; i <= n; i++){

            for(int j = 0; j<= n; j++){

                cout << globalM[i][j] << " ";

            }

            cout << endl;

        }

    }

    MPI_Finalize();

}
