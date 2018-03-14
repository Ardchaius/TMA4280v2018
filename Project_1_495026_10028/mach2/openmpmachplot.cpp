#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;

const double pi = 3.14159265358979323846;

int main() {

    vector<double> errors(25,0);
    vector<double> timings(25,0);

    FILE * fpMatlab;
    fpMatlab = fopen("matlab_friendly_OMP","w");

    for (int k = 1; k<26;k++){

        int n = pow(2,k);

        double sum5 = 0;
        double sum239 = 0;

        clock_t t;
        t = clock();

        #pragma omp parallel for reduction (+: sum5,sum239)
        for(int i = 1; i<n; i++)
        {

            sum5 += pow(-1,i-1)*pow(0.2,2*i-1)/(2*i-1);
            sum239 += pow(-1,i-1)*pow(239.0,-(2*i-1))/(2*i-1);

        }

        timings[k-1] = ((float)clock()-t)/CLOCKS_PER_SEC;

        errors[k-1] = fabs(pi - (16*sum5 - 4*sum239));

    }

    fprintf(fpMatlab, "Errors: ");
    for(int i = 1;i<26;i++){

        fprintf(fpMatlab,"%.17g,",errors[i-1]);
        fflush(stdout);

    }

    fprintf(fpMatlab, "\nTimings: ");
    for(int i = 1;i<26;i++){

        fprintf(fpMatlab,"%.17g,",timings[i-1]);
        fflush(stdout);

    }

    fclose (fpMatlab);

    return 0;

}
