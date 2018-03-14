#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;

const double pi = 3.14159265358979323846;

int main(){


    int n = 3;
    double sum = 0;

    #pragma omp parallel for reduction (+: sum)
    for(int i = 1;i<=n;i++){

        sum += 1/(pow(i,2));

    }

    double pi_estimate_zeta = sum;
    printf("(Estimate == Expected) = %d\n", (pi_estimate_zeta == (49.0/36.0)));
    printf("Estimate = %.17g\n", pi_estimate_zeta);
    printf("Expected = %.17g\n", (49.0/36.0));

    return 0;

}
