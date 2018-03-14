#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;

const double pi = 3.14159265358979323846;

int main() {

    cout << "Please enter a value for n (Number of terms to be summed): " << endl;
    int n;
    cin >> n;

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

    double pi_estimate_mach = 16*sum5 - 4*sum239;
    double duration = ((float)clock()-t)/CLOCKS_PER_SEC;

    printf("Calcualted value of pi with Mach method: %.17g\n", pi_estimate_mach);
    printf("The error with %d steps was: %.17g\n", n, fabs(pi-pi_estimate_mach));
    printf("Time taken for calculation: %.17g seconds.\n", duration);

    return 0;

}
