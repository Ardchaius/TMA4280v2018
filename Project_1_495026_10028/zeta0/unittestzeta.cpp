#include <iostream>
#include <stdio.h>
#include <math.h>
#include <iomanip>

using namespace std;

int main()
{

    double pi_estimate_zeta = 0;

    int n = 3;

    for(int i = 1; i <=n; i++){

        pi_estimate_zeta += 1/(pow(i,2));

    }

    cout << "(Estimate == Expected) = " << setprecision(17) << (pi_estimate_zeta == (49.0/36.0)) << endl;
    printf("Estimate = %.17g\n", pi_estimate_zeta);
    printf("Expected = %.17g\n", (49.0/36.0));

    return 0;

}
