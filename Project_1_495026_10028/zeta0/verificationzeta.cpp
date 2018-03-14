#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdio.h>

const double pi = 3.14159265358979323846;

using namespace std;

int main()
{

    FILE * fp;
    fp = fopen ("zeta_verification_data.txt","w");
    fprintf(fp,"Steps\t\t Error\n");
    fflush(stdout);

    vector<double> error(25,0);

    for(int k = 1; k<=25; k++){

        int n = pow(2,k);

        double pi_estimate_zeta = 0;

        for(int i = 1; i <=n; i++){

            pi_estimate_zeta += 1/(pow(i,2));

        }

        pi_estimate_zeta = sqrt(pi_estimate_zeta*6);

        error[k-1] = abs(pi - pi_estimate_zeta);

        if(k<24){
            fprintf(fp,"%d\t\t %.17g\n",  n, error[k-1]);
        }
        else{
            fprintf(fp,"%d\t %.17g\n", n, error[k-1]);
        }
        fflush(stdout);


        cout << "n = " << n << ": " << setprecision(17) << error[k-1] << endl;

    }

    return 0;

}
