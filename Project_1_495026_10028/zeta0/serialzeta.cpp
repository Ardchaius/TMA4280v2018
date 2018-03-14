#include <iostream>
#include <math.h>
#include <iomanip>

using namespace std;

int main()
{

    cout << "Please enter a value for n (Number of terms to be summed): " << endl;
    double n;
    cin >> n;

    double pi_estimate_zeta = 0;

    for(int i = 1; i <=n; i++){

        pi_estimate_zeta += 1/(pow(i,2));

    }

    pi_estimate_zeta = sqrt(pi_estimate_zeta*6);

    cout << "Estimate with Riemann Zeta: " << setprecision(17) << pi_estimate_zeta << endl;

    return 0;

}
