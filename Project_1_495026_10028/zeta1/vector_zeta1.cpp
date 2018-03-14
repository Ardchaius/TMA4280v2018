#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>

using namespace std;

int main()
{

    cout << "Please enter a value for n (Number of terms to be summed): " << endl;
    double n;
    cin >> n;

    vector<double> pi_estimate_zeta(n,0);

    for(int i = 1; i <=n; i++){

        pi_estimate_zeta[i-1] = 1/(pow(i,2));

    }

    double pi_estimate_zeta_sum = 0;

    for(int i = 1; i <= n; i++){

        pi_estimate_zeta_sum += pi_estimate_zeta[i-1];

    }

    pi_estimate_zeta_sum = sqrt(pi_estimate_zeta_sum*6);

    cout << "Estimate with Riemann Zeta: " << setprecision(18) << pi_estimate_zeta_sum << endl;

    return 0;

}
