#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;

const double pi = 3.14159265358979323846;

int main() {

    int n = 3;

    double sum5 = 0;
    double sum239 = 0;

    #pragma omp parallel for reduction (+: sum5,sum239)
    for(int i = 1; i<=n; i++)
    {

        sum5 += pow(-1,i-1)*pow(0.2,2.0*i-1)/(2.0*i-1);
        sum239 += pow(-1,i-1)*pow(239.0,-(2.0*i-1))/(2.0*i-1);

    }

    double expected_arc5 = 0.197397333333333333333333333333333333333333333333333333333;

    double expected_arc239 = 0.004184076002074727071625204054231315906772178779975694618;

    printf("(tan(1/5) == Estimate tan(1/5)) =  %d\n", (sum5 == expected_arc5));
    printf("(tan(1/239) == Estimate tan(1/239)) =  %d\n", (sum239 == expected_arc239));

    return 0;

}
