#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;

const double pi = 3.14159265358979323846;

int main(){

    vector<double> errors(25,0);
    vector<double> timings(25,0);

    FILE * fp;
    char buf[0x100];

    int size;

    #pragma omp parallel
    {
        size = omp_get_num_threads();
    }

    snprintf(buf,sizeof(buf), "zeta_OMP_data_%d_threads.txt", size);
    fp = fopen(buf,"w");
    fprintf(fp,"Processes\t Steps\t\t Error\t\t\t Time\n");
    fflush(stdout);


    for (int k = 1; k<26;k++){

        int n = pow(2,k);
        double sum = 0;

        clock_t t;
        t = clock();

        #pragma omp parallel for reduction (+: sum)
        for(int i = 1;i<=n;i++){

            sum += 1/(pow(i,2));

        }

        timings[k-1] = ((float)clock()-t)/CLOCKS_PER_SEC;

        errors[k-1] = fabs(pi - sqrt(6*sum));

        if(k<20){
            fprintf(fp,"%d\t\t %d\t\t %.17g\t %.17g\n", size, n, errors[k-1], timings[k-1]);
        }
        else{
            fprintf(fp,"%d\t\t %d\t %.17g\t %.17g\n", size, n, errors[k-1], timings[k-1]);
        }
        fflush(stdout);

    }



    fclose (fp);

    return 0;



}
