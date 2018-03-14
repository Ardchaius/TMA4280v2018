#include<stdio.h>
#include<math.h>
#include<stdlib.h>
//Question 3 vtest
#define PI_VALUE 3.141592653589793238462643383279502

int main()
{
 	FILE * fp;
  	fp = fopen ("vtest.txt","w");
	fprintf(fp,"Steps\t\t Error\n");
        fflush(stdout);

	int i,k,n;
	
	
	for(k=1;k<=24;k++){
		n=pow(2,k);
		double pi=0.;
		double error=0.;
		double sum=0.;
		for(i=1;i<=n;i++){ 

		sum+=4*pow(-1.0,(i-1))*1.0/(pow(5,(2*i-1))*(2*i-1))-pow(-1.0,(i-1))*1.0/(pow(239,(2*i-1))*(2*i-1));
		}

		pi=4*sum;
		error=pi-PI_VALUE;
		printf("Pi_n is=%f\t Error=%e\n",pi,error);
		fprintf(fp,"%d\t %e\n",n,error);
		fflush(stdout);
		}

		return 0;
}


