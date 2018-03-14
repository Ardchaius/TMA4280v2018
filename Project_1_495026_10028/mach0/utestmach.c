#include<stdio.h>
#include<math.h>
#include<stdlib.h>
//Question 3 vtest
#define PI_VALUE 3.141592653589793238462643383279502

double mach0(int n)
{

int i;
double error=0.;
double sum,v1[n],v2[n],pi;
for(i=1;i<=n ;i++)
    
{ 
v1[i]=pow(-1.0,(i-1))*1.0/(pow(5,(2*i-1))*(2*i-1));
v2[i]=pow(-1.0,(i-1))*1.0/(pow(239,(2*i-1))*(2*i-1));
sum+=4*v1[i]-v2[i];
}
pi=4*sum;
error=pi-PI_VALUE;
printf("Pi_n is=%f\t Error=%e\n",pi,error);
return 0;
}

int main() 
{ 
	double pi;
    int n=3;  
    pi=mach0(n);

    return (EXIT_SUCCESS);
}

