#include<stdio.h>
#include<math.h>
#include<stdlib.h>
//Question 1 Serial implementation

double mach0(int n)
{

int i;

double sum,v1[n],v2[n],pi;
for(i=1;i<=n ;i++)
    
{ 
v1[i]=pow(-1.0,(i-1))*1.0/(pow(5,(2*i-1))*(2*i-1));
v2[i]=pow(-1.0,(i-1))*1.0/(pow(239,(2*i-1))*(2*i-1));
sum+=4*v1[i]-v2[i];
}
pi=4*sum;
//printf("Value of Pi is=%f",pi);
return pi;
}

int main() 
{ 
    int n=3;  
    printf("%f\n",mach0(n));

    return (EXIT_SUCCESS);
}

