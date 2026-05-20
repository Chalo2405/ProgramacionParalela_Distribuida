#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int main(int argc, char *argv[])
{
int i, count; /* points inside the unit quarter circle */
unsigned short xi[3]; /* random number seed */
int samples; /* samples Number of points to generate */
double x,y; /* Coordinates of points */
double pi; /* Estimate of pi */
samples = atoi(argv[1]);
#pragma omp parallel
{
xi[0] = 1; /* These statements set up the random seed */
xi[1] = 1;
xi[2] = omp_get_thread_num();
count = 0;
printf("I am thread %d\n", xi[2]);
#pragma omp for firstprivate(xi) private(x,y) reduction(+:count)
for (i = 0; i < samples; i++)
{
x = erand48(xi);
y = erand48(xi);
if (x*x + y*y <= 1.0) count++;
}
}
pi = 4.0 * (double)count / (double)samples;
printf("Count = %d, Samples = %d, Estimate of pi: %7.5f\n", count, samples, pi);
return 0;
}
