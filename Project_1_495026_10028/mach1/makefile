# Run mach1
mach1: mach1q4.c
	mpicc mach1q4.c -o mach1q4 -lm
	mpirun -n 8 ./mach1q4

# Run utest
utest: mach1q5.c
	mpicc mach1q5.c -o mach1q5 -lm
	mpirun -n 2 ./mach1q5
	mpirun -n 4 ./mach1q5	
	mpirun -n 8 ./mach1q5
	mpirun -n 16 ./mach1q5
# Run vtest
vtest: mach1plot.c
	mpicc mach1plot.c -o mach1plot -lm
	mpirun -n 8 ./mach1plot
