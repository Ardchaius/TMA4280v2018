# Run mpizeta
# Run the basic implementation of MPI zeta with 8 processes
# and letting you input n
mpizeta: mpizeta.c
	mpicc mpizeta.c -lm -o mpizeta
	mpirun -np 8 ./mpizeta

# Run vectorzeta
# Question 4. We might have possibly misunderstood this excercise
vectorzeta: vector_zeta1.cpp
	g++ vector_zeta1.cpp -o vector_zeta1
	./vector_zeta1

# Run mpizetaplot
# Run the vtest for MPI zeta for several numbers of processes
# Outputs multiple txt files, individually named depending
# On the number of processes.
vtest: plotzeta.c
	mpicc plotzeta.c -lm -o plotzeta
	mpirun -np 2 ./plotzeta
	mpirun -np 4 ./plotzeta
	mpirun -np 8 ./plotzeta
	mpirun -np 16 ./plotzeta
	mpirun -np 32 ./plotzeta
	mpirun -np 64 ./plotzeta

# Run mpizeta utest
# Run utest for MPI zeta for multiple numbers of processes
utest: utestmpizeta.c
	mpicc utestmpizeta.c -lm -o utest
	mpirun -np 2 ./utest
	mpirun -np 4 ./utest
	mpirun -np 8 ./utest
	mpirun -np 16 ./utest
	mpirun -np 32 ./utest
	mpirun -np 64 ./utest
