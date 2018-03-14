# Run reduc
reduc: reduc.c
	mpicc reduc.c -o reduc -lm
	mpirun -n 2 ./reduc
	mpirun -n 4 ./reduc	
	mpirun -n 8 ./reduc
	mpirun -n 16 ./reduc

# Run mpireduc
mpireduc: mpireduc.c
	mpicc mpireduc.c -o mpireduc -lm
	mpirun -n 2 ./mpireduc
	mpirun -n 4 ./mpireduc	
	mpirun -n 8 ./mpireduc
	mpirun -n 16 ./mpireduc

# Run mpi recursive doubling zeta
zetareduc: mpirecdouzeta.c
	mpicc mpirecdouzeta.c -lm -o mpizetareduc
	mpirun -n 2 ./mpizetareduc
	mpirun -n 4 ./mpizetareduc
	mpirun -n 8 ./mpizetareduc
	mpirun -n 16 ./mpizetareduc

# Run mpi All reduce zeta
allzeta: mpiallzeta.c
	mpicc mpiallzeta.c -lm -o mpiallzeta
	mpirun -n 2 ./mpiallzeta
	mpirun -n 4 ./mpiallzeta
	mpirun -n 8 ./mpiallzeta
	mpirun -n 16 ./mpiallzeta
