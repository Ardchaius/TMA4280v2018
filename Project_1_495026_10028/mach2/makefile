# Run hybridmach
# Runs the basic hybrid mach method letting you input an n
hybridmach: hybridmach.c
	mpicc hybridmach.c -lm -o hybridmach -fopenmp
	OMP_NUM_THREADS=8 mpirun -np 8 ./hybridmach

# Run hybridmachplot
# Runs the vtest for the hybrid mach method outputting
# multiple files of data on the method
hybridmachplot: hybridmachplot.c
	mpicc hybridmachplot.c -lm -o hybridmachplot -fopenmp
	OMP_NUM_THREADS=2 mpirun -np 2 ./hybridmachplot
	OMP_NUM_THREADS=2 mpirun -np 8 ./hybridmachplot
	OMP_NUM_THREADS=4 mpirun -np 2 ./hybridmachplot
	OMP_NUM_THREADS=4 mpirun -np 8 ./hybridmachplot
	OMP_NUM_THREADS=8 mpirun -np 2 ./hybridmachplot
	OMP_NUM_THREADS=8 mpirun -np 8 ./hybridmachplot

# Run Hybrid mach utest
# Runs the utest for the hybrid mach method
utesthybrid: utesthybridmach.c
	mpicc utesthybridmach.c -lm -o utesthybrid -fopenmp
	OMP_NUM_THREADS=2 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=2 mpirun -np 8 ./utesthybrid
	OMP_NUM_THREADS=4 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=4 mpirun -np 8 ./utesthybrid
	OMP_NUM_THREADS=8 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=8 mpirun -np 8 ./utesthybrid

# Run OMP mach utest
# Runs the utest for the OMP mach method
utestomp: utestopenmpmach.cpp
	g++ utestopenmpmach.cpp -lm -o utestomp -fopenmp
	OMP_NUM_THREADS=2 ./utestomp
	OMP_NUM_THREADS=4 ./utestomp
	OMP_NUM_THREADS=8 ./utestomp
	OMP_NUM_THREADS=16 ./utestomp
	OMP_NUM_THREADS=32 ./utestomp
	OMP_NUM_THREADS=64 ./utestomp
