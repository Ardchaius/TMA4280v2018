# Run hybridzeta
# Run basic hybrid zeta letting you input an n
hybridzeta: hybridzeta.c
	mpicc hybridzeta.c -lm -o hybridzeta -fopenmp
	OMP_NUM_THREADS=8 mpirun -np 8 ./hybridzeta

# Run hybridzetaplot
# Runs the vtest for hybrid zeta outputting
# multiple files with data on the method
hybridzetaplot: hybridzetaplot.c
	mpicc hybridzetaplot.c -lm -o hybridzetaplot -fopenmp
	OMP_NUM_THREADS=2 mpirun -np 2 ./hybridzetaplot
	OMP_NUM_THREADS=2 mpirun -np 8 ./hybridzetaplot
	OMP_NUM_THREADS=4 mpirun -np 2 ./hybridzetaplot
	OMP_NUM_THREADS=4 mpirun -np 8 ./hybridzetaplot
	OMP_NUM_THREADS=8 mpirun -np 2 ./hybridzetaplot
	OMP_NUM_THREADS=8 mpirun -np 8 ./hybridzetaplot

# Run OpenMP Zeta
# Runs the vtest for the OpenMP zeta outputting
# multiple files with data on the method
ompzeta: openmpzeta.cpp
	g++ openmpzeta.cpp -lm -o ompzeta -fopenmp
	OMP_NUM_THREADS=2 ./ompzeta
	OMP_NUM_THREADS=4 ./ompzeta
	OMP_NUM_THREADS=8 ./ompzeta
	OMP_NUM_THREADS=16 ./ompzeta
	OMP_NUM_THREADS=32 ./ompzeta
	OMP_NUM_THREADS=64 ./ompzeta

# Run Hybrid Zeta utest
# Runs the utest for hybrid zeta
utesthybrid: utesthybridzeta.c
	mpicc utesthybridzeta.c -lm -o utesthybrid -fopenmp
	OMP_NUM_THREADS=2 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=2 mpirun -np 8 ./utesthybrid
	OMP_NUM_THREADS=4 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=4 mpirun -np 8 ./utesthybrid
	OMP_NUM_THREADS=8 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=8 mpirun -np 8 ./utesthybrid

# Run OMP Zeta utest
# Runs the utest for OpenMP zeta
utestomp: utestopenmpzeta.cpp
	g++ utestopenmpzeta.cpp -lm -o utestomp -fopenmp
	OMP_NUM_THREADS=2 ./utestomp
	OMP_NUM_THREADS=4 ./utestomp
	OMP_NUM_THREADS=8 ./utestomp
	OMP_NUM_THREADS=16 ./utestomp
	OMP_NUM_THREADS=32 ./utestomp
	OMP_NUM_THREADS=64 ./utestomp
	
