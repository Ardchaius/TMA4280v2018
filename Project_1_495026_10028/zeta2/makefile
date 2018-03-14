# Run hybridzeta
hybridzeta: hybridzeta.c
	mpicc hybridzeta.c -lm -o hybridzeta -fopenmp
	OMP_NUM_THREADS=8 mpirun -np 8 ./hybridzeta

# Run hybridzetaplot
hybridzetaplot: hybridzetaplot.c
	mpicc hybridzetaplot.c -lm -o hybridzetaplot -fopenmp
	OMP_NUM_THREADS=2 mpirun -np 2 ./hybridzetaplot
	OMP_NUM_THREADS=2 mpirun -np 8 ./hybridzetaplot
	OMP_NUM_THREADS=4 mpirun -np 2 ./hybridzetaplot
	OMP_NUM_THREADS=4 mpirun -np 8 ./hybridzetaplot
	OMP_NUM_THREADS=8 mpirun -np 2 ./hybridzetaplot
	OMP_NUM_THREADS=8 mpirun -np 8 ./hybridzetaplot

# Run OpenMP Zeta
ompzeta: openmpzeta.cpp
	g++ openmpzeta.cpp -lm -o ompzeta -fopenmp
	OMP_NUM_THREADS=2 ./ompzeta
	OMP_NUM_THREADS=4 ./ompzeta
	OMP_NUM_THREADS=8 ./ompzeta
	OMP_NUM_THREADS=16 ./ompzeta
	OMP_NUM_THREADS=32 ./ompzeta
	OMP_NUM_THREADS=64 ./ompzeta

# Run Hybrid Zeta utest
utesthybrid: utesthybridzeta.c
	mpicc utesthybridzeta.c -lm -o utesthybrid -fopenmp
	OMP_NUM_THREADS=2 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=2 mpirun -np 8 ./utesthybrid
	OMP_NUM_THREADS=4 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=4 mpirun -np 8 ./utesthybrid
	OMP_NUM_THREADS=8 mpirun -np 2 ./utesthybrid
	OMP_NUM_THREADS=8 mpirun -np 8 ./utesthybrid

# Run OMP Zeta utest
utestomp: utestopenmpzeta.cpp
	g++ utestopenmpzeta.cpp -lm -o utestomp -fopenmp
	OMP_NUM_THREADS=2 ./utestomp
	OMP_NUM_THREADS=4 ./utestomp
	OMP_NUM_THREADS=8 ./utestomp
	OMP_NUM_THREADS=16 ./utestomp
	OMP_NUM_THREADS=32 ./utestomp
	OMP_NUM_THREADS=64 ./utestomp
	
