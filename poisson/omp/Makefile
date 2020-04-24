DEBUG=0
CC=g++ -m64
OMP=-fopenmp -DOMP
CFLAGS=-O3 -Wall
CFLAGS+= -DOPENCV `pkg-config opencv --cflags --libs`
OPENCVFLAGS=-L/afs/andrew.cmu.edu/usr18/yuxindin/private/15-618/opencv34/lib64 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs
LDFLAGS= -lm

CFILES = PoissonImageEdit.cpp
all: PoissonImageEdit

PoissonImageEdit: $(CFILES) 
	$(CC) $(CFLAGS) $(OMP) -o PoissonImageEdit $(CFILES) $(LDFLAGS) $(OPENCVFLAGS)

clean:
	rm -f PoissonImageEdit