CC = g++
FLAGS = -O3 -std=c++11 `pkg-config --cflags --libs opencv` -fopenmp
SOURCES = pre-training.cpp NeuralNet.cpp main.cpp
EXECNAME = ANN

pre-training: pre-training.cpp
			$(CC) $(FLAGS) $(SOURCES) -o $(EXECNAME)
clean:
	rm -rf *.o







