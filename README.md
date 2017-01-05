# Artificial Neural Network in Parallel Computing

This is the archives of my PFE(projet de fin d'etude).In this project,
I realize an Artificial Neural Network which can recognize two different 
gestures in C++. Then I parallelize this program by library OpenMP to improve 
the performance.

## Installation:

### Systems Tested:

- Fedora 20

### Requirements:

- a computer with a Xeon-phi CPU
- g++
- OpenMp library
- a database of the gestures(set the path of you directory of the database at line 92 in the pre-training.cpp file)

## Start:

    $ make
    $ ./main

## PS:

You can see the details of this project by reading Report.pdf