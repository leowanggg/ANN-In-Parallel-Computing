#include "pre-training.h"
#include "NeuralNet.h"
#include "main.h"
#include <omp.h>
#include <chrono>
//set the threshold for the convergent of the algorithm
#define THRESHOLD 0.01
//for mesuring the time
double para_loop_time = 0.0;
double sing_time_1 = 0.0;
double sing_time_2 = 0.0;

int main(int argc, char const *argv[])
{
    //declare a InputVector object
    InputVector iv;
    //declare a vector of vector for storing the inputvectors
    vector<vector<double> > inputvectorsTrain;
    vector<vector<double> > inputvectorsTest;
    //calculate the fourier series from the images
    cout << "************************************" << endl;
    cout << "* The parameters of Neural Network *" << endl;
    cout << "************************************" << endl;
    cout << "The number of inputs: " << iNumInputs << endl;
    cout << "The number of hidden layers: "<< iNumHiddenLayer << endl;
    cout << "The number of neurons per layer: " << iNeuronsPerHiddenLayer << endl;
    cout << "The number of outputs " << iNumOutputs << endl;
    cout << "------------------------------------" << endl;
    cout << "Loading data..." << endl;
    iv.CalVector();
    //store the fourier series in vector
    iv.GetVector(inputvectorsTrain, inputvectorsTest);
    //declare a NeuralNet object
    NeuralNet nn;
    //declare a vector for storing the result of the neuralnetwork
    //training
    int k;
    vector<vector<vector<double> > > weights;
    vector<vector<vector<double> > > n_weights;
    vector<vector<double> > errors;
    vector<vector<double> > outputs;
    vector<double> outputsPerLayer;
    //number of threads
    int num_threads = atoi(argv[1]);
    //start training
    cout << "Start training..." << endl;
    cout << "The number of threads for training: " << num_threads << endl;
    cout << "The number of examples for training: " << inputvectorsTrain.size() << endl;
    cout << "The number of iteration for training: " << 60000 << endl;
    std::chrono::microseconds dur_1{0};
    std::chrono::microseconds dur_2{0};
    std::chrono::microseconds dur_3{0};
    chrono::time_point<chrono::steady_clock> start_1;
    chrono::time_point<chrono::steady_clock> start_2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 60000; ++i)
    {
        int randIndex = rand() % inputvectorsTrain.size();
        if (randIndex < 100)
            k = 0;
        else
            k = 1;
        start_1 = std::chrono::steady_clock::now();
        vector<vector<double> > outputs = nn.Propagation(inputvectorsTrain[randIndex], num_threads);
        dur_1 += chrono::duration_cast<chrono::microseconds>(std::chrono::steady_clock::now() - start_1);
        weights = nn.GetWeights();
        errors = nn.CalError(outputs[iNumHiddenLayer], targets[k], weights);
        start_1 = std::chrono::steady_clock::now();
        n_weights = nn.BackPropagation(weights, outputs, eta, inputvectorsTrain[randIndex], errors, num_threads);
        dur_3 += chrono::duration_cast<chrono::microseconds>(std::chrono::steady_clock::now() - start_1);
        nn.PutWeights(n_weights);
    }
    dur_2 = chrono::duration_cast<chrono::microseconds>(std::chrono::steady_clock::now() - start_2);
    cout << "propagation took " << (float) dur_1.count()/1000000 << " seconds." << endl;
    cout << "backpropagation took " << (float) dur_3.count()/1000000 << " seconds." << endl;
    cout << "Training took " << (float) dur_2.count()/1000000 << " seconds." << endl;
    cout << "para_loop_time = " << para_loop_time << " seconds" << endl;
    cout << "single_time_1 = " << sing_time_1 << " seconds" << endl;
    cout << "single_time_2 = " << sing_time_2 << " seconds" << endl;
    //test
    cout << "Start testing..." << endl;
    cout << "The number of examples for testing: " << inputvectorsTest.size() << endl;
    int correct = 0;
    for (int i = 0; i < inputvectorsTest.size(); ++i)
    {
        if (i < 100)
            k = 0;
        else
            k = 1;
        outputs = nn.Propagation(inputvectorsTest[i], 1);
        outputsPerLayer = outputs[iNumHiddenLayer];
        vector<double> target = targets[k];
        if(abs(target[0]-outputsPerLayer[0])<THRESHOLD)
            correct++;
    }
    double accuracy = (double)correct/(double)inputvectorsTest.size();
    cout << "The accuracy of recognition: " << accuracy*100 << "%" << endl;
    return 0;
}

