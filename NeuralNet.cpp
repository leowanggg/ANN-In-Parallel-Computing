#include <vector>
#include <fstream>
#include <math.h>
#include <cmath>
#include <omp.h>
#include "utils.h"
#include "main.h"
#include "NeuralNet.h"




//*************************** methods for Neuron **********************
//
//---------------------------------------------------------------------
Neuron::Neuron(int _NumInputs): NumInputs(_NumInputs+1)

{
        //we need an additional weight for the bias
        for (int i=0; i<_NumInputs+1; ++i)
        {
                //set up the weights with an initial random value
                vecWeight.push_back(RandomClamped());
        }
}

//************************ methods for NeuronLayer **********************
//
//-----------------------------------------------------------------------
NeuronLayer::NeuronLayer(int _NumNeurons, int _NumInputsPerNeuron): NumNeurons(_NumNeurons)
{
        for (int i=0; i<_NumNeurons; ++i)

               	vecNeurons.push_back(Neuron(_NumInputsPerNeuron));
}




//************************ methods for NeuralNet ************************
//
//-----------------------------------------------------------------------
NeuralNet::NeuralNet()
{
        NumInputs = iNumInputs;
        NumOutputs = iNumOutputs;
        NumHiddenLayers = iNumHiddenLayer;
        NeuronsPerHiddenLyr = iNeuronsPerHiddenLayer;

        CreateNet();

}
//------------------------------createNet()------------------------------
//
//	this method builds the ANN. The weights are all initially set to
//	random values -1 < w < 1
//------------------------------------------------------------------------
void NeuralNet::CreateNet()
{
        //create the layers of the network
        if (NumHiddenLayers > 0)
        {
            //create first hidden layer
        	vecLayers.push_back(NeuronLayer(NeuronsPerHiddenLyr, NumInputs));

            for (int i=0; i<NumHiddenLayers-1; ++i)
            {

                vecLayers.push_back(NeuronLayer(NeuronsPerHiddenLyr, NeuronsPerHiddenLyr));
            }

            //create output layer
            vecLayers.push_back(NeuronLayer(NumOutputs, NeuronsPerHiddenLyr));
        }

        else
        {
          	//create output layer
          	vecLayers.push_back(NeuronLayer(NumOutputs, NumInputs));
        }
}
//---------------------------------GetWeights-----------------------------
//
//	returns a vector containing the weights
//
//------------------------------------------------------------------------
vector<vector<vector<double> > > NeuralNet::GetWeights() const
{
        //this will hold the weights
        vector<vector<vector<double> > > weights;
        vector<vector<double> > weightsPerLayer;


        //for each layer
        for (int i=0; i<NumHiddenLayers + 1; ++i)
        {

                //for each neuron
                for (int j=0; j<vecLayers[i].NumNeurons; ++j)
                {
                        weightsPerLayer.push_back(vecLayers[i].vecNeurons[j].vecWeight);
                }

                weights.push_back(weightsPerLayer);
                weightsPerLayer.clear();
        }

	return weights;
}
//---------------------------------GetSingleWeight-----------------------------
//
//	returns the weight
//
//------------------------------------------------------------------------
vector<vector<double> > NeuralNet::GetLayerWeights(int layer_num) const
{
        //this will hold the weights
        vector<vector<double> > weightsPerLayer;


                //for each neuron
                for (int j=0; j<vecLayers[layer_num].NumNeurons; ++j)
                {
                        weightsPerLayer.push_back(vecLayers[layer_num].vecNeurons[j].vecWeight);
                }


        return weightsPerLayer;

}
//-----------------------------------PutWeights---------------------------
//
//	replace the weights in the NN with the new values
//
//------------------------------------------------------------------------
void NeuralNet::PutWeights(const vector<vector<vector<double> > > &weights)
{

        vector<vector<double> > weightsPerLayer;

        //for each layer
        for (int i=0; i<NumHiddenLayers + 1; ++i)
        {
                weightsPerLayer = weights[i];
                //for each neuron
                for (int j=0; j<vecLayers[i].NumNeurons; ++j)
               	{

                       	vecLayers[i].vecNeurons[j].vecWeight = weightsPerLayer[j];

               	}
        }

       	return;
}

//-------------------------------Propagation-----------------------------------
//
//      given an input vector this function calculates the output vector
//
//---------------------------------------------------------------------------
vector<vector<double> > NeuralNet::Propagation(vector<double> inputs, int num_threads)
{
        //stores the resultant outputs from each layer
        vector<double> outputsPerLayer;
        //stores the resultant outputs from each layer
        vector<vector<double> > outputs;
        //for mesuring the time
        double t0, t1;

        //first check that we have the correct amount of inputs
        if (inputs.size() != NumInputs)
        {
                //just return an empty vector if incorrect.
                return outputs;
        }

	    //For each layer....
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {
          for (int i=0; i<NumHiddenLayers + 1; ++i)
          {
           	if(omp_get_thread_num() == 0){
                  t0 = omp_get_wtime();
                }
                #pragma omp single
                {
                  if ( i > 0 )
                  {
                   	inputs = outputsPerLayer;
                  }

                outputsPerLayer.clear();
                outputsPerLayer.resize(vecLayers[i].NumNeurons);
                }
                if(omp_get_thread_num() == 0){
                  t1 = omp_get_wtime();
                  sing_time_1 += t1 - t0;
                  t0 = t1;
                }
                //for each neuron 
                #pragma omp for
                for (int j=0; j<vecLayers[i].NumNeurons; ++j)
                {
                        double netinput = 0;
                        int NumInputs = vecLayers[i].vecNeurons[j].NumInputs;
                        vector<vector<double> > weightsPerLayer = GetLayerWeights(i);
                        vector<double> weightsPerNeuron = weightsPerLayer[j];

                        //for each weight
                        for (int k=0; k<NumInputs; ++k)
                        {
                                //sum the weights x inputs
                                //add in the bias
                                if(k == (NumInputs - 1))
                                  netinput += weightsPerNeuron[NumInputs-1] * dBias;
                                else
                                  netinput += weightsPerNeuron[k] * inputs[k];
                        }

                        //we can store the outputs from each layer as we generate them.
                        //The combined activation is first filtered through the sigmoid function
                        outputsPerLayer[j] = Sigmoid(netinput, dActivationResponse);

                }
                if(omp_get_thread_num() == 0){
                  t1 = omp_get_wtime();
                  para_loop_time += t1 - t0;
                  t0 = t1;
                }
                //#pragma omp barrier
                #pragma omp single
                {
                  outputs.push_back(outputsPerLayer);
                }
                if(omp_get_thread_num() == 0){
                  t1 = omp_get_wtime();
                  sing_time_2 += t1 - t0;
                }
          }
	}

        return outputs;
}
//-------------------------------Sigmoid function-------------------------
//
//------------------------------------------------------------------------
double NeuralNet::Sigmoid(double netInput, double response)
{
       return ( 1 / ( 1 + exp(-netInput / response)));
}


//---------------------------------CalError---------------------------------------
//
//	calculate the error (desired output value - real output value) for each neuron
//
//--------------------------------------------------------------------------------
vector<vector<double> > NeuralNet::CalError(const vector<double> &outputOfLastLayer,
											const vector<double> &targets,
                                            const vector<vector<vector<double> > > &weights)
{
        vector<vector<double> > errors;
        vector<double> errorsPerLayer;
        vector<double> errorsOfLastLayer;
        vector<vector<double> > weightsOfLastLayer;
        vector<double> weightsPerNeuron;
        double e = 0.0;
        //for the output layer
        for (int i = 0; i < NumOutputs; ++i)
        {
                errorsPerLayer.push_back(targets[i] - outputOfLastLayer[i]);
        }
	    errors.push_back(errorsPerLayer);
        //for each Hiddenlayer
        for (int i = NumHiddenLayers - 1; i >= 0; --i)
        {
                errorsOfLastLayer = errors[NumHiddenLayers-1-i];
                weightsOfLastLayer = weights[i+1];
                //for each Neuron
                for (int j = 0; j < vecLayers[i].NumNeurons; ++j)
                {
                        for (int k = 0; k < vecLayers[i+1].NumNeurons; ++k)
                        {
                                weightsPerNeuron = weightsOfLastLayer[k];
                                e += errorsOfLastLayer[k]*weightsPerNeuron[j];
                        }
                        errorsPerLayer.push_back(e);
                        e = 0.0;
                }
                errors.push_back(errorsPerLayer);
                errorsPerLayer.clear();
        }
        return errors;
}

//---------------------------------BackPropagation---------------------
//
//      //calculate the new value of weights after training
//
//------------------------------------------------------------------------
vector<vector<vector<double> > > NeuralNet::BackPropagation(const vector<vector<vector<double> > > &weights,
                                                            const vector<vector<double> > &outputs,
                                                            const double eta, const vector<double> &firstLayerInputs,
                                                            const vector<vector<double> > &errors,
                                                            int num_threads)
{
        //this will hold the new weights
        vector<vector<vector<double> > > n_weights;
        vector<vector<double> > n_weightsPerLayer;
        //this will hold the old weights
        vector<vector<double> > weightsPerLayer;
        //this will hold the outputs
        vector<double> outputsPerLayer;
        vector<double> errorsPerLayer;
        vector<double> inputs;
        double w = 0.0;
        //for each layer
        for (int i = 0; i < NumHiddenLayers + 1; ++i)
        {
                weightsPerLayer = weights[i];
                outputsPerLayer = outputs[i];
                errorsPerLayer = errors[NumHiddenLayers-i];
                if( i > 0)
                {
                        inputs = outputs[i-1];
                }
                else
                {
                        inputs = firstLayerInputs;
                }
                //for each neuron
                n_weightsPerLayer.resize(vecLayers[i].NumNeurons);

                for (int j = 0; j < vecLayers[i].NumNeurons; ++j)
                {
                        vector<double> weightsPerNeuron = weightsPerLayer[j];
                        //for each weight
                        vector<double> n_weightsPerNeuron = weightsPerNeuron;
                        for (int k = 0; k < vecLayers[i].vecNeurons[j].NumInputs; ++k)
                        {
                          n_weightsPerNeuron[k] = weightsPerNeuron[k] + eta*errorsPerLayer[j]*outputsPerLayer[j]*(1.0-outputsPerLayer[j])*inputs[k];
                        }
                          n_weightsPerLayer[j] = n_weightsPerNeuron;

                }
                n_weights.push_back(n_weightsPerLayer);
                n_weightsPerLayer.clear();
        }
	return n_weights;
}












