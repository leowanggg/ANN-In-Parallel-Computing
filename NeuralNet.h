#ifndef NEURALNET_H
#define NEURALNET_H

using namespace std;

//-------------------------------------------------------------------
//	struct of neuron.
//-------------------------------------------------------------------
struct Neuron
{
        //the number of inputs into the neuron
        int NumInputs;

        //the weights for each input
        vector<double> vecWeight;


        //constructor
        Neuron(int _NumInputs);
};


//---------------------------------------------------------------------
//	struct of layer.
//---------------------------------------------------------------------

struct NeuronLayer
{
        //the number of neurons in this layer
        int     NumNeurons;

        //the layer of neurons
		vector<Neuron>  vecNeurons;

        NeuronLayer(int _NumNeurons, int _NumInputsPerNeuron);
};


//----------------------------------------------------------------------
//	neural net class
//----------------------------------------------------------------------

class NeuralNet
{

private:

        int     NumInputs;

        int     NumOutputs;

        int     NumHiddenLayers;

        int     NeuronsPerHiddenLyr;

        //storage for each layer of neurons including the output layer
        vector<NeuronLayer> vecLayers;



public:

       	NeuralNet();

        void CreateNet();

        //gets the weights from the NN
        vector<vector<vector<double> > > GetWeights()const;
        //gets the weight
        vector<vector<double> > GetLayerWeights(int layer_num) const;
        //replaces the weights with new ones
        void PutWeights(const vector<vector<vector<double> > > &weights);

        //calculates the outputs from a set of inputs
        vector<vector<double> > Propagation(vector<double> inputs, int num_threads);

        //sigmoid response curve
        inline double Sigmoid(double activation, double response);

        //calculate the error (desired output value - real output value) for each neuron
        vector<vector<double> > CalError(const vector<double> &outputOfLastLayer,
                                         const vector<double> &targets,
                                         const vector<vector<vector<double> > > &weights);

        //calculate the new value of weights after training
        vector<vector<vector<double> > > BackPropagation(const vector<vector<vector<double> > > &weights,
                                                         const vector<vector<double> > &outputs,
                                                         const double eta, const vector<double> &firstLayerInputs,
                                                         const vector<vector<double> > &errors, int num_threads);
};




#endif



