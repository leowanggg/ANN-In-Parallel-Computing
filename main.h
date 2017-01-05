#ifndef MAIN_H
#define MAIN_H

  //used for the neural network
  static int    iNumInputs = 21;
  static int    iNumHiddenLayer = 5;
  static int    iNeuronsPerHiddenLayer = 10;
  static int    iNumOutputs = 1;

  //for tweeking the sigmoid function
  static double dActivationResponse = 1.0;
  //bias value
  static double dBias = 1.0;
  //desired output value
  static vector<vector<double> > targets = {{0.0},{1.0}};
  //the step of the training
  static double eta = 0.1;
  // used for measuring the execution time
  extern double para_loop_time;
  extern double sing_time_1;
  extern double sing_time_2;
  
#endif

