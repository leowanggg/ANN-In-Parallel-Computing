#ifndef _PRE_TRAINING_H
#define _PRE_TRAINING_H
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <math.h>
#include <cmath>
#include <string>
#include <sstream>
#include "ml.h"
#include "highgui.h"

using namespace cv;
using namespace std;

#define PI 3.1415926
#define K_NUM 10

//----------------------------------------------------------------------
//  Input vectors class
//----------------------------------------------------------------------
class InputVector
{
private:
  //vectors of the input for training
  vector<vector<double> > inputvectorsTrain;
  //vectors of the input for test
  vector<vector<double> > inputvectorsTest;
public:

  //constructor
  InputVector(){};

  //deconstructor
  ~InputVector(){};

  //calculate the Fourier series from the images and store them in vectors
  void CalVector(void);

  //get the input vectors
  void GetVector(vector<vector<double> > &_inputvectorsTrain,
                 vector<vector<double> > &_inputvectorsTest){_inputvectorsTrain = inputvectorsTrain;
                                                             _inputvectorsTest = inputvectorsTest;}
};
//extract contour from image
vector<Point> PreTrain(Mat& frame);

//convert the ordinates of the contour into series of complex numbers
//(x,y) -> x+iy
vector<complex<double> > GetComplex(vector<Point>& largest_contour);

//function to calculate the average of series of complex numbers
complex<double> CalAvg(vector<complex<double> >& zm, int N);

//Fourier Descriptor
vector<complex<double> > FourierDescriptor(vector<Point>& contour);

//get the absulute value of complex numbers
vector<double> ToDouble(vector<complex<double> >& ak);

#endif
