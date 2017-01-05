#include "pre-training.h"
#include <omp.h>

vector<Point> PreTrain(Mat& frame)
{
  Size size = frame.size();
  int rows = size.height;
  int cols = size.width;
  //cvtColor(frame, frame, CV_BGR2GRAY);
  vector<Point> contour;
  Point point;
  for(int i = 1; i < rows; ++i)
  {
    for(int j = 1; j < cols; ++j)
    {
      if (frame.at<uchar>(i,j) != 0)
      {
       	point.x = i;
        point.y = j;
        contour.push_back(point);
      }
    }
  }
  return contour;
}

vector<complex<double> > GetComplex(vector<Point>& largest_contour)
{
  complex<double> z;
  vector<complex<double> > zm;
  vector<complex<double> >::iterator it_zm;
  vector<Point>::iterator it_points;
  for (it_points = largest_contour.begin(); it_points != largest_contour.end(); ++it_points)
  {
    z.real(it_points -> x);
    z.imag(it_points -> y);
    zm.push_back(z);
  }
  return zm;
}

complex<double> CalAvg(vector<complex<double> >& zm, int N)
{
  complex<double> z_average;
  vector<complex<double> >::iterator it_zm;
  for (it_zm = zm.begin(); it_zm != zm.end(); ++it_zm)
  {
    z_average += *it_zm;
  }
  z_average = z_average/double(N);
  return z_average;
}

vector<complex<double> > FourierDescriptor(vector<Point>& contour)
{
  vector<complex<double> > zm = GetComplex(contour);
  int N = zm.size();
  complex<double> z_average = CalAvg(zm, N);
  vector<complex<double> > ak;
  complex<double> a;
  complex<double> e;
  complex<double> j;
  double t = -2*PI/N;
  for (int k = -K_NUM; k < K_NUM + 1; ++k)
  {
    for (int m = 0; m < N; ++m)
    {
      j = complex <double>(0,t*k*m);
      e = exp(j);
      a += (zm[m]-z_average)*e;
    }
    a = a/double(N);
    ak.push_back(a);
    a = complex <double>(0,0);
  }
  return ak;
}

vector<double> ToDouble(vector<complex<double> >& ak)
{
  vector<double> vec;
  for(auto iter = ak.begin(); iter != ak.end(); ++iter)
  {
    vec.push_back(abs(*iter));
  }
   return vec;
}

void InputVector::CalVector(void)
{
  //path of the images
  string a = "../../../opt/base/contour/motion_";
  string b = "/";
  string c = ".png";
  string a_ ;
  string b_;
  a_ = to_string(1);
  for (int j = 21; j < 121; ++j)
  {
    b_ = to_string(50);
    Mat img = imread(a + a_ + b + b_ + c, CV_LOAD_IMAGE_GRAYSCALE);
    vector<Point> contour = PreTrain(img);
    vector<complex<double> > ak = FourierDescriptor(contour);
    vector<double> vector = ToDouble(ak);
    inputvectorsTrain.push_back(vector);
  }
  a_ = to_string(5);
  for (int j = 21; j < 121; ++j)
  {
    b_ = to_string(50);
    Mat img = imread(a + a_ + b + b_ + c, CV_LOAD_IMAGE_GRAYSCALE);
    vector<Point> contour = PreTrain(img);
    vector<complex<double> > ak = FourierDescriptor(contour);
    vector<double> vector = ToDouble(ak);
    inputvectorsTrain.push_back(vector);
  }
  a_ = to_string(1);
  for (int j = 121; j < 221; ++j)
  {
    b_ = to_string(j);
    Mat img = imread(a + a_ + b + b_ + c, CV_LOAD_IMAGE_GRAYSCALE);
    vector<Point> contour = PreTrain(img);
    vector<complex<double> > ak = FourierDescriptor(contour);
    vector<double> vector = ToDouble(ak);
    inputvectorsTest.push_back(vector);
  }
  a_ = to_string(5);
  for (int j = 121; j < 221; ++j)
  {
    b_ = to_string(j);
    Mat img = imread(a + a_ + b + b_ + c, CV_LOAD_IMAGE_GRAYSCALE);
    vector<Point> contour = PreTrain(img);
    vector<complex<double> > ak = FourierDescriptor(contour);
    vector<double> vector = ToDouble(ak);
    inputvectorsTest.push_back(vector);
  }
  a_ = to_string(5);
  for (int j = 121; j < 221; ++j)
  {
    b_ = to_string(j);
    Mat img = imread(a + a_ + b + b_ + c, CV_LOAD_IMAGE_GRAYSCALE);
    vector<Point> contour = PreTrain(img);
    vector<complex<double> > ak = FourierDescriptor(contour);
    vector<double> vector = ToDouble(ak);
    inputvectorsTest.push_back(vector);
  }
}

 