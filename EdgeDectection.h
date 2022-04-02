#pragma once
#pragma once
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include<math.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
using namespace cv;
using namespace std;
class EdgeDetection
{
public:
	int detectBySobel(Mat src, Mat& dst, int method);
	int detectByCanny(Mat sourceImage, Mat& destinationImage);
	int CannyOpenCV(Mat sourceImage, Mat& destinationImage);
	EdgeDetection();
	~EdgeDetection();
private:
	vector<vector<double>> Filter(int r, int c, double sigma); //create gaussian filter
	int useFilter(Mat src, Mat& dst, vector<vector<double>> f); //use gaussian filter
	int NonMaxSupp(Mat src, Mat& dst); //Non-maxima supp.
	Mat _angles;
};