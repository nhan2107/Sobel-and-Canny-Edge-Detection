#define _CRT_SECURE_NO_WARNINGS
#include "EdgeDectection.h"
int main()
{
	Mat gray_img;
	Mat dst_img;
	Mat img = imread("al.jpg", IMREAD_COLOR);
	if (img.empty())
	{
		cout << "Could not find the image";
		return 1;
	}
	cvtColor(img, gray_img, COLOR_RGB2GRAY);
	EdgeDetection ED;
	if (ED.detectBySobel(gray_img, dst_img, 1) == 0)
	{
		imshow("SobelX", dst_img);
		imwrite("SobelX.png", dst_img);
	}
	if (ED.detectBySobel(gray_img, dst_img, 2) == 0)
	{
		imshow("SobelY", dst_img);
		imwrite("SobelY.png", dst_img);
	}
	if (ED.detectBySobel(gray_img, dst_img, 3) == 0)
	{
		imshow("SobelXY", dst_img);
		imwrite("SobelXY.png", dst_img);
	}
	if (ED.detectByCanny(gray_img, dst_img) == 0)
	{
		imshow("Canny", dst_img);
		imwrite("Canny.png", dst_img);
	}
	if (ED.CannyOpenCV(gray_img, dst_img) == 0)
	{
		imshow("CannyOpenCV", dst_img);
		imwrite("CannyOpenCV.png", dst_img);
	}
	else cout << "Errors occur!";
	waitKey(0);
	return 0;
}