#include "EdgeDectection.h"
int EdgeDetection::detectBySobel(Mat src, Mat& dst, int method)
{
	if (src.channels() != 1)
	{
		return 1;
	}
	//Sobel X Filter
	double x1[] = { -1.0, 0, 1.0 };
	double x2[] = { -2.0, 0, 2.0 };
	double x3[] = { -1.0, 0, 1.0 };
	vector<vector<double>> Fx(3);
	Fx[0].assign(x1, x1 + 3);
	Fx[1].assign(x2, x2 + 3);
	Fx[2].assign(x3, x3 + 3);
	//Sobel Y Filter
	double y1[] = { 1.0, 2.0, 1.0 };
	double y2[] = { 0, 0, 0 };
	double y3[] = { -1.0, -2.0, -1.0 };
	vector<vector<double>> Fy(3);
	Fy[0].assign(y1, y1 + 3);
	Fy[1].assign(y2, y2 + 3);
	Fy[2].assign(y3, y3 + 3);
	//Limit Size
	int size = (int)Fx.size() / 2;
	int r = src.rows;
	int c = src.cols;
	dst = Mat(r - 2 * size, c - 2 * size, CV_8UC1);
	switch (method)
	{
	case 3:
	{
		_angles = Mat(r - 2 * size, c - 2 * size, CV_32FC1); //AngleMap
		for (int i = size; i < r - size; i++)
			for (int j = size; j < c - size; j++)
			{
				double sx = 0;
				double sy = 0;
				for (int x = 0; x < Fx.size(); x++)
					for (int y = 0; y < Fy.size(); y++)
					{
						sx += Fx[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size)); //X Filter Value
						sy += Fy[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size)); //Y Filter Value
					}
				double e = sqrt(sx * sx + sy * sy);
				if (e > 255)  //Unsigned Char Fix
				{
					e = 255;
				}
				dst.at<uchar>(i - size, j - size) = e;

				if (sx == 0) //Arctan Fix
				{
					_angles.at<float>(i - size, j - size) = 90;
				}
				else
				{
					_angles.at<float>(i - size, j - size) = atan(sy / sx);
				}
			}
	}
	break;
	case 1:
	{
		for (int i = size; i < r - size; i++)
			for (int j = size; j < c - size; j++)
			{
				double sx = 0;
				for (int x = 0; x < Fx.size(); x++)
					for (int y = 0; y < Fx.size(); y++)
					{
						sx += Fx[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size)); //X Filter Value
					}
				double e = sx;
				if (e > 255)  //Unsigned Char Fix
				{
					e = 255;
				}
				dst.at<uchar>(i - size, j - size) = e;
			}
	}
	break;
	case 2:
	{
		for (int i = size; i < r - size; i++)
			for (int j = size; j < c - size; j++)
			{
				double sy = 0;
				for (int x = 0; x < Fy.size(); x++)
					for (int y = 0; y < Fy.size(); y++)
					{
						sy += Fy[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size)); //Y Filter Value
					}
				double e = sy;
				if (e > 255)  //Unsigned Char Fix
				{
					e = 255;
				}
				dst.at<uchar>(i - size, j - size) = e;
			}
	}
	break;
	}
	return 0;
}

vector<vector<double>> EdgeDetection::Filter(int r, int c, double sigma)
{
	vector<vector<double>> f;

	for (int i = 0; i < r; i++)
	{
		vector<double> col;
		for (int j = 0; j < c; j++)
		{
			col.push_back(-1);
		}
		f.push_back(col);
	}

	float cSum = 0;
	float cons = 2.0 * sigma * sigma;

	// Sum is for normalization
	float s = 0.0;

	for (int x = -r / 2; x <= r / 2; x++)
		for (int y = -c / 2; y <= c / 2; y++)
		{
			cSum = (x * x + y * y);
			f[x + r / 2][y + c / 2] = (exp(-(cSum) / cons)) / (M_PI * cons);
			s += f[x + r / 2][y + c / 2];
		}

	// Normalize the Filter
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
		{
			f[i][j] /= s;
		}
	return f;
}

int EdgeDetection::useFilter(Mat src, Mat& dst, vector<vector<double>> f)
{
	if (src.channels() != 1)
	{
		return 1;
	}
	int size = (int)f.size() / 2;
	int r = src.rows;
	int c = src.cols;
	dst = Mat(r - 2 * size, r - 2 * size, CV_8UC1);
	for (int i = size; i < r - size; i++)
	{
		for (int j = size; j < c - size; j++)
		{
			double s = 0;

			for (int x = 0; x < f.size(); x++)
				for (int y = 0; y < f.size(); y++)
				{
					s += f[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size));
				}

			dst.at<uchar>(i - size, j - size) = s;
		}

	}
	return 0;
}
int EdgeDetection::NonMaxSupp(Mat src, Mat& dst)
{
	Mat blur_img;
	if (useFilter(src, blur_img, Filter(3, 3, 1)) == 0)
	{
		Mat sobel_img;
		if (detectBySobel(blur_img, sobel_img, 3) == 0)
		{
			int r = sobel_img.rows;
			int c = sobel_img.cols;
			dst = Mat(r - 2, c - 2, CV_8UC1);
			for (int i = 1; i < r - 1; i++) 
				for (int j = 1; j < c - 1; j++) 
				{
					float t = _angles.at<float>(i, j);

					dst.at<uchar>(i - 1, j - 1) = sobel_img.at<uchar>(i, j);
					//Horizontal Edge
					if (((-22.5 < t) && (t <= 22.5)) || ((157.5 < t) && (t <= -157.5)))
					{
						if ((sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i, j + 1)) || (sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i, j - 1)))
						{
							dst.at<uchar>(i - 1, j - 1) = 0;
						}
					}
					//Vertical Edge
					if (((-112.5 < t) && (t <= -67.5)) || ((67.5 < t) && (t <= 112.5)))
					{
						if ((sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i + 1, j)) || (sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i - 1, j)))
						{
							dst.at<uchar>(i - 1, j - 1) = 0;
						}
					}

					//-45 Degree Edge
					if (((-67.5 < t) && (t <= -22.5)) || ((112.5 < t) && (t <= 157.5)))
					{
						if ((sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i - 1, j + 1)) || (sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i + 1, j - 1)))
						{
							dst.at<uchar>(i - 1, j - 1) = 0;
						}
					}

					//45 Degree Edge
					if (((-157.5 < t) && (t <= -112.5)) || ((22.5 < t) && (t <= 67.5)))
					{
						if ((sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i + 1, j + 1)) || (sobel_img.at<uchar>(i, j) < sobel_img.at<uchar>(i - 1, j - 1)))
						{
							dst.at<uchar>(i - 1, j - 1) = 0;
						}
					}
				}
			return 0;
		}
		else return 1;
	}
	else return 1;
}
int EdgeDetection::detectByCanny(Mat sourceImage, Mat& destinationImage)
{
	Mat non_img;
	if (NonMaxSupp(sourceImage, non_img) == 0)
	{
		int low = 20;
		int high = 40;
		int r = non_img.rows;
		int c = non_img.cols;
		destinationImage = Mat(r, c, non_img.type());
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
			{
				destinationImage.at<uchar>(i, j) = non_img.at<uchar>(i, j);
				if (destinationImage.at<uchar>(i, j) > high)
				{
					destinationImage.at<uchar>(i, j) = 255;
				}
				else if (destinationImage.at<uchar>(i, j) < low)
				{
					destinationImage.at<uchar>(i, j) = 0;
				}
				else
				{
					bool H = false;
					bool between = false;
					for (int x = i - 1; x < i + 2; x++)
					{
						for (int y = j - 1; y < j + 2; y++)
						{
							if (x <= 0 || y <= 0 || destinationImage.rows || y > destinationImage.cols) //Out of bounds
								continue;
							else
							{
								if (destinationImage.at<uchar>(x, y) > high)
								{
									destinationImage.at<uchar>(i, j) = 255;
									H = true;
									break;
								}
								else if (destinationImage.at<uchar>(x, y) <= high && destinationImage.at<uchar>(x, y) >= low)
								{
									between = true;
								}
							}
						}
						if (H)
						{
							break;
						}
					}
					if (!H && between)
						for (int x = i - 2; x < i + 3; x++)
						{
							for (int y = j - 1; y < j + 3; y++)
							{
								if (x < 0 || y < 0 || x > destinationImage.rows || y > destinationImage.cols) //Out of bounds
									continue;
								else
								{
									if (destinationImage.at<uchar>(x, y) > high)
									{
										destinationImage.at<uchar>(i, j) = 255;
										H = true;
										break;
									}
								}
							}
							if (H)
							{
								break;
							}
						}
					if (!H)
					{
						destinationImage.at<uchar>(i, j) = 0;
					}
				}
			}
		return 0;
	}
	else return 1;
}
int EdgeDetection::CannyOpenCV(Mat sourceImage, Mat& destinationImage)
{
		if (sourceImage.channels() != 1)
		{
			return 1;
		}
		Mat blur_img;
		if (useFilter(sourceImage, blur_img, Filter(3, 3, 1)) == 0)
		{
			Canny(blur_img, destinationImage, 20, 40);
		}
		return 0;
	
}
EdgeDetection::EdgeDetection() 
{
}

EdgeDetection::~EdgeDetection() 
{
}