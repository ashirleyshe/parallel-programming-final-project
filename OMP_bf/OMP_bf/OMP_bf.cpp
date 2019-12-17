#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <omp.h>

#include <sstream>

using namespace std;
using namespace cv;

/* omp version */
void bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG);

int main() {

	vector<String> filenames;
	String folder = "D:\\lesson\\parallel\\img1000";
	glob(folder, filenames);
	int datasize;
	int maxsize = filenames.size();
	cout << "input image number(1~" << maxsize << "): ";
	while (cin >> datasize) {
		if (datasize > 0 && datasize <= maxsize)
			break;
		cout << "error input, input must in 1~" << maxsize << ":";
	}

	char write_img;
	cout << "write result?[y/n]: ";
	while (cin >> write_img) {
		if (write_img == 'y' || write_img == 'n')
			break;
		cout << "error input, input y or n: ";
	}

	unsigned long start, end;
	start = clock();
	for (size_t i = 0; i < datasize; ++i)
	{
		Mat src = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;

		Mat dst(src.rows, src.cols, CV_8U);

		// do bilateral filter
		bilateralFilter(src, dst, 13, 13);
		if (write_img == 'y') {
			stringstream ss;
			string name = filenames[i].substr(filenames[i].find_last_of('\\') + 1).operator string();
			ss << "OMP_bf_" << name;
			string filename = ss.str();
			cout << filename << endl;
			ss.str("");
			imwrite(filename, dst);
		}

	}
	end = clock();
	cout << "OMP bilateralfilter total time = " << (end - start) / 1000.0 << "seconds" << endl;

	waitKey(0);
	return(0);
}

double spacedistance(int x1, int y1, int x2, int y2, double sigmaS) {
	double X = pow(abs(x1 - x2), 2);
	double Y = pow(abs(y1 - y2), 2);

	return exp(-(X + Y) / (2 * pow(sigmaS, 2)));
}

double GSdistance(int g1, int g2, double sigmaG) {
	double X = pow(abs(g1 - g2), 2);
	return exp(-X / (2 * pow(sigmaG, 2)));
}

void bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG) { //雙側濾波器
	int m = 7; //15*15
	int rows = src.rows;
	int cols = src.cols;

#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double k = 0;
			double f = 0;
#pragma omp parallel for reduction(+:f, k)
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					f = f + src.at<uchar>(y, x) * spacedistance(i, j, y, x, sigmaS) * GSdistance(src.at<uchar>(i, j), src.at<uchar>(y, x), sigmaG);
					k = k + spacedistance(i, j, y, x, sigmaS) * GSdistance(src.at<uchar>(i, j), src.at<uchar>(y, x), sigmaG);
				}
			}
			int g = f / k;
			if (g < 0) g = 0;
			else if (g > 255) g = 255;
			dst.at<uchar>(i, j) = (uchar)g;
		}
	}
}
