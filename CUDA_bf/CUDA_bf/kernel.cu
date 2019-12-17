#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

using namespace std;
using namespace cv;

/*CUDA version*/
void bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG);
double __device__ CUDA_spacedistance(int x1, int y1, int x2, int y2, double sigmaS);
double __device__ CUDA_GSdistance(int g1, int g2, double sigmaG);
void __global__ CUDA_bilateralfilter(uchar *d_src, uchar *d_dst, int rows, int cols, double sigmaS, double sigmaG);

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
			ss << "CUDA_bf_" << name;
			string filename = ss.str();
			cout << filename << endl;
			ss.str("");
			imwrite(filename, dst);
		}

	}
	end = clock();
	cout << "CUDA bilateralfilter total time = " << (end - start) / 1000.0 << " seconds" << endl;

	waitKey(0);
	return(0);
}


//空間距離差異
double __device__ CUDA_spacedistance(int x1, int y1, int x2, int y2, double sigmaS) {
	int xx = abs(x1 - x2);
	int yy = abs(y1 - y2);
	double X = xx * xx;
	double Y = yy * yy;
	return exp(-(X + Y) / (2 * pow(sigmaS, 2)));
}

//灰階距離差異
double __device__ CUDA_GSdistance(int g1, int g2, double sigmaG) {
	int xx = abs(g1 - g2);
	double X = xx * xx;
	return exp(-X / (2 * pow(sigmaG, 2)));
}


void __global__ CUDA_bilateralfilter(uchar *d_src, uchar *d_dst, int rows, int cols, double sigmaS, double sigmaG) {
	int m = 7; //15*15
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < rows && j < cols)
	{
		double k = 0;
		double f = 0;
		for (int y = i - m; y <= i + m; y++)  //row
		{
			for (int x = j - m; x <= j + m; x++)  //column
			{
				if (y < 0 || x < 0 || y >= rows || x >= cols)
				{
					continue;
				}
				int YX = (int)d_src[y*cols + x];
				int IJ = (int)d_src[i*cols + j];
				f = f + YX * CUDA_spacedistance(i, j, y, x, sigmaS) * CUDA_GSdistance(IJ, YX, sigmaG);
				k = k + CUDA_spacedistance(i, j, y, x, sigmaS) * CUDA_GSdistance(IJ, YX, sigmaG);
			}
		}
		int g = f / k;
		if (g < 0) g = 0;
		else if (g > 255) g = 255;
		d_dst[i*cols + j] = (uchar)g;

	}

}

void bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG)
{
	uchar *d_src;
	uchar *d_dst;
	int size = src.rows * src.cols * sizeof(uchar);
	cudaMalloc(&d_src, size);
	cudaMalloc(&d_dst, size);
	cudaMemcpy(d_src, src.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst.data, size, cudaMemcpyHostToDevice);

	dim3 block(8, 8);
	dim3 grid((src.rows + block.x - 1) / block.x, (src.cols + block.y - 1) / block.y);
	CUDA_bilateralfilter << < grid, block >> > (d_src, d_dst, src.rows, src.cols, 13, 13);

	cudaMemcpy(dst.data, d_dst, size, cudaMemcpyDeviceToHost);
	cudaFree(d_src);
	cudaFree(d_dst);
}