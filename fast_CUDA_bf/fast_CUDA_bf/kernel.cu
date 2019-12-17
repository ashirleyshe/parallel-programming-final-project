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

/*CUDA fast version*/
void fast_bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG);
void __global__ CUDA_init_color(float* color_weight, double gauss_color_coeff);
void __global__ CUDA_init_space(float* space_weight, int* space_ofs, int cols, double gauss_space_coeff, int radius);
void __global__ CUDA_filter(uchar* src, uchar* dst, int dst_rows, int dst_cols, float* space_weight, float* color_weight, int* space_ofs, int radius, int maxk);

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
		fast_bilateralFilter(src, dst, 13, 13);
		if (write_img == 'y') {
			stringstream ss;
			string name = filenames[i].substr(filenames[i].find_last_of('\\')+1).operator string();
			ss << "CUDA_fast_bf_" << name;
			string filename = ss.str();
			cout << filename << endl;
			ss.str("");
			imwrite(filename, dst);
		}
		
	}
	end = clock();
	cout << "CUDA fast bilateralfilter total time = " << (end - start) / 1000.0 << "seconds" << endl;

	waitKey(0);
	return(0);
}


// initialize color-related bilateral filter coefficients
void __global__ CUDA_init_color(float* color_weight, double gauss_color_coeff) {
	int i = threadIdx.x;
	color_weight[i] = (float)exp(i*i*gauss_color_coeff);
}

// initialize space-related bilateral filter coefficients
void __global__ CUDA_init_space(float* space_weight, int* space_ofs, int cols, double gauss_space_coeff, int radius) {
	int i = threadIdx.x - radius;
	int j = threadIdx.y - radius;
	int maxk = threadIdx.x * (radius * 2) + threadIdx.y;

	if (i < radius && j < radius) {
		double r = sqrt((double)i*i + (double)j*j);  //sqrt不可拆
		space_weight[maxk] = (float)exp(r*r*gauss_space_coeff);
		space_ofs[maxk] = (int)(i * cols + j);
	}
}

void __global__ CUDA_filter(uchar* src, uchar* dst, int dst_rows, int dst_cols, float* space_weight, float* color_weight, int* space_ofs, int radius, int maxk) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < dst_rows && j < dst_cols)
	{
		float sum = 0, wsum = 0;
		int src_cols = dst_cols + 2 * radius;
		int val0 = src[(i + radius) * src_cols + j + radius];

		for (int k = 0; k < maxk; k++)
		{
			int val = src[(i + radius) * src_cols + j + radius + space_ofs[k]];
			float w = space_weight[k] * color_weight[abs(val - val0)];
			sum += val * w;
			wsum += w;
		}
		dst[i*dst_cols + j] = (uchar)round(sum / wsum);
	}

}

void fast_bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG)
{
	double gauss_color_coeff = -0.5 / (sigmaG*sigmaG);
	double gauss_space_coeff = -0.5 / (sigmaS*sigmaS);
	//int maxk = 0;
	//Size size = src.size();

	int radius = 7;
	int d = radius * 2 + 1;

	Mat temp;
	copyMakeBorder(src, temp, radius, radius, radius, radius, BORDER_DEFAULT);

	vector<float> _color_weight(256);
	vector<float> _space_weight(d*d);
	vector<int> _space_ofs(d*d);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	uchar *d_src, *d_dst;
	float *d_sW, *d_cW;
	int *d_spof;

	int src_size = temp.rows * temp.cols * sizeof(uchar);  //d_src is padded
	int dst_size = dst.rows * dst.cols * sizeof(uchar);
	int sW_size = d * d * sizeof(float);
	int cW_size = 256 * sizeof(float);
	int spof_size = d * d * sizeof(int);

	cudaMalloc(&d_src, src_size);
	cudaMalloc(&d_dst, dst_size);
	cudaMalloc(&d_sW, sW_size);
	cudaMalloc(&d_cW, cW_size);
	cudaMalloc(&d_spof, spof_size);

	cudaMemcpy(d_src, temp.data, src_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst.data, dst_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sW, space_weight, sW_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cW, color_weight, cW_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_spof, space_ofs, spof_size, cudaMemcpyHostToDevice);

	CUDA_init_color << < 1, 256 >> > (d_cW, gauss_color_coeff);

	dim3 filter_block(d, d);
	CUDA_init_space << < 1, filter_block >> > (d_sW, d_spof, temp.cols, gauss_space_coeff, radius);

	int maxk = d * d;

	dim3 block(8, 8);
	dim3 grid((dst.rows + block.x - 1) / block.x, (dst.cols + block.y - 1) / block.y);
	CUDA_filter << < grid, block >> > (d_src, d_dst, dst.rows, dst.cols, d_sW, d_cW, d_spof, radius, maxk);

	cudaMemcpy(dst.data, d_dst, dst_size, cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(d_sW);
	cudaFree(d_cW);
	cudaFree(d_spof);
}