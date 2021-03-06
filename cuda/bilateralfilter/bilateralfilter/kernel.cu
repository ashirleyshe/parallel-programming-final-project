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
#include <omp.h>
using namespace std;
using namespace cv;

void gaussian(Mat src, Mat dst);
void neighborhood_averaging(Mat src, Mat dst);
void median_flitering(Mat src, Mat dst);
void quickSort(int *a, int left, int right);
void peak_and_valley_flitering(Mat src, Mat dst);
void max_flitering(Mat src, Mat dst);
void min_flitering(Mat src, Mat dst);
//void colorchange(Mat src, Mat dst);
double spacedistance(int x1, int y1, int x2, int y2, double sigmaS);
double GSdistance(int g1, int g2, double sigmaG);
void bilateralfilter(Mat src, Mat dst, double sigmaS, double sigmaG);

//openMP version
void openMP_bilateralfilter(Mat src, Mat dst, double sigmaS, double sigmaG);

/*CUDA version*/
double __device__ CUDA_spacedistance(int x1, int y1, int x2, int y2, double sigmaS);
double __device__ CUDA_GSdistance(int g1, int g2, double sigmaG);
void __global__ CUDA_bilateralfilter(uchar *d_src, uchar *d_dst, int rows, int cols, double sigmaS, double sigmaG);

int main() {
	Mat src = imread("freckle.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst(src.rows, src.cols, CV_8U);
	Mat dst2(src.rows, src.cols, CV_8U);
	Mat dst3(src.rows, src.cols, CV_8U);
	Mat dst4(src.rows, src.cols, CV_8U);
	Mat dst5(src.rows, src.cols, CV_8U);
	Mat dst6(src.rows, src.cols, CV_8U);
	Mat dst7(src.rows, src.cols, CV_8U);


	unsigned long start = clock();
	neighborhood_averaging(src,dst);
	unsigned long end = clock();
	cout<<"neigbor total time="<<(end-start)/1000.0<<"seconds"<<endl;

	start = clock();
	gaussian(src, dst2);
	end = clock();
	cout<<"guassian total time="<<(end-start)/1000.0<<"seconds"<<endl;

	start = clock();
	median_flitering(src,dst3);
	end = clock();
	cout<<"median total time="<<(end-start)/1000.0<<"seconds"<<endl;

	start = clock();
	peak_and_valley_flitering(src, dst4);
	end = clock();
	cout<<"peak_and_valley total time="<<(end-start)/1000.0<<"seconds"<<endl;


	start = clock();
	bilateralfilter(src,dst5,13,13);
	end = clock();
	cout<<"bilateralfilter total time="<<(end-start)/1000.0<<"seconds"<<endl;

	//start = clock();
	//openMP_bilateralfilter(src, dst7, 13, 13);
	//end = clock();
	//cout << "openMP_bilateralfilter total time=" << (end - start) / 1000.0 << "seconds" << endl;

	start = clock();
	uchar *d_src;
	uchar *d_dst;
	int size = src.rows * src.cols * sizeof(uchar);
	cudaMalloc(&d_src, size);
	cudaMalloc(&d_dst, size);
	cudaMemcpy(d_src, src.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst6.data, size, cudaMemcpyHostToDevice);

	dim3 block(8, 8);
	dim3 grid((src.rows + block.x - 1) / block.x, (src.cols + block.y - 1) / block.y);
	CUDA_bilateralfilter <<< grid, block >>> (d_src, d_dst, src.rows, src.cols, 13, 13);

	cudaMemcpy(dst6.data, d_dst, size, cudaMemcpyDeviceToHost);
	cudaFree(d_src);
	cudaFree(d_dst);
	end = clock();
	cout << "CUDA bilateralfilter total time=" << (end - start) / 1000.0 << "seconds" << endl;

	//最大最小濾波在最小最大濾波
	/*max_flitering(src,dst);
	min_flitering(dst,dst2);
	min_flitering(dst2,dst3);
	max_flitering(dst3,dst4);*/


	imshow("orgin", src);
	imshow("neigbor",dst);
	imshow("guassian",dst2);
	imshow("median",dst3);
	imshow("peak_and_valley",dst4);
	imshow("bilateralfilter", dst5);
	imshow("CUDA_bilateralfilter", dst6);
	//imshow("openMP_bilateralfilter", dst7);


	imwrite("neigbor.jpg", dst);
	imwrite("guassian.jpg", dst2);
	imwrite("median.jpg", dst3);
	imwrite("peak_and_valley.jpg", dst4);
	imwrite("bilateralfilter.jpg", dst5);
	imwrite("CUDA_bilateralfilter.jpg", dst6);
	//imwrite("openMP_bilateralfilter.jpg", dst7);


	waitKey(0);
	return(0);
}

//相鄰像素平均法
void neighborhood_averaging(Mat src, Mat dst){
	int mask[25] = {1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1};
	int divisor = 0;
	int m = 2;
	int rows = src.rows;
	int cols = src.cols;

	for(int i = 0;i< rows;i++)
	{
		for(int j = 0;j< cols; j++)
		{
			int sum = 0;
			int index = 0;
			for ( int y= i-m; y<= i+m; y++)
			{
				for (int x= j-m; x<= j+m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					divisor = divisor + mask[index];
					sum = sum + src.at<uchar>(y, x) * mask[index++] ;
				}
			}
			sum = sum/divisor;
			if(sum>255){sum=255;}
			dst.at<uchar>(i, j) = sum;
			divisor = 0;
		}
	}
}

//高斯平滑化
void gaussian(Mat src,Mat dst){
	int mask[25] = {1,2,4,2,1,
					2,4,10,4,2,
					4,10,16,10,4,
					2,4,10,4,2,
					1,2,4,2,1,};
	int m = 2; //5x5
	int rows = src.rows;
	int cols = src.cols;
	int divisor=0;
	for(int i = 0;i< rows;i++)
	{
		for(int j = 0;j< cols; j++)
		{
			int sum = 0;
			int index = 0;

			for ( int y= i-m; y<= i+m; y++)
			{
				for (int x= j-m; x<= j+m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					divisor = divisor + mask[index];
					sum = sum + src.at<uchar>(y, x) * mask[index++] ;
				}
			}
			sum = sum/divisor;
			if(sum>255){sum=255;}
			dst.at<uchar>(i, j) = sum;
			divisor = 0;
		}
	}
}

//中值濾波
void median_flitering(Mat src, Mat dst){
	int temparray[25]; //5x5
	int m = 2;
	int rows = src.rows;
	int cols = src.cols;

	for(int i = 0;i< rows;i++)
	{
		for(int j = 0;j< cols; j++)
		{
			int index = 0;
			for ( int y= i-m; y<= i+m; y++)
			{
				for (int x= j-m; x<= j+m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					temparray[index++] = src.at<uchar>(y, x);
				}
			}
			quickSort(temparray,0,sizeof(temparray)/4-1);
			dst.at<uchar>(i, j) = temparray[sizeof(temparray)/8-1];
		}
	}
}

void quickSort(int *a,int left,int right){
	int pivot,i,j,temp; //pivot=基準點；temp=兩數互換用的暫存值
	pivot=a[left]; //基準值先等於第一個數字
	i=left; //比較比pivot小的數值用index
	j=right; //比較比pivot大的數值用index
	temp=0;

	if(left<right){
		while(i<j){
			i++;
			while(a[i]<pivot){
				i++;
			}
			while(a[j]>pivot){
				j--;
			}
			//找到不符合的值時，交換兩者位置 -> 左段小於基準值，右段大於基準值
			if(i<j){
				temp=a[j];
				a[j]=a[i];
				a[i]=temp;
			}
		}
		//將基準值換到已經分好大小兩邊時的中間去
		temp=a[j];
		a[j]=a[left];
		a[left]=temp;

		//利用遞迴繼續排序大小兩邊
		quickSort(a,left,j-1); //排序左段
		quickSort(a,j+1,right);//排序右段
	}
}

//波峰波谷濾波
void peak_and_valley_flitering(Mat src, Mat dst){
	int m = 2; //5x5
	int rows = src.rows;
	int cols = src.cols;

	for(int i = 0;i< rows;i++)
	{
		for(int j = 0;j< cols; j++)
		{
			int min = 256;
			int max = 0;
			for ( int y= i-m; y<= i+m; y++)
			{
				for (int x= j-m; x<= j+m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					if(x!=j && y!=i)
					{
						if(src.at<uchar>(y, x)>max)
						{
							max = src.at<uchar>(y, x);
						}
						if(src.at<uchar>(y, x)<min)
						{
							min = src.at<uchar>(y, x);
						}
					}


				}
			}
			if(src.at<uchar>(i, j)<=min)
			{
				dst.at<uchar>(i, j) = min;
			}
			else if(src.at<uchar>(i, j)>=max)
			{
				dst.at<uchar>(i, j) = max;
			}
			else{
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);
			}
		}
	}
}

//最大濾波
void max_flitering(Mat src, Mat dst){
	int m = 1; //3x3
	int rows = src.rows;
	int cols = src.cols;

	for(int i = 0;i< rows;i++)
	{
		for(int j = 0;j< cols; j++)
		{
			int max = 0;
			for ( int y= i-m; y<= i+m; y++)
			{
				for (int x= j-m; x<= j+m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					if(src.at<uchar>(y, x)> max)
					{
						max =src.at<uchar>(y, x);
					}
				}
			}
			dst.at<uchar>(i,j) = max ;
		}
	}
}

//最小濾波
void min_flitering(Mat src, Mat dst){
	int m = 1; //3x3
	int rows = src.rows;
	int cols = src.cols * src.channels();

	for(int i = 0;i< rows;i++)
	{
		for(int j = 0;j< cols; j++)
		{
			int min = 256;
			for ( int y= i-m; y<= i+m; y++)
			{
				for (int x= j-m; x<= j+m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					if(src.at<uchar>(y, x) < min)
					{
						min =src.at<uchar>(y, x);
					}
				}
			}
			dst.at<uchar>(i,j) = min ;
		}
	}
}

//空間距離差異
double spacedistance(int x1, int y1, int x2, int y2, double sigmaS) {
	double X = pow(abs(x1 - x2), 2);
	double Y = pow(abs(y1 - y2), 2);

	return exp(-(X + Y) / (2 * pow(sigmaS, 2)));
}

//灰階距離差異
double GSdistance(int g1, int g2, double sigmaG) {
	double X = pow(abs(g1 - g2), 2);
	return exp(-X / (2 * pow(sigmaG, 2)));
}

//雙側濾波器
void bilateralfilter(Mat src, Mat dst, double sigmaS, double sigmaG) { //雙側濾波器
	int m = 7; //15*15
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols; j++)
		{
			double k = 0;
			double f = 0;
			
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					f = f + src.at<uchar>(y, x)*spacedistance(i, j, y, x, sigmaS)*GSdistance(src.at<uchar>(i, j), src.at<uchar>(y, x), sigmaG);
					k = k + spacedistance(i, j, y, x, sigmaS)*GSdistance(src.at<uchar>(i, j), src.at<uchar>(y, x), sigmaG);
				}
			}
			int g = f / k;
			if (g < 0) g = 0;
			else if (g > 255) g = 255;
			dst.at<uchar>(i, j) = (uchar)g;
		}
	}
}

//雙側濾波器
void openMP_bilateralfilter(Mat src, Mat dst, double sigmaS, double sigmaG) { //雙側濾波器
	int m = 7; //15*15
	int rows = src.rows;
	int cols = src.cols;

	//#pragma omp parallel for collapse(4)
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double k = 0;
			double f = 0;

			#pragma omp parallel for collapse(2) reduction(+:f, k)
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					f = f + src.at<uchar>(y, x)*spacedistance(i, j, y, x, sigmaS)*GSdistance(src.at<uchar>(i, j), src.at<uchar>(y, x), sigmaG);
					k = k + spacedistance(i, j, y, x, sigmaS)*GSdistance(src.at<uchar>(i, j), src.at<uchar>(y, x), sigmaG);
				}
			}
			int g = f / k;
			if (g < 0) g = 0;
			else if (g > 255) g = 255;
			dst.at<uchar>(i, j) = (uchar)g;
		}
	}
}


//空間距離差異
double __device__ CUDA_spacedistance(int x1, int y1, int x2, int y2, double sigmaS) {
	int xx = abs(x1 - x2);
	int yy = abs(y1 - y2);
	double X = xx*xx;
	double Y = yy*yy;
	return exp(-(X + Y) / (2 * pow(sigmaS, 2)));
}

//灰階距離差異
double __device__ CUDA_GSdistance(int g1, int g2, double sigmaG) {
	int xx = abs(g1 - g2);
	double X = xx*xx;
	return exp(-X / (2 * pow(sigmaG, 2)));
}


void __global__ CUDA_bilateralfilter(uchar *d_src, uchar *d_dst, int rows, int cols, double sigmaS, double sigmaG) {
	int m = 7; //15*15
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < rows && j < cols)
	{
		//int idx = i * cols + j;
		//int A = d_src[idx];
		//d_dst[idx] = (uchar)exp((double)A);
		
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