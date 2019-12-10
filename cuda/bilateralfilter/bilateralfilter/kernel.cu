#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
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
void colorchange(Mat src, Mat dst);
double spacedistance(int x1, int y1, int x2, int y2, double sigmaS);
double GSdistance(int g1, int g2, double sigmaG);
void bilateralfilter(Mat src, Mat dst, double sigmaS, double sigmaG);

int main() {
	Mat src = imread("freckle.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst(src.rows, src.cols, CV_8U);
	Mat dst2(src.rows, src.cols, CV_8U);
	Mat dst3(src.rows, src.cols, CV_8U);
	Mat dst4(src.rows, src.cols, CV_8U);
	Mat dst5(src.rows, src.cols, CV_8U);

	unsigned long start = clock();
	neighborhood_averaging(src, dst);
	unsigned long end = clock();
	cout << "neigbor total time=" << (end - start) / 1000.0 << "seconds" << endl;

	start = clock();
	gaussian(src, dst2);
	end = clock();
	cout << "guassian total time=" << (end - start) / 1000.0 << "seconds" << endl;

	start = clock();
	median_flitering(src, dst3);
	end = clock();
	cout << "median total time=" << (end - start) / 1000.0 << "seconds" << endl;

	start = clock();
	peak_and_valley_flitering(src, dst4);
	end = clock();
	cout << "peak_and_valley total time=" << (end - start) / 1000.0 << "seconds" << endl;


	start = clock();
	//dim3 blocksPerGrid(512, 1, 1)
	//dim3 threadsPerBlock(512, 1, 1)
	//bilateralfilter << <blocksPerGrid, threadsPerBlock >> > ()
	bilateralfilter(src, dst5, 13, 13);
	end = clock();
	cout << "bilateralfilter total time=" << (end - start) / 1000.0 << "seconds" << endl;

	//�̤j�̤p�o�i�b�̤p�̤j�o�i
	/*max_flitering(src,dst);
	min_flitering(dst,dst2);
	min_flitering(dst2,dst3);
	max_flitering(dst3,dst4);*/


	imshow("orgin", src);
	imshow("neigbor", dst);
	imshow("guassian", dst2);
	imshow("median", dst3);
	imshow("peak_and_valley", dst4);
	imshow("bilateralfilter", dst5);

	imwrite("neigbor.jpg", dst);
	imwrite("guassian.jpg", dst2);
	imwrite("median.jpg", dst3);
	imwrite("peak_and_valley.jpg", dst4);
	imwrite("bilateralfilter.jpg", dst5);


	waitKey(0);
	return(0);
}

//�۾F���������k
void neighborhood_averaging(Mat src, Mat dst) {
	int mask[25] = { 1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1 };
	int divisor = 0;
	int m = 2;
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int sum = 0;
			int index = 0;
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					divisor = divisor + mask[index];
					sum = sum + src.at<uchar>(y, x) * mask[index++];
				}
			}
			sum = sum / divisor;
			if (sum > 255) { sum = 255; }
			dst.at<uchar>(i, j) = sum;
			divisor = 0;
		}
	}
}

//�������Ƥ�
void gaussian(Mat src, Mat dst) {
	int mask[25] = { 1,2,4,2,1,
					2,4,10,4,2,
					4,10,16,10,4,
					2,4,10,4,2,
					1,2,4,2,1, };
	int m = 2; //5x5
	int rows = src.rows;
	int cols = src.cols;
	int divisor = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int sum = 0;
			int index = 0;

			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					divisor = divisor + mask[index];
					sum = sum + src.at<uchar>(y, x) * mask[index++];
				}
			}
			sum = sum / divisor;
			if (sum > 255) { sum = 255; }
			dst.at<uchar>(i, j) = sum;
			divisor = 0;
		}
	}
}

//�����o�i
void median_flitering(Mat src, Mat dst) {
	int temparray[25]; //5x5
	int m = 2;
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int index = 0;
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					temparray[index++] = src.at<uchar>(y, x);
				}
			}
			quickSort(temparray, 0, sizeof(temparray) / 4 - 1);
			dst.at<uchar>(i, j) = temparray[sizeof(temparray) / 8 - 1];
		}
	}
}

void quickSort(int *a, int left, int right) {
	int pivot, i, j, temp; //pivot=����I�Ftemp=��Ƥ����Ϊ��Ȧs��
	pivot = a[left]; //��ǭȥ�����Ĥ@�ӼƦr
	i = left; //�����pivot�p���ƭȥ�index
	j = right; //�����pivot�j���ƭȥ�index
	temp = 0;

	if (left < right) {
		while (i < j) {
			i++;
			while (a[i] < pivot) {
				i++;
			}
			while (a[j] > pivot) {
				j--;
			}
			//��줣�ŦX���ȮɡA�洫��̦�m -> ���q�p���ǭȡA�k�q�j���ǭ�
			if (i < j) {
				temp = a[j];
				a[j] = a[i];
				a[i] = temp;
			}
		}
		//�N��ǭȴ���w�g���n�j�p����ɪ������h
		temp = a[j];
		a[j] = a[left];
		a[left] = temp;

		//�Q�λ��j�~��ƧǤj�p����
		quickSort(a, left, j - 1); //�Ƨǥ��q
		quickSort(a, j + 1, right);//�Ƨǥk�q
	}
}

//�i�p�i���o�i
void peak_and_valley_flitering(Mat src, Mat dst) {
	int m = 2; //5x5
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int min = 256;
			int max = 0;
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					if (x != j && y != i)
					{
						if (src.at<uchar>(y, x) > max)
						{
							max = src.at<uchar>(y, x);
						}
						if (src.at<uchar>(y, x) < min)
						{
							min = src.at<uchar>(y, x);
						}
					}


				}
			}
			if (src.at<uchar>(i, j) <= min)
			{
				dst.at<uchar>(i, j) = min;
			}
			else if (src.at<uchar>(i, j) >= max)
			{
				dst.at<uchar>(i, j) = max;
			}
			else {
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);
			}
		}
	}
}

//�̤j�o�i
void max_flitering(Mat src, Mat dst) {
	int m = 1; //3x3
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int max = 0;
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					if (src.at<uchar>(y, x) > max)
					{
						max = src.at<uchar>(y, x);
					}
				}
			}
			dst.at<uchar>(i, j) = max;
		}
	}
}

//�̤p�o�i
void min_flitering(Mat src, Mat dst) {
	int m = 1; //3x3
	int rows = src.rows;
	int cols = src.cols * src.channels();

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int min = 256;
			for (int y = i - m; y <= i + m; y++)
			{
				for (int x = j - m; x <= j + m; x++)
				{
					if (y < 0 || x < 0 || y >= rows || x >= cols)
					{
						continue;
					}
					if (src.at<uchar>(y, x) < min)
					{
						min = src.at<uchar>(y, x);
					}
				}
			}
			dst.at<uchar>(i, j) = min;
		}
	}
}

//�Ŷ��Z���t��
//��ӹ��������Z��
double spacedistance(int x1, int y1, int x2, int y2, double sigmaS) {
	double X = pow(abs(x1 - x2), 2);
	double Y = pow(abs(y1 - y2), 2);

	return exp(-(X + Y) / (2 * pow(sigmaS, 2)));
}

//�Ƕ��Z���t��
//�ھڨ�ۦ��{�� ��ӹ������Ȥ������Z��
double GSdistance(int g1, int g2, double sigmaG) {
	double X = pow(abs(g1 - g2), 2);
	return exp(-X / (2 * pow(sigmaG, 2)));
}

//�����o�i��
// sigmaS = sigmaG = 13
void bilateralfilter(Mat src, Mat dst, double sigmaS, double sigmaG) { //�����o�i��
	int m = 7; //15*15
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//int i = blockIdx.x * blockDim.x + threadIdx.x;
			//int j = blockIdx.y * blockDim.y + threadIdx.y;
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
					// src.at<uchar>(column, row) �]�� MAT �O column-major
					// CV_8U �Ω�8��1�q�D�ǫ׹Ϲ�
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
