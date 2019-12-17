#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

#include <sstream>

using namespace std;
using namespace cv;

/*fast version*/
void fast_bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG);


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
			string name = filenames[i].substr(filenames[i].find_last_of('\\') + 1).operator string();
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


void fast_bilateralFilter(Mat src, Mat dst, double sigmaS, double sigmaG)
{
	double gauss_color_coeff = -0.5 / (sigmaG*sigmaG);
	double gauss_space_coeff = -0.5 / (sigmaS*sigmaS);
	int maxk = 0;
	Size size = src.size();

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

	// initialize color-related bilateral filter coefficients

	for (int i = 0; i < 256; i++) {
		color_weight[i] = (float)exp(i*i*gauss_color_coeff);
	}

	// initialize space-related bilateral filter coefficients

	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++)
		{
			double r = sqrt((double)i*i + (double)j*j);  //sqrt不可拆
			if (r > radius)
				continue;
			space_weight[maxk] = (float)exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*temp.step + j);
			maxk++;
		}
	}

	for (int i = 0; i < size.height; i++)
	{
		for (int j = 0; j < size.width; j++)
		{
			const uchar* sptr = temp.data + (i + radius)*temp.step + radius;
			uchar* dptr = dst.data + i * dst.step;

			float sum = 0, wsum = 0;
			int val0 = sptr[j];

			for (int k = 0; k < maxk; k++)
			{
				int val = sptr[j + space_ofs[k]];
				float w = space_weight[k] * color_weight[abs(val - val0)];
				sum += val * w;
				wsum += w;
			}
			// overflow is not possible here => there is no need to use CV_CAST_8U
			dptr[j] = (uchar)cvRound(sum / wsum);
		}
	}
}