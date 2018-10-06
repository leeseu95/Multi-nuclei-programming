//Seung Lee - A01021720
//Matrix Mult CPU (No threads)
//g++ -o MatrixMultOMP matrix_mult_cpu_omp.cpp -std=c++11
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <iomanip> 
#include <string>
#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include <cuda_runtime.h>

using namespace std;

void fillMat(float * ip, const int size) { //Funcion para llenar nuestras matrices (hecho como el ejemplo en clase matrix_sum_1d)
    for(int i = 0; i < size; i++) {
        ip[i] = i;
    }
}

void multMat(float *A, float *B, float *C, const int nx, const int ny) { //Funcion para multiplicar matriz (como ejemplo)
    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < ny; k++) { //Regla del karatazo pu pi pao
                C[i * nx + j] += (A[i * nx + k] * B[k *nx + i]);
                // printf("G"); //Debug
            }
        }
    }
}

int main() {
string imagePath;
	
    //Si no hay parametro utilizamos image.jpg
	if(argc < 2)
		imagePath = "image.jpg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	//Call the wrapper function
	convert_to_gray(input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
