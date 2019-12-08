// COMP_AS2_Q2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#include <stdio.h>






using namespace cv;
using namespace std;
#define NORMAL_HT

int edgeThresh = 100;
Mat image, gray, edge, cedge;




// define a trackbar callback
static void onTrackbar(int, void*)
{
}

int main(int argc, const char** argv)
{

	string filename = "track.jpg";

	image = imread(filename, 1);
	if (image.empty())
	{
		printf("Cannot read image file: %s\n", filename.c_str());
		return -1;
	}






	cedge.create(image.size(), image.type());
	cvtColor(image, gray, COLOR_BGR2GRAY);






	blur(gray, edge, Size(5, 5));

	// Run the edge detector on grayscale
	Canny(edge, edge, edgeThresh, edgeThresh * 3, 3);
	cedge = Scalar::all(0);

	image.copyTo(cedge, edge);
	imshow("Edge map", cedge);


	Mat dst, cdst;
	Canny(cedge, dst, 50, 200, 3);
	cvtColor(dst, cdst, COLOR_GRAY2BGR);

#ifdef NORMAL_HT
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 180, 100, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
#else
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
#endif
	imshow("source", image);
	imshow("detected lines", cdst);
	// Create a window
	namedWindow("Edge map", 1);

	
	// Wait for a key stroke; the same function arranges events processing
	waitKey(0);









	return 0;
}

