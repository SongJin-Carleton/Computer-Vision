// COMP4102_AS2_Q1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define W_XSIZE 200
#define W_YSIZE 300

using namespace cv;
using namespace std;

Mat src; 
Mat src_gray, warped_result;
Mat edge;
Mat speed_80, speed_40;
int canny_thresh = 120;


#define VERY_LARGE_VALUE 100000

#define NO_MATCH    0
#define STOP_SIGN            1
#define SPEED_LIMIT_40_SIGN  2
#define SPEED_LIMIT_80_SIGN  3



/** @function main */
int main(int argc, char** argv)
{
	int sign_recog_result = NO_MATCH;
	speed_40 = imread("speed_40.bmp", 1);
	speed_80 = imread("speed_80.bmp", 1);

	// you run your program on these three examples (uncomment the two lines below)
	string sign_name = "stop4";
	//string sign_name = "speedsign12";
	//string sign_name = "speedsign13";
	//string sign_name = "speedsign14";
	//string sign_name = "speedsign3";
	//string sign_name = "speedsign4";
	//string sign_name = "speedsign5";
	string final_sign_input_name = sign_name + ".jpg";
	string final_sign_output_name = sign_name + "_result" + ".jpg";

	/// Load source image and convert it to gray
	src = imread(final_sign_input_name, 1);

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(2, 2));


	imshow("show_gray", src_gray);

	// here you add the code to do the recognition, and set the variable 
	Mat canny_src;
	vector <vector<Point> > contoursFindOut;
	vector< Vec4i > hierarchy;

	
	//Canny function
	Canny(src_gray, edge, canny_thresh, canny_thresh * 2, 3);
	imshow("show_edge", edge);

	//got contours
	findContours(edge, contoursFindOut, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));
	cout << "show_gray";

	
	vector<Point> contoursFindOut_curve;
	vector<Point> largestcontoursFindOut_curve;

	bool stop_sign = false;
	bool speed_sign = false;


	int biggestArea = 0;
	for (int i = 0; i < contoursFindOut.size(); i = hierarchy[i][0]) {

		approxPolyDP(Mat(contoursFindOut[i]), contoursFindOut_curve, arcLength(Mat(contoursFindOut[i]), true)*0.02, true);


		if (isContourConvex(contoursFindOut_curve) && 
			fabs(contourArea(Mat(contoursFindOut_curve))) > biggestArea)
		{

			largestcontoursFindOut_curve = contoursFindOut_curve;
			biggestArea = fabs(contourArea(Mat(contoursFindOut_curve)));

		}
	}
	
	if (largestcontoursFindOut_curve.size() == 8) {
		stop_sign = true;
		
	}
	if (largestcontoursFindOut_curve.size() == 4) {
		speed_sign = true;
	}




	
	if (stop_sign == true) {

		sign_recog_result = STOP_SIGN;

	}else if (speed_sign == true) {


		Point2f point_tl;
		Point2f point_tr;
		Point2f point_bl;
		Point2f point_br;

		int small_sum = 999999;
		int large_sum = 0;
		int small_diff = 999999;
		int large_diff = 0;



		for (int i = 0; i < 4; i++) {
			if (largestcontoursFindOut_curve[i].x + largestcontoursFindOut_curve[i].y < small_sum) {
				point_tl = largestcontoursFindOut_curve[i];
				cout << "\n  point_tl";
				small_sum = largestcontoursFindOut_curve[i].x + largestcontoursFindOut_curve[i].y;
			}

			if (largestcontoursFindOut_curve[i].x + largestcontoursFindOut_curve[i].y > large_sum) {
				point_br = largestcontoursFindOut_curve[i];
				cout << "\n  point_br";
				large_sum = largestcontoursFindOut_curve[i].x + largestcontoursFindOut_curve[i].y;
			}

			if (largestcontoursFindOut_curve[i].x - largestcontoursFindOut_curve[i].y < small_diff) {
				point_bl = largestcontoursFindOut_curve[i];
				cout << "\n  point_bl";
				small_diff = largestcontoursFindOut_curve[i].x - largestcontoursFindOut_curve[i].y;
			}

			if (largestcontoursFindOut_curve[i].x - largestcontoursFindOut_curve[i].y > large_diff) {
				point_tr = largestcontoursFindOut_curve[i];
				cout << "\n  point_tr";
				large_diff = largestcontoursFindOut_curve[i].x - largestcontoursFindOut_curve[i].y;
			}
			cout << "\n  small_sum: " << small_sum  << "\n  large_sum: " << large_sum  << "\n  ";
			cout << "\n  small_diff:" << small_diff << "\n  large_diff: " << large_diff << "\n  ";
		}

		Point2f source_image[4];

		source_image[0] = point_tl;
		source_image[1] = point_tr;
		source_image[2] = point_br;
		source_image[3] = point_bl;




		//Point2f dst_p[4];
		Point2f destination_image[4];


		destination_image[0] = cv::Point2f(0.0f, 0.0f);
		destination_image[1] = cv::Point2f(W_XSIZE, 0.0f);
		destination_image[2] = cv::Point2f(W_XSIZE, W_YSIZE);
		destination_image[3] = cv::Point2f(0.0f, W_YSIZE);



		Mat M;
		Mat warp_output;
		Mat matchResult40;
		Mat matchResult80;

		M = getPerspectiveTransform(source_image, destination_image);



		warpPerspective(src, warp_output, M, Size(W_XSIZE, W_YSIZE));

		imshow("warp_output", warp_output);

		matchTemplate(speed_40, warp_output, matchResult40, TM_CCOEFF_NORMED);

		matchTemplate(speed_80, warp_output, matchResult80, TM_CCOEFF_NORMED);

		double min_40; 
		double max_40; 
		Point minLocation40; 
		Point maxLocation40;


		minMaxLoc(matchResult40, &min_40, &max_40, &minLocation40, &maxLocation40, Mat());

		double min_80; 
		double max_80; 
		Point minLocation80; 
		Point maxLocation80;


		minMaxLoc(matchResult80, &min_80, &max_80, &minLocation80, &maxLocation80, Mat());


		cout << "\n max_40: \n" << max_40 << "\n max_80: \n" << max_80  << "\n ";


		cout << fabs(fabs(fabs(max_40) - fabs(max_80)) / max_80);

		if ((max_40 > max_80) && 
			(fabs(fabs(fabs(max_40) - fabs(max_80)) / max_80) <= 0.7)) {
			sign_recog_result = NO_MATCH;
		}
		if ((max_80 > max_40) && 
			(fabs(fabs(fabs(max_40) - fabs(max_80)) / max_40) >= 0.3)) {
			sign_recog_result = NO_MATCH;
		}
		else if (max_40 > max_80) {
			sign_recog_result = SPEED_LIMIT_40_SIGN;
		}
		else {
			sign_recog_result = SPEED_LIMIT_80_SIGN;
		}
			
	}

	else sign_recog_result = NO_MATCH;




	string text;
	if (sign_recog_result == SPEED_LIMIT_40_SIGN) text = "Speed 40";
	else if (sign_recog_result == SPEED_LIMIT_80_SIGN) text = "Speed 80";
	else if (sign_recog_result == STOP_SIGN) text = "Stop";
	else if (sign_recog_result == NO_MATCH) text = "Fail";

	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 2;
	int thickness = 3;
	cv::Point textOrg(10, 130);
	cv::putText(src, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

	/// Create Window
	string source_window = "Result";
	namedWindow(source_window, WINDOW_AUTOSIZE);
	imshow(source_window, src);
	imwrite(final_sign_output_name, src);

	waitKey(0);

	return(0);
}
