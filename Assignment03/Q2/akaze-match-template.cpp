// COMP4102_A3Q2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(void)
{
	puts("opening");
	Mat img1 = imread("keble_a_half.bmp", IMREAD_GRAYSCALE);
	Mat img2 = imread("keble_b_long.bmp", IMREAD_GRAYSCALE);
	Mat img3 = Mat(img2.rows, img2.cols, CV_8UC1);
	//img2.copyTo(img3);

	Mat homography;

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;
	puts("Have opened");

	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	puts("have commputed akaze");

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);
	puts("Have done match");

	vector<Point2f> matched1, matched2;
	vector<Point2f> inliers1, inliers2;

	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx].pt);
			matched2.push_back(kpts2[first.trainIdx].pt);
		}
	}
	printf("Matches %d %d\n", matched1.size(), matched2.size());


	
	homography = findHomography(matched1, matched2, RANSAC);
	warpPerspective(img1, img3, homography, img3.size());

	
	for (int m = 0; m < img3.size().height; ++m) {
		for (int n = 0; n < img3.size().width; ++n) {
			img2.at<uchar>(m, n) = (img2.at<uchar>(m, n) | img3.at<uchar>(m, n));
		}
	}
	

	//Display input and output
	imshow("Input1_merge", img2);
	imshow("Input2_warp", img3);
	//output warped and merged image to file
	imwrite("merged.jpg", img2);
	imwrite("warped.jpg", img3);
	waitKey(0);

	return 0;
}