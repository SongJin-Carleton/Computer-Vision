// COMP4102_A3Q1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <string>
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include "opencv2/calib3d/calib3d.hpp"


#include <math.h>


#define NUM_POINTS    10
#define RANGE         100.00

#define MAX_CAMERAS   100 
#define MAX_POINTS    3000


float projection[3][4] = {
	0.902701, 0.051530, 0.427171, 12.0,
	0.182987, 0.852568, -0.489535, 16.0,
	-0.389418, 0.520070, 0.760184, 21.0,
};

float intrinsic[3][3] = {
	-1100.000000, 0.000000, 0.000000,
	0.000000, -2200.000000, 0.000000,
	0.000000, 0.000000,     1.000000,
};

float all_object_points[10][3] = {
	0.1251, 56.3585, 19.3304,
	80.8741, 58.5009, 47.9873,
	35.0291, 89.5962, 82.2840,
	74.6605, 17.4108, 85.8943,
	71.0501, 51.3535, 30.3995,
	1.4985, 9.1403, 36.4452,
	14.7313, 16.5899, 98.8525,
	44.5692, 11.9083, 0.4669,
	0.8911, 37.7880, 53.1663,
	57.1184, 60.1764, 60.7166
};

// you write this routine
void decomposeprojectionmatrix(CvMat* projection_matrix, CvMat* rotation_matrix, CvMat* translation, CvMat* camera_matrix){

	CvMat* projMatrix = cvCreateMat(3, 4, CV_32F);

	float summy = pow(cvmGet(projection_matrix, 2, 0), 2) + pow(cvmGet(projection_matrix, 2, 1), 2) + pow(cvmGet(projection_matrix, 2, 2), 2);
	float element = sqrt(summy);


	for (int m = 0; m < 3; m++)
	{
		for (int n = 0; n < 4; n++)
		{
			cvmSet(projMatrix, m, n, cvmGet(projection_matrix, m, n) / element);
		}
	}



	float symbol;
	if (cvmGet(projection_matrix, 2, 3) < 0)
	{
		symbol = -1;
	}else{
		symbol = 1;
	}



	float Tz = symbol * cvmGet(projMatrix, 2, 3);



	for (int m = 0; m < 3; m++)
	{
		cvmSet(rotation_matrix, 2, m, symbol * cvmGet(projMatrix, 2, m));
	}

	float Ox = ((cvmGet(projection_matrix, 0, 0) * cvmGet(projection_matrix, 2, 0)) + (cvmGet(projection_matrix, 0, 1) * cvmGet(projection_matrix, 2, 1)) + (cvmGet(projection_matrix, 0, 2) * cvmGet(projection_matrix, 2, 2)));
	float Oy = ((cvmGet(projection_matrix, 1, 0) * cvmGet(projection_matrix, 2, 0)) + (cvmGet(projection_matrix, 1, 1) * cvmGet(projection_matrix, 2, 1)) + (cvmGet(projection_matrix, 1, 2) * cvmGet(projection_matrix, 2, 2)));
	float fx = sqrt((pow(cvmGet(projMatrix, 0, 0), 2) + pow(cvmGet(projMatrix, 0, 1), 2) + pow(cvmGet(projMatrix, 0, 2), 2)) - pow(Ox, 2));
	float fy = sqrt((pow(cvmGet(projMatrix, 1, 0), 2) + pow(cvmGet(projMatrix, 1, 1), 2) + pow(cvmGet(projMatrix, 1, 2), 2)) - pow(Oy, 2));



	for (int m = 0; m < 3; m++)
	{
		cvmSet(rotation_matrix, 0, m, symbol * (Ox * cvmGet(projMatrix, 2, m) - cvmGet(projMatrix, 0, m)) / fx);
		cvmSet(rotation_matrix, 1, m, symbol * (Oy * cvmGet(projMatrix, 2, m) - cvmGet(projMatrix, 1, m)) / fy);
	}

	float Tx = (symbol * ((Ox * Tz) - cvmGet(projMatrix, 0, 3)) / fx);
	float Ty = (symbol * ((Oy * Tz) - cvmGet(projMatrix, 1, 3)) / fy);

	cvmSet(translation, 0, 0, Tx);
	cvmSet(translation, 1, 0, Ty);
	cvmSet(translation, 2, 0, Tz);

	cvmSet(camera_matrix, 0, 0, -fx);
	cvmSet(camera_matrix, 1, 1, -fy);
	cvmSet(camera_matrix, 0, 2, Ox);
	cvmSet(camera_matrix, 1, 2, Oy);
	cvmSet(camera_matrix, 2, 2, 1);
}




// you write this routine
void computeprojectionmatrix(CvMat* image_points, CvMat* object_points, CvMat* projection_matrix){

	CvMat* evects = cvCreateMat(12, 12, CV_64F);
	CvMat* evals = cvCreateMat(12, 1, CV_64F);

	CvMat* src = cvCreateMat(NUM_POINTS * 2, 12, CV_32F);
	CvMat* dst = cvCreateMat(12, NUM_POINTS * 2, CV_32F);
	CvMat* outPutMat = cvCreateMat(12, 12, CV_32F);

	int point = 0;
	for (int i = 0; i < (NUM_POINTS * 2); i += 2, point++)
	{
		float Xi = cvmGet(object_points, point, 0);
		float Yi = cvmGet(object_points, point, 1);
		float Zi = cvmGet(object_points, point, 2);
		float xi = cvmGet(image_points, point, 0);
		float yi = cvmGet(image_points, point, 1);



		cvmSet(src, i, 0, Xi);
		cvmSet(src, i, 1, Yi);
		cvmSet(src, i, 2, Zi);
		cvmSet(src, i, 3, 1);
		cvmSet(src, i, 4, 0);
		cvmSet(src, i, 5, 0);
		cvmSet(src, i, 6, 0);
		cvmSet(src, i, 7, 0);
		cvmSet(src, i, 8, -(xi * Xi));
		cvmSet(src, i, 9, -(xi * Yi));
		cvmSet(src, i, 10, -(xi * Zi));
		cvmSet(src, i, 11, -xi);



		cvmSet(src, i + 1, 0, 0);
		cvmSet(src, i + 1, 1, 0);
		cvmSet(src, i + 1, 2, 0);
		cvmSet(src, i + 1, 3, 0);
		cvmSet(src, i + 1, 4, Xi);
		cvmSet(src, i + 1, 5, Yi);
		cvmSet(src, i + 1, 6, Zi);
		cvmSet(src, i + 1, 7, 1);
		cvmSet(src, i + 1, 8, -(yi * Xi));
		cvmSet(src, i + 1, 9, -(yi * Yi));
		cvmSet(src, i + 1, 10, -(yi * Zi));
		cvmSet(src, i + 1, 11, -yi);
	}



	cvTranspose(src, dst);
	cvMatMul(dst, src, outPutMat);
	cvEigenVV(outPutMat, evects, evals);



	int index = 0;
	for (int m = 0; m < 3; m++)
	{
		for (int n = 0; n < 4; n++)
		{
			cvmSet(projection_matrix, m, n, cvmGet(evects, 11, index++));
		}
	}



	printf("Projection Matrix\n");
	for (int m = 0; m < 3; m++)
	{
		for (int n = 0; n < 4; n++)
		{
			printf("%f ", cvmGet(projection_matrix, m, n));
		}
		printf("\n");
	}
	printf("\n");
}

int main() {
	CvMat* camera_matrix, *computed_camera_matrix; 
	CvMat* rotation_matrix, *computed_rotation_matrix; 
	CvMat* translation, *computed_translation;
	CvMat* image_points, *transp_image_points;
	CvMat* rot_vector;
	CvMat* object_points, *transp_object_points;
	CvMat* computed_projection_matrix;
	CvMat *final_projection;
	CvMat temp_projection, temp_intrinsic;
	FILE *fp;

	cvInitMatHeader(&temp_projection, 3, 4, CV_32FC1, projection);
	cvInitMatHeader(&temp_intrinsic, 3, 3, CV_32FC1, intrinsic);

	final_projection = cvCreateMat(3, 4, CV_32F);

	object_points = cvCreateMat(NUM_POINTS, 4, CV_32F);
	transp_object_points = cvCreateMat(4, NUM_POINTS, CV_32F);

	image_points = cvCreateMat(NUM_POINTS, 3, CV_32F);
	transp_image_points = cvCreateMat(3, NUM_POINTS, CV_32F);

	rot_vector = cvCreateMat(3, 1, CV_32F);
	camera_matrix = cvCreateMat(3, 3, CV_32F);
	rotation_matrix = cvCreateMat(3, 3, CV_32F);
	translation = cvCreateMat(3, 1, CV_32F);

	computed_camera_matrix = cvCreateMat(3, 3, CV_32F);
	computed_rotation_matrix = cvCreateMat(3, 3, CV_32F);
	computed_translation = cvCreateMat(3, 1, CV_32F);
	computed_projection_matrix = cvCreateMat(3, 4, CV_32F);

	fp = fopen("assign3-out", "w");

	fprintf(fp, "Rotation matrix\n");
	for (int i = 0; i<3; i++) {
		for (int j = 0; j<3; j++) {
			cvmSet(camera_matrix, i, j, intrinsic[i][j]);
			cvmSet(rotation_matrix, i, j, projection[i][j]);
		}
		fprintf(fp, "%f %f %f\n",
			cvmGet(rotation_matrix, i, 0), cvmGet(rotation_matrix, i, 1), cvmGet(rotation_matrix, i, 2));
	}
	for (int i = 0; i<3; i++)
		cvmSet(translation, i, 0, projection[i][3]);

	fprintf(fp, "\nTranslation vector\n");
	fprintf(fp, "%f %f %f\n",
		cvmGet(translation, 0, 0), cvmGet(translation, 1, 0), cvmGet(translation, 2, 0));

	fprintf(fp, "\nCamera Calibration\n");
	for (int i = 0; i<3; i++) {
		fprintf(fp, "%f %f %f\n",
			cvmGet(camera_matrix, i, 0), cvmGet(camera_matrix, i, 1), cvmGet(camera_matrix, i, 2));
	}

	fprintf(fp, "\n");
	for (int i = 0; i < NUM_POINTS; i++) {
		cvmSet(object_points, i, 0, all_object_points[i][0]);
		cvmSet(object_points, i, 1, all_object_points[i][1]);
		cvmSet(object_points, i, 2, all_object_points[i][2]);
		cvmSet(object_points, i, 3, 1.0);
		fprintf(fp, "Object point %d x %f y %f z %f\n",
			i, all_object_points[i][0], all_object_points[i][1], all_object_points[i][2]);
	}
	fprintf(fp, "\n");
	cvTranspose(object_points, transp_object_points);

	cvMatMul(&temp_intrinsic, &temp_projection, final_projection);
	cvMatMul(final_projection, transp_object_points, transp_image_points);
	//cvTranspose(transp_image_points, image_points);


	for (int i = 0; i<NUM_POINTS; i++) {
		cvmSet(image_points, i, 0, cvmGet(transp_image_points, 0, i) / cvmGet(transp_image_points, 2, i));
		cvmSet(image_points, i, 1, cvmGet(transp_image_points, 1, i) / cvmGet(transp_image_points, 2, i));
		fprintf(fp, "Image point %d x %f y %f\n",
			i, cvmGet(image_points, i, 0), cvmGet(image_points, i, 1));
	}

	computeprojectionmatrix(image_points, object_points, computed_projection_matrix);
	decomposeprojectionmatrix(computed_projection_matrix, computed_rotation_matrix, computed_translation, computed_camera_matrix);

	fprintf(fp, "\nComputed Rotation matrix\n");
	for (int i = 0; i<3; i++) {
		fprintf(fp, "%f %f %f\n",
			cvmGet(computed_rotation_matrix, i, 0), cvmGet(computed_rotation_matrix, i, 1), cvmGet(computed_rotation_matrix, i, 2));
	}

	fprintf(fp, "\nComputed Translation vector\n");
	fprintf(fp, "%f %f %f\n",
		cvmGet(computed_translation, 0, 0), cvmGet(computed_translation, 1, 0), cvmGet(computed_translation, 2, 0));

	fprintf(fp, "\nComputed Camera Calibration\n");
	for (int i = 0; i<3; i++) {
		fprintf(fp, "%f %f %f\n",
			cvmGet(computed_camera_matrix, i, 0), cvmGet(computed_camera_matrix, i, 1), cvmGet(computed_camera_matrix, i, 2));
	}

	fclose(fp);
	getchar();
	return 0;
}


