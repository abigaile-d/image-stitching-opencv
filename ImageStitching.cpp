/**************************************************
* Abigaile Dionisio
* Programming Assignment 2: Image Stitching
* Submission Date: May 1, 2015
**************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv.h>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{

///// LOADING IMAGES AND GETTING USER INPUT

	Mat left_image, right_image, left_gray, right_gray;
	bool use_affine = true;

	string left_filename = "imgpairs\\tower_left.jpg";
	string right_filename = "imgpairs\\tower_right.jpg";

	do {
		//open the images
		left_image = imread(left_filename, CV_LOAD_IMAGE_COLOR);
		right_image = imread(right_filename, CV_LOAD_IMAGE_COLOR);
		
		//if default images not present, ask user to input filenames
		//do this until correct files entered
		if (!left_image.data || !right_image.data){
			cout << "Cannot open or find the images!" << endl;
			cout << "Please type the filename of the left image: ";
			getline(cin, left_filename);
			cout << "Please type the filename of the right image: ";
			getline(cin, right_filename);
		}
	} while (!left_image.data || !right_image.data);

	//display menu for the transformation method to use later
	//ask user if affine or homography
	char choice;
	do{
		cout << endl << endl;
		cout << "Select transformation method to use:" << endl;
		cout << "[1] AFFINE TRANSFORMATION" << endl;
		cout << "[2] HOMOGRAPHY MAPPING" << endl;
		cout << "[0] Exit" << endl;
		cin >> choice;
		cout << endl << endl;;

		system("CLS");
		switch (choice){
		case '0':
			break;
		case '1':
			use_affine = true;
			break;
		case '2':
			use_affine = false;
			break;
		default:
			cout << choice << "is not in the menu!" << endl;
		}
	} while (choice != '0' && choice != '1' && choice != '2');

	if (choice == '0')
		return EXIT_SUCCESS;

	//convert images to greyscale for harris detector processing
	cvtColor(left_image, left_gray, COLOR_BGR2GRAY);
	cvtColor(right_image, right_gray, COLOR_BGR2GRAY);



///// DETECTING FEATURE POINTS AND EXTRACTING PATCHES: LEFT IMAGE!

	//parameters and variables used in Harris detector
	int blocksize = 7;
	int aperturesize = 5;
	double k = 0.05;
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(left_image.size(), CV_32FC1);

	//detecting corners
	cornerHarris(left_gray, dst, blocksize, aperturesize, k, BORDER_DEFAULT);

	//normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(dst_norm, dst_norm_scaled);
	
	//parameters and variables used for extracting patches
	int harris_thresh = 100;							//threshold (intensity value) for detecting keypoints, best value obtained experimentally
	int patchsize = 17;									//patchsize, best value obtained experimentally
	Vec3b pixel;
	vector<Mat> left_patches, right_patches;			//contains BGR values of descriptors
	vector<Point2i> left_keypoints, right_keypoints;	//contains x,y coords of descriptors (center)

	//extracting patches
	//loop through all the pixels in the post-harris gray image, and get patches based on the points exceeding the threshold
	for (int row = 0; row < dst_norm.rows; row++) {
		for (int col = 0; col < dst_norm.cols; col++) {

			//check if exceeded the intensity threshold
			if ((int)dst_norm.at<float>(row, col) > harris_thresh) {
				
				//marking Harris image
				//circle(dst_norm_scaled, Point(col,row), 10, Scalar(0), 1, CV_AA, 0);

				//get patch of size = patchsize
				Mat patch = Mat::zeros(patchsize, patchsize, CV_8UC3);
				getRectSubPix(left_image, Size(patchsize, patchsize), Point2f(col, row), patch);

				//convert the patch color values from 3Ch NxN Mat to 1Ch 1row Mat and add to vector of Mat
				Mat patch_vec = patch.reshape(1, 1);
				left_patches.push_back(patch_vec);
				//add x,y coords of keypoints to vector of Points
				left_keypoints.push_back(Point(col, row));
			}
		}
	}

	

///// DETECTING FEATURE POINTS AND EXTRACTING PATCHES: RIGHT IMAGE!
	
	//detecting corners
	cornerHarris(right_gray, dst, blocksize, aperturesize, k, BORDER_DEFAULT);

	//normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(dst_norm, dst_norm_scaled);
	
	//extracting patches
	//same as the procedure as above
	for (int row = 0; row < dst_norm.rows; row++) {
		for (int col = 0; col < dst_norm.cols; col++) {

			if ((int)dst_norm.at<float>(row, col) > harris_thresh) {
				Mat patch = Mat::zeros(patchsize, patchsize, CV_8UC3);
				getRectSubPix(right_image, Size(patchsize, patchsize), Point2f(col, row), patch);

				Mat patch_vec = patch.reshape(1, 1);
				right_patches.push_back(patch_vec);
				right_keypoints.push_back(Point(col, row));
			}
		}
	}

	

///// GETTING LEFT AND RIGHT KEYPOINT MATCHES BASED ON EUCLIDEAD DISTANCE

	//variables for getting euclidean distance
	vector<Vec3d> matches;
	Vec3d match;
	double currdist;

	//variables for getting putative matches
	vector<Point2f> left_good_keypoints, right_good_keypoints;
	int dist_thresh = 1700;		//euc distance threshold, best value obtained experimentally

	//iterating through list of left keypoint patches and right keypoint patches
	for (int i = 0; i < left_patches.size(); i++){

		//distance, left index, right index
		match = { FLT_MAX, 0, 0 };
		for (int j = 0; j < right_patches.size(); j++){

			//compute norm_l2 = euclidean distance for curr left to curr right descriptor pair
			currdist = norm(left_patches[i], right_patches[j], NORM_L2);

			//if the distance is the smallest yet for the current left descriptor
			//save the distance, index of the current left descriptor, and index of the best right match so far
			if (currdist < match[0]){
				match = { currdist, (double)i, (double)j };
			}
		}
		//at the end of the inner for loop, will get the best right match for the current left descriptor
		
		//check if the distance is below the threshold
		//if yes, save them as "good" keypoints
		if (match[0] < dist_thresh){
			left_good_keypoints.push_back(left_keypoints[match[1]]);
			right_good_keypoints.push_back(right_keypoints[match[2]]);

		}
	}

	

///// TRANSFORMING THE RIGHT IMAGE FOR STITCHING

	//finding transformation using RANSAC
	//right image is the object (to be projected) and left image is the scene
	//flag if point is inlier or outlier is returned in mask Mat
	Mat mask, transformation;
	if (use_affine){
		//only function that i can find for affine using RANSAC with inlier reporting
		//function works on 3D points so need to convert to homogeneous coords first
		vector<Point3f> right_3d_keypoints, left_3d_keypoints;
		convertPointsToHomogeneous(left_good_keypoints, left_3d_keypoints);
		convertPointsToHomogeneous(right_good_keypoints, right_3d_keypoints);
		estimateAffine3D(right_3d_keypoints, left_3d_keypoints, transformation, mask, 2);

		//result is 3x4 matrix
		//3rd+4th column since 4th col is only translation vector
		//remove 4th col
		transformation.col(2) = transformation.col(2) * 2;
		transformation = transformation.colRange(0, 3);
		
		cout << endl;
		cout << "AFFINE MATRIX:" << endl;
		cout << transformation << endl;
	}
	else{
		transformation = findHomography(right_good_keypoints, left_good_keypoints, CV_RANSAC, 4, mask);

		cout << endl;
		cout << "HOMOGRAPHY MATRIX:" << endl;
		cout << transformation << endl;
	}

	//Mat for displaying matches, need the original left and right image place beside each other
	RNG rng(12345);
	Mat matches_image(Size(left_image.cols + right_image.cols, left_image.rows), CV_8UC3);
	Mat left_roi(matches_image, Rect(0, 0, left_image.cols, left_image.rows));
	Mat right_roi(matches_image, Rect(left_image.cols, 0, right_image.cols, left_image.rows));
	left_image.copyTo(left_roi);
	right_image.copyTo(right_roi);

	//using the transformation matrix returned by affine/homography, compute new x,y coords if right keypoints are projected to the left image's plane
	//this is needed for computation of residuals
	vector<Point2f> right_transformed;
	perspectiveTransform(right_good_keypoints, right_transformed, transformation);

	double ave_residual = 0;
	int c = 0;
	for (int k = 0; k < mask.rows; k++){
		//if inlier
		if ((unsigned int)mask.at<uchar>(k) == 1){
			c++;

			//save to variables for code readability 
			int Lx = left_good_keypoints[k].x;
			int Ly = left_good_keypoints[k].y;
			int Rx = right_good_keypoints[k].x;
			int Ry = right_good_keypoints[k].y;
			int Tx = right_transformed[k].x;
			int Ty = right_transformed[k].y;

			//compute residual = squared distance between the point coords in left image and the corresponding transformed coords of right
			//or euclidean distance squared
			double residual = (Lx - Tx)*(Lx - Tx) + (Ly - Ty)*(Ly - Ty);
			ave_residual += residual;

			//display inlier#, left coords, right coords, transformed right coords, residual
			//cout << "INLIER " << c << ": left (" << Lx << "," << Ly << "), " << "right (" << Rx << "," << Ry << "), right-to-left transformed(" << Tx << ", " << Ty << "), residual(" << residual << ")" << endl;

			//create circles and connecting lines to inlier matches
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			circle(matches_image, Point(Lx, Ly), 5, color, 1, CV_AA, 0);
			circle(matches_image, Point(Rx + left_image.cols, Ry), 5, color, 1, CV_AA, 0);
			line(matches_image, Point(Lx, Ly), Point(Rx + left_image.cols, Ry), color, 1, CV_AA);
		}
	}
	
	//display total inlier vs all descriptor pairs
	cout << endl;
	cout << "#INLIERS:" << sum(mask)[0] << " / TOTAL:" << mask.rows << endl;

	//compute and display average residual
	ave_residual = ave_residual / sum(mask)[0];
	cout << endl;
	cout << "AVERAGE RESIDUAL: " << ave_residual << endl;

	//warping right image based on the transformation matrix returned by RANSAC
	//use BORDER_REPLICATE so it wouldn't use black background -> looks neater when stitched/blended
	Mat warped_image;
	warpPerspective(right_image, warped_image, transformation, Size(left_image.cols + right_image.cols, right_image.rows), INTER_LINEAR, BORDER_REPLICATE);
	


//STITCHING AND BLENDING THE TWO IMAGES

	//need to identify: 
	//(1) the leftmost x-coordinate of the warped image to know the start of the overlapping region
	//(2) where the final_image should end so that the image will not show the blank parts of the warp
	//note: 1st item corresponds to top-left corner of the original right image then transformed
	//while 2nd corresponds to the top-right corner of the original right image then transformed
	vector<Point2f> corners, corners_transformed;
	corners.push_back(Point2f(0, 0));					//top-left corner
	corners.push_back(Point2f(right_image.cols, 0));	//top-right corner
	
	//to get new coordinates on warped image when above corners were transformed
	perspectiveTransform(corners, corners_transformed, transformation);

	//create final_image, length size = cropped to the top-right corner of the warped image
	Mat final_image(warped_image.rows, corners_transformed[1].x, CV_8UC3);

	//creating the final image
	//blend overlapping pixel by using weighted average, weight is a gradient from left image to right image
	for (int row = 0; row < final_image.rows; row++) {
		for (int col = 0; col < final_image.cols; col++) {

			//define overlapping area as < right-most side of the left image
			//and > left-most side of the warped right image
			if (col < left_image.cols){
				if (col > corners_transformed[0].x){

					//weights for non-blended image
					//double weight1 = 0;
					//double weight2 = 1;

					//compute for weight, this is gradient from left image to right
					//if the pixel is nearer to left, the value of weight2 is higher
					//if the pixel is nearer to right, the value of weight1 is higher
					double weight1 = (col - (corners_transformed[0].x)) / (double)(left_image.cols - (corners_transformed[0].x));
					if (weight1 < 0) weight1 = 0;
					weight1 = weight1*weight1;
					double weight2 = 1 - weight1;

					//compute weighted average to determine new color of pixel
					uchar B = weight1*warped_image.at<Vec3b>(row, col)[0] + weight2*left_image.at<Vec3b>(row, col)[0];
					uchar G = weight1*warped_image.at<Vec3b>(row, col)[1] + weight2*left_image.at<Vec3b>(row, col)[1];
					uchar R = weight1*warped_image.at<Vec3b>(row, col)[2] + weight2*left_image.at<Vec3b>(row, col)[2];
					pixel = { B, G, R };

				}
				//pixels at the left of the overlapping region will use left image's pixels exclusively
				else{
					pixel = left_image.at<Vec3b>(row, col);
				}
			}
			//pixels at the right of the overlapping region will use left image's pixels exclusively
			else{
				pixel = warped_image.at<Vec3b>(row, col);
			}

			//assign pixel in the final_image	
			final_image.at<Vec3b>(row, col) = pixel;
		}
	}



//OUTPUT

	//display matches and final image
	imshow("Inlier Matches", matches_image);
	imshow("Image Stitching", final_image);

	//save matches and final image to JPEG file
	if (use_affine){
		imwrite("imgpairs\\inlier_matches_output_aff.jpg", matches_image);
		imwrite("imgpairs\\stitched_output_aff.jpg", final_image);
	}
	else{
		imwrite("imgpairs\\inlier_matches_output_hom.jpg", matches_image);
		imwrite("imgpairs\\stitched_output_hom.jpg", final_image);
	}

	cout << endl;
	cout << "Press any key while in the image to exit . . .";
	waitKey(0);
	destroyWindow("Image Stitching");

	return 0;
}