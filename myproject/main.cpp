// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>


namespace aia
{
	// local (8-neighborhood) region-growing segmentation based on color difference
	void colorDiffSegmentation(const cv::Mat & img, cv::Scalar colorDiff = cv::Scalar::all(3));
}

void standardize_orientation(cv::Mat &orig);
void appplyCLAHE(cv::Mat &orig, int clip_limit = 6, int tile_size = 10);
void segmentation(cv::Mat &orig);

int main()
{
	cv::Mat img;
	std::string win_name = "CLAHE";
	int clip_limit = 6;
	int tile_size = 10;

	cv::startWindowThread();

	img = cv::imread(std::string(DATASET_PATH) + "/images/20587080_b6a4f750c6df4f90_MG_R_ML_ANON.tif", cv::IMREAD_GRAYSCALE);
	
	// resize it to 20% of its original size
	cv::resize(img, img, cv::Size(0,0), 0.2, 0.2);

	// standardize orientation
	standardize_orientation(img);

/* 	// mean shift filtering
	cv::Mat img_ms;
	cv::Mat bgr_image;
	cv::cvtColor(img, bgr_image, cv::COLOR_GRAY2BGR);
	cv::pyrMeanShiftFiltering(bgr_image, img_ms, 20, 20, 0);
	aia::imshow("Mean-Shift", img_ms, true, 0.2f);

	// apply CLAHE
	//appplyCLAHE(img);

	// show image
	aia::imshow(win_name, img, false, 0.2f); */

	segmentation(img);

	// wait for ESC key to be pressed to exit
	char exit_key_press = 0;
	while (exit_key_press != 27) // or key != 'q'
	{
		exit_key_press = cv::waitKey(10);
	}

	return EXIT_SUCCESS;
}

void standardize_orientation(cv::Mat &original) {

	int right_cntr = 0;
	int left_cntr = 0;

	// check in which side the breast is located
	for (int i = 0; i < original.cols / 2; i++) {
		for (int j = 0; j < original.rows / 2; j++) {
			if (original.at<uchar>(j, i) > 0) {
				right_cntr++;
			}
			if (original.at<uchar>(j, original.cols - i - 1) > 0) {
				left_cntr++;
			}
		}
	}

	std::cout << "right_cntr: " << right_cntr << std::endl;
	std::cout << "left_cntr: " << left_cntr << std::endl;

	// flip the image if the breast is on the left side
	if (right_cntr > left_cntr) {
		cv::flip(original, original, 1);
	}
}

void appplyCLAHE(cv::Mat &orig, int clip_limit, int tile_size)
{
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(tile_size, tile_size));
	clahe->apply(orig, orig);
}

// implement segmentation using mean shift filtering and CLAHE
void segmentation(cv::Mat &orig)
{
	cv::Mat img_ms;
	cv::Mat bgr_image;
	cv::cvtColor(orig, bgr_image, cv::COLOR_GRAY2BGR);
	cv::pyrMeanShiftFiltering(bgr_image, img_ms, 10, 10, 0);
	aia::imshow("Mean-Shift", img_ms);

	// region growing based on color difference
	aia::colorDiffSegmentation(img_ms);
	aia::imshow("Postprocessing", img_ms);

	// convert back to grayscale
	cv::cvtColor(img_ms, img_ms, cv::COLOR_BGR2GRAY);
	aia::imshow("Grayscale", img_ms);

	// thresholding
	cv::threshold(img_ms, img_ms, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	aia::imshow("Thresholding", img_ms);

	// select top-right connected component
	unsigned char muscle_intensity = img_ms.at<unsigned char>(10, img_ms.cols-10);
	std::cout << "Muscle intensity: " << (int)muscle_intensity << std::endl;
	cv::inRange(img_ms, muscle_intensity-5, muscle_intensity+5, img_ms);
	std::vector < std::vector <cv::Point> > components;
	cv::findContours(img_ms, components, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	int muscle_component_idx = 0;
	if(components.size() != 1)
	{
		double maxArea = 0;
		for(int i=0; i<components.size(); i++)
		{
			double compArea = cv::contourArea(components[i]);
			if(compArea > maxArea)
			{
				maxArea = compArea;
				muscle_component_idx = i;
			}
		}
	}

	// overlay with original image
	cv::Mat selection_layer = orig.clone();
	cv::drawContours(selection_layer, components, muscle_component_idx, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
	cv::addWeighted(orig, 0.8, selection_layer, 0.2, 0, orig);
	aia::imshow("Result", orig);
}

// local (8-neighborhood) region-growing segmentation based on color difference
void aia::colorDiffSegmentation(const cv::Mat & img, cv::Scalar colorDiff)
{
	// a number generator we will use to assign random colors to
	cv::RNG rng = cv::theRNG();

	// mask used to accelerate processing
	cv::Mat mask( img.rows+2, img.cols+2, CV_8UC1, cv::Scalar::all(0) );

	// use every pixel as seed for region growing
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			// avoid growing from a seed that has already been merged with another region
			if( mask.at<uchar>(y+1, x+1) == 0 )
			{
				cv::Scalar newVal( rng(256), rng(256), rng(256) );
				cv::floodFill( img, mask, cv::Point(x,y), newVal, 0, colorDiff, colorDiff );
			}
		}
	}
}