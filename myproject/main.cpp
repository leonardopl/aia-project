// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	std::string win_name = "CLAHE";
	int clip_limit = 10;
	int tile_size = 4;

	void CLAHECallback(int pos, void* userdata)
	{
		cv::imshow(win_name, img);

		if (tile_size < 1)
			return;

		cv::Mat transformed_img;
		std::vector<cv::Mat> transformed_img_channels;
		cv::Mat out;
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(tile_size, tile_size));

		// BGR-based
		// channels split
		cv::split(img, transformed_img_channels);
		// 3x per-channel CLAHE
		clahe->apply(transformed_img_channels[0], transformed_img_channels[0]);
		clahe->apply(transformed_img_channels[1], transformed_img_channels[1]);
		clahe->apply(transformed_img_channels[2], transformed_img_channels[2]);
		// channels merge
		cv::merge(transformed_img_channels, out);
		//cv::startWindowThread();
		cv::namedWindow("Result (BGR)", cv::WINDOW_NORMAL);
		cv::imshow("Result (BGR)", out);
		cv::resizeWindow("Result (BGR)", 700, 900);

/* 		// HSV-based
		// color space selection BGR --> HSV
		cv::cvtColor(img, transformed_img, cv::COLOR_BGR2HSV);
		cv::split(transformed_img, transformed_img_channels);
		// channel selection and CLAHE
		clahe->apply(transformed_img_channels[2], transformed_img_channels[2]);
		// HSV --> BGR color space restoration
		cv::merge(transformed_img_channels, out);
		cv::cvtColor(out, out, cv::COLOR_HSV2BGR);
		cv::imshow("Result (HSV)", out);

		// Lab-based
		// color space selection BGR --> Lab
		cv::cvtColor(img, transformed_img, cv::COLOR_BGR2Lab);
		cv::split(transformed_img, transformed_img_channels);
		// channel selection and CLAHE
		clahe->apply(transformed_img_channels[0], transformed_img_channels[0]);
		// Lab --> BGR color space restoration
		cv::merge(transformed_img_channels, out);
		cv::cvtColor(out, out, cv::COLOR_Lab2BGR);
		cv::imshow("Result (Lab)", out); */
	}
};



int main()
{
	cv::startWindowThread();

	img = cv::imread(std::string(DATASET_PATH) + "/images/22678495_60995d51033e24b8_MG_R_ML_ANON.tif");

	// 1) resize it to 20% of its original size
	//cv::resize(img, img, cv::Size(0,0), 0.2, 0.2);
	// 2) rescale from 14-bits to 16-bits
	//cv::normalize(img, img, 0, 65535, cv::NORM_MINMAX);
	
	cv::namedWindow(win_name, cv::WINDOW_NORMAL);
	
	cv::resizeWindow(win_name, 700, 900);
	cv::createTrackbar("clip_limit", win_name, &clip_limit, 100, CLAHECallback);
	cv::createTrackbar("tile_size", win_name, &tile_size, 50, CLAHECallback);
	CLAHECallback(0, 0);

	// wait for key press to exit
	char exit_key_press = 0;
	while (exit_key_press != 'q') // or key != ESC
	{
		exit_key_press = cv::waitKey(10);
	}

	return EXIT_SUCCESS;
}