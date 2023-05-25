// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

void apply_ms(int, void*);
void segmentation(cv::Mat &);
void intensity_based_segmentation(int, void*);
void colorDiffSegmentation(const cv::Mat &, cv::Scalar);
bool standardize_orientation(cv::Mat &);
void appplyCLAHE(int, void*);
cv::Mat quantizeImage(cv::Mat, int);

// since we work with a GUI, we need parameters (and the images) to be stored in global variables
namespace aia
{
	float scale = 0.15f;				// scale factor for the input image

	// the images we need to keep in memory
	cv::Mat img;         			// input image
	cv::Mat imgEdges;      	  		// binary image after edge detection
	cv::Mat orig;			 		// original image
	cv::Mat img_ms;			 		// mean-shift image
	cv::Mat img_ms_kmeans;	 		// mean-shift image for k-means (BGR)
	cv::Mat segmented_hough;		// segmented image by hough method
	cv::Mat segmented_intensity;	// segmented image by intensity method
	cv::Mat segmented_final;		// final segmented image
	cv::Mat clahe;					// CLAHE image
	cv::Mat mask_hough;
	cv::Mat mask_intensity;
	cv::Mat img_fs;

	// parameters of edge detection
	int stdevX10;           // standard deviation of the gaussian smoothing applied as denoising prior to the calculation of the image derivative
	// 'X10' means it is multiplied by 10: unfortunately the OpenCV GUI does not support real-value trackbar, so we have to deal with integers
	int threshold;          // threshold applied on the gradient magnitude image normalized in [0, 255]
	int alpha0;             // filter on gradient orientation: only gradients whose orientation is between [alpha0,alpha1] are considered
	int alpha1;             // filter on gradient orientation: only gradients whose orientation is between [alpha0,alpha1] are considered

	// parameters of Hough line detection
	int drho;               // quantization along rho axis
	int dtheta;             // quantization along theta axis
	int accum;              // accumulation threshold
	int n;                  // if != 0, we take the 'n' most voted (highest accumulation) lines

	// parameters of intensity-based segmentation
	int l_range = 10;
	int u_range = 10;
	int muscle_component_idx = 0;

	// parameters of mean-shift segmentation
	int termcrit = 20;

	// parameters of k-means segmentation
	int kmeans_clusters = 4;

	// parameters of CLAHE
	int clip_limit = 6;
	int tile_size = 10;

	// Window names
	const std::string win_name_edge = "Edge detection (gradient)";
	const std::string win_name_hough = "Hough transform lines";
	const std::string win_name_int_seg = "Intensity-based segmentation";
	const std::string win_name_kmeans = "K-means segmentation";
	const std::string win_name_ms = "Mean shift";
	const std::string win_name_clahe = "CLAHE";

	void edgeDetectionGrad(int, void*);
	void Hough(int, void*);

	int use_ms = 0;
	int show_kmeans = 0;
	int use_intensity = 0;
	int refresh_ui = 0;
	bool change_ui = false;
	int use_nline = 0;
}

void set_toggle_ui(int state, void* d) {
    int* int_ptr = static_cast<int*>(d);
    *int_ptr = state;
	aia::change_ui = true;
}
void set_toggle(int state, void* d) {
    int* int_ptr = static_cast<int*>(d);
    *int_ptr = state;
}

//////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////
int main()
{

    std::string path = std::string(DATASET_PATH) + "/images/";
    for (const auto & entry : fs::directory_iterator(path)) {

		fs::path resultsPath = fs::current_path() / "../../results/preprocessing/";
		std::cout << resultsPath.string() << std::endl;

        std::cout << fs::path(entry).stem() << std::endl;
		std::string filename = fs::path(entry).stem().string();

		if (filename.find("_CC_") != std::string::npos) {
			std::cout << fs::path(entry).stem() << std::endl;
			continue;
		}

		// check if file exists
		if (fs::exists("../../results/preprocessing/" + filename + "_preprocessed" + ".tif")) {
			std::cout << "File " << filename << " already preprocessed" << std::endl;
			continue;
		}

		cv::Mat img;

		cv::startWindowThread();



		img = cv::imread(
			std::string(DATASET_PATH) + 
			"/images/" + filename + ".tif", 
			cv::IMREAD_GRAYSCALE);

		// check if the image was loaded
		if (img.empty())
		{
			std::cout << "Error: Image cannot be loaded!" << std::endl;
			return EXIT_FAILURE;
		}

		// standardize orientation
		bool rotated = false;
		rotated = standardize_orientation(img);

		aia::img_fs = img.clone();

		// resize it to 15% of its original size
		cv::resize(img, aia::orig, cv::Size(0,0), aia::scale, aia::scale);

		aia::imshow("Orig", aia::orig, false);
		cv::createButton("Use mean-shift", set_toggle_ui, &aia::use_ms, cv::QT_CHECKBOX, aia::use_ms ? 1 : 0);
		cv::createButton("Show k-means (needs mean-shift)", set_toggle_ui, &aia::show_kmeans, cv::QT_CHECKBOX, aia::show_kmeans ? 1 : 0);
		cv::createButton("Use intensity seg", set_toggle, &aia::use_intensity, cv::QT_CHECKBOX, aia::use_intensity ? 1 : 0);
		cv::createButton("Refresh UI", set_toggle_ui, &aia::refresh_ui, cv::QT_PUSH_BUTTON|cv::QT_NEW_BUTTONBAR, 0);
		cv::createButton("Use n line", set_toggle, &aia::use_nline, cv::QT_CHECKBOX, aia::use_nline ? 1 : 0);


		// set default parameters
		aia::stdevX10 = 20;
		aia::threshold = 10;
		aia::alpha0 = 0;
		aia::alpha1 = 360;
		aia::drho = 1;
		aia::dtheta = 6;
		aia::accum = 11;
		aia::n = 10;

		segmentation(aia::orig);

		// wait for ESC key to be pressed to exit
		char exit_key_press = 0;
		while (exit_key_press != 27) // or key != 'q'
		{
		
			if (aia::change_ui) {
				aia::change_ui = false;
				aia::refresh_ui = 0;
				segmentation(aia::orig);
			}

			exit_key_press = cv::waitKey(1);

		}

		cv::destroyAllWindows();

		if (aia::use_intensity) {
			aia::segmented_final = aia::segmented_intensity;
		}
		else {
			aia::segmented_final = aia::segmented_hough;
		}

		// flip image to original orientation
		if (rotated) {
			cv::flip(aia::segmented_final, aia::segmented_final, 1);
			rotated = false;
		}

		appplyCLAHE(1, 0);
		cv::createTrackbar("clip_limit", aia::win_name_clahe, &aia::clip_limit, 40, appplyCLAHE);
		cv::createTrackbar("tile_size", aia::win_name_clahe, &aia::tile_size, 40, appplyCLAHE);

		// wait for ESC key to be pressed to exit
		exit_key_press = 0;
		while (exit_key_press != 27) // or key != 'q'
		{
			exit_key_press = cv::waitKey(10);
		}

		cv::imwrite(resultsPath.string() +
		filename + "_preprocessed" + ".tif", aia::clahe);

		cv::imwrite(resultsPath.string() + "/noclahe/" + 
		filename + "_nopecmuscle" + ".tif", aia::segmented_final);

		std::ofstream log;
		log.open (resultsPath.string() + "/logs/" +
		filename + "_log" + ".txt");
		log <<
		"filename: " << filename << std::endl <<
		"stdevX10: " << aia::stdevX10 << std::endl <<
		"threshold: " << aia::threshold << std::endl <<
		"alpha0: " << aia::alpha0 << std::endl <<
		"alpha1: " << aia::alpha1 << std::endl <<
		"drho: " << aia::drho << std::endl <<
		"dtheta: " << aia::dtheta << std::endl <<
		"accum: " << aia::accum << std::endl <<
		"n: " << aia::n << std::endl <<
		"clip_limit: " << aia::clip_limit << std::endl <<
		"tile_size: " << aia::tile_size << std::endl <<
		"scale: " << aia::scale << std::endl <<
		"rotated: " << rotated << std::endl <<
		"use_ms: " << aia::use_ms << std::endl <<
		"show_kmeans: " << aia::show_kmeans << std::endl <<
		"use_intensity: " << aia::use_intensity << std::endl;
		log.close();

		cv::destroyAllWindows();
	}

	return EXIT_SUCCESS;
}

void apply_ms(int, void*) {
	cv::Mat img_ms;
	cv::Mat bgr_image;
	cv::medianBlur(aia::orig, img_ms, 7);
	cv::cvtColor(img_ms, bgr_image, cv::COLOR_GRAY2BGR);
	cv::pyrMeanShiftFiltering(bgr_image, img_ms, aia::termcrit, aia::termcrit, 0);
	aia::img_ms_kmeans = img_ms.clone();
	cv::cvtColor(img_ms, aia::img_ms, cv::COLOR_BGR2GRAY);
	aia::imshow(aia::win_name_ms, aia::img_ms, false);
}


//////////////////////////////////////////
// SEGMENTATION
//////////////////////////////////////////
void segmentation(cv::Mat &orig)
{
	if (aia::use_ms) {
		cv::namedWindow(aia::win_name_ms);
		cv::createTrackbar("termcrit", aia::win_name_ms, &aia::termcrit, 40, apply_ms);
		apply_ms(1, 0);

	}
	else {
		if (cv::getWindowProperty(aia::win_name_ms, cv::WND_PROP_VISIBLE) == 1)
			cv::destroyWindow(aia::win_name_ms);
	}

	cv::Mat img_kmeans;
	if (aia::use_ms && aia::show_kmeans) {
		img_kmeans = quantizeImage(aia::img_ms_kmeans, aia::kmeans_clusters);
		aia::imshow("K-means test", img_kmeans, false);
	}
	else {
		std::cout << "test" << cv::getWindowProperty("K-means test", cv::WND_PROP_VISIBLE) << std::endl;
		if (cv::getWindowProperty("K-means test", cv::WND_PROP_VISIBLE) == 1)
			cv::destroyWindow("K-means test");
	}

/* 	// region growing based on color difference
	cv::Mat img_ms_copy = img_ms.clone();
	aia::colorDiffSegmentation(img_ms_copy);
	aia::imshow("Postprocessing", img_ms_copy); */


	//////////////////////////////////////////
	// HOUGH-BASED SEGMENTATION
	//////////////////////////////////////////
	if (aia::use_ms) {
		aia::img = aia::img_ms.clone();
	} else {
		aia::img = orig.clone();
	}

	// create a window named 'Edge detection (gradient)' and insert the trackbars
	cv::namedWindow(aia::win_name_edge);
	cv::createTrackbar("stdev(x10)", aia::win_name_edge, &aia::stdevX10, 100, aia::edgeDetectionGrad);
	cv::createTrackbar("threshold", aia::win_name_edge, &aia::threshold, 100, aia::edgeDetectionGrad);
	cv::createTrackbar("alpha0", aia::win_name_edge, &aia::alpha0, 360, aia::edgeDetectionGrad);
	cv::createTrackbar("alpha1", aia::win_name_edge, &aia::alpha1, 360, aia::edgeDetectionGrad);

	// create a window named 'Hough transform lines' and insert the trackbars
	cv::namedWindow(aia::win_name_hough);
	cv::createTrackbar("drho", aia::win_name_hough, &aia::drho, 100, aia::Hough);
	cv::createTrackbar("dtheta", aia::win_name_hough, &aia::dtheta, 100, aia::Hough);
	cv::createTrackbar("accum", aia::win_name_hough, &aia::accum, 100, aia::Hough);
	cv::createTrackbar("n", aia::win_name_hough, &aia::n, 50, aia::Hough);

	// run edge detection + Hough for the first time with default parameters
	aia::edgeDetectionGrad(1, 0);
	aia::Hough(1, 0);
	//////////////////////////////////////////

/* 	// Apply Gaussian filter to smooth image
	cv::Mat blurred;
	cv::GaussianBlur(img_ms, blurred, cv::Size(5, 5), 0);

	// Threshold image to separate pectoral muscle from background
	cv::Mat binary;
	cv::threshold(blurred, binary, 95, 255, cv::THRESH_BINARY_INV);
	aia::imshow("Binary", binary);

	// Apply binary mask to original mammogram	 image to extract pectoral muscle
	cv::Mat pectoral;
	img_ms.copyTo(pectoral, binary);
	aia::imshow("Pectoral", pectoral); */

/* 	// thresholding
	cv::threshold(img_ms, img_ms, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	aia::imshow("Thresholding", img_ms); */

	if (aia::use_ms) {
		aia::img_ms = aia::img_ms.clone();
		cv::namedWindow(aia::win_name_int_seg);
		cv::createTrackbar("lower range", aia::win_name_int_seg, &aia::l_range, 20, intensity_based_segmentation);
		cv::createTrackbar("upper range", aia::win_name_int_seg, &aia::u_range, 20, intensity_based_segmentation);
		cv::createTrackbar("component id", aia::win_name_int_seg, &aia::muscle_component_idx, 5, intensity_based_segmentation);

		intensity_based_segmentation(1, 0);
	}
	else {
		if (cv::getWindowProperty(aia::win_name_int_seg, cv::WND_PROP_VISIBLE) == 1)
			cv::destroyWindow(aia::win_name_int_seg);
	}

}


//////////////////////////////////////////
// INTENSITY-BASED SEGMENTATION
//////////////////////////////////////////
// comparison function object
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
}
void intensity_based_segmentation(int, void* data) {
	unsigned char muscle_intensity = aia::img_ms.at<unsigned char>(10, aia::img_ms.cols-10);
	cv::Mat mask = cv::Mat::zeros(aia::img_ms.size(), CV_8UC1);
	cv::inRange(aia::img_ms, muscle_intensity-aia::l_range, muscle_intensity+aia::u_range, mask);
	std::vector < std::vector <cv::Point> > components;
	cv::findContours(mask, components, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::cout << "Number of components: " << components.size() << std::endl;

	if (aia::muscle_component_idx > components.size())
		return;

	// sort contours
	std::sort(components.begin(), components.end(), compareContourAreas);
	
	// overlay with original image
	cv::Mat selection_layer = aia::orig.clone();
	cv::drawContours(selection_layer, components, aia::muscle_component_idx, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
	aia::imshow(aia::win_name_int_seg, selection_layer, false);


	aia::segmented_intensity = aia::img_fs.clone();
	cv::fillConvexPoly(aia::segmented_intensity, cv::Mat (components[aia::muscle_component_idx] ) / aia::scale, cv::Scalar(0, 255, 255));
}


//////////////////////////////////////////
// STANDARDIZE ORIENTATION
//////////////////////////////////////////
bool standardize_orientation(cv::Mat &original) {
	int rotated = false;
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
		rotated = true;
	}

	return rotated;
}


//////////////////////////////////////////
// CLAHE
//////////////////////////////////////////
void appplyCLAHE(int, void*)
{
	if (aia::tile_size < 1)
		return;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(aia::clip_limit, cv::Size(aia::tile_size, aia::tile_size));
	clahe->apply(aia::segmented_final, aia::clahe);
	aia::imshow(aia::win_name_clahe, aia::clahe, false, 0.17f);
}


//////////////////////////////////////////////////////////
// local (8-neighborhood) region-growing segmentation
// based on color difference
//////////////////////////////////////////////////////////
void colorDiffSegmentation(const cv::Mat &img, cv::Scalar colorDiff)
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


//////////////////////////////////////////////////////////
// QUANTIZATION
//////////////////////////////////////////////////////////
cv::Mat quantizeImage(cv::Mat _inputImage, int _quantizationColors) {
	cv::Mat src = _inputImage.clone();  //cloning mat data
	cv::Mat data = cv::Mat::zeros(src.cols * src.rows, 3, CV_32F);  //Creating the matrix that holds all pixel data
	cv::Mat bestLabels, centers, clustered; //Returns from the K Means
	std::vector<cv::Mat> bgr;    //Holds the BGR channels
	cv::split(src, bgr);

	//Getting all pixels in the Data row column to be compatible with K Means
	for (int i = 0; i < src.cols * src.rows; i++) {
			data.at<float>(i, 0) = bgr[0].data[i] / 255.0;
			data.at<float>(i, 1) = bgr[1].data[i] / 255.0;
			data.at<float>(i, 2) = bgr[2].data[i] / 255.0;
	}

	int K = _quantizationColors;    //Number of clusters
	cv::kmeans(data, K, bestLabels,
			cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
			3, cv::KMEANS_PP_CENTERS, centers);

	centers = centers.reshape(3, centers.rows);
	data = data.reshape(3, data.rows);

	clustered = cv::Mat(src.rows, src.cols, CV_32F);


	cv::Vec3f* p = data.ptr<cv::Vec3f>();
	for (size_t i = 0; i < data.rows; i++) {
			int center_id = bestLabels.at<int>(i);
			p[i] = centers.at<cv::Vec3f>(center_id);
	}

	clustered = data.reshape(3, src.rows);
	return clustered;
}

// edge detection using gradient / first-order derivatives
void aia::edgeDetectionGrad(int, void*)
{
	// if 'stdevX10' is valid, we apply gaussian smoothing
	if (stdevX10 > 0)
		cv::GaussianBlur(img, imgEdges, cv::Size(0, 0), (stdevX10 / 10.0), (stdevX10 / 10.0));
	// otherwise we simply clone the image as it is
	else
		imgEdges = img.clone();
	// NOTE: we store the result into 'imgEdges', that we will re-use after
	//       in this way we can avoid allocating multiple cv::Mat and make the processing faster

	// compute first-order derivatives along X and Y
	cv::Mat img_dx, img_dy;
	cv::Sobel(imgEdges, img_dx, CV_32F, 1, 0);
	cv::Sobel(imgEdges, img_dy, CV_32F, 0, 1);

	// compute gradient magnitude and angle
	cv::Mat mag, angle;
	cv::cartToPolar(img_dx, img_dy, mag, angle, true);

	// generate a binary image from gradient magnitude and angle matrices
	// how?
	// - take pixels whose gradient magnitude is higher than the specified threshold
	//   AND
	// - take pixels whose angle is within the specified range
	for (int y = 0; y < imgEdges.rows; y++)
	{
		aia::uint8* imgEdgesYthRow = imgEdges.ptr<aia::uint8>(y);
		float* magYthRow = mag.ptr<float>(y);
		float* angleYthRow = angle.ptr<float>(y);

		for (int x = 0; x < imgEdges.cols; x++)
		{
			if (magYthRow[x] > threshold && (angleYthRow[x] >= alpha0 || angleYthRow[x] <= alpha1))
				imgEdgesYthRow[x] = 255;
			else
				imgEdgesYthRow[x] = 0;
		}
	}

	cv::imshow(aia::win_name_edge, imgEdges);
}


// line detection using Hough transform
void aia::Hough(int, void*)
{
	// in case we have invalid parameters, we do nothing
	if (drho <= 0)
		return;
	if (dtheta <= 0)
		return;
	if (accum <= 0)
		return;

	// Hough returns a vector of lines represented by (rho,theta) pairs
	// the vector is automatically sorted by decreasing accumulation scores
	// this means the first 'n' lines are the most voted
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(imgEdges, lines, drho, dtheta / 180.0, accum, 0, 0, aia::PI / 2.0, aia::PI);


	//////////////////////////////////////////
	// FILTER OUT IRRELEVANT LINES
	//////////////////////////////////////////
	// Iterate through the lines and filter out irrelevant ones
	std::vector<cv::Vec2f> filtered_lines;
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec2f line = lines[i];
		float rho = line[0];
		float theta = line[1];
		double angle = theta * 180 / CV_PI;
		if (angle < 120 || angle > 175)
			continue;

		filtered_lines.push_back(line);
	}
	std::cout << "Number of lines: " << filtered_lines.size() << std::endl;

	lines = filtered_lines;


	//////////////////////////////////////////
	// DRAW FIRST N LINES
	//////////////////////////////////////////

	// we draw the first 'n' lines
	cv::Mat img_copy = img.clone();
	for (int i = 0; i < std::min(size_t(n), lines.size()); i++)
	{
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( img_copy, pt1, pt2, cv::Scalar(255), 1, cv::LINE_AA);
	}

	cv::imshow(aia::win_name_hough, img_copy);
	//////////////////////////////////////////


	//////////////////////////////////////////
	// DETERMINE CLOSEST LINE TO TOP
	//////////////////////////////////////////

	// initialize a variable to store the minimum distance
	double min_dist = std::numeric_limits<double>::max();

	// initialize a variable to store the index of the closest line
	int closest_line = -1;

	// loop through the lines
	for (int i = 0; i < std::min(size_t(n), lines.size()); i++)
	{
		// get the rho and theta values of the line
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

		// Calculate the middle point of the line
		cv::Point mid((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);

		// Get the top right corner of the image
		cv::Point tr(img.cols, 0);

		// calculate distance
		//double dist = cv::norm(mid - tr);
		double dist = img.cols + rho;
		
		// check if the distance is smaller than the current minimum
		if (dist < min_dist)
		{
			// update the minimum distance and the index of the closest line
			min_dist = dist;
			closest_line = i;

			// print distance
			std::cout << "Distance to top right corner = " << dist << std::endl;
		}
	}
	// check if a closest line was found
	if (closest_line != -1)
	{
		// print the rho and theta values of the closest line
		std::cout << "The closest line to the top right corner has angle = " <<
		lines[closest_line][1] * 180 / CV_PI <<
		" and rho =" << lines[closest_line][0] << std::endl;
	}
	else
	{
		// print a message that no line was found
		std::cout << "No line was found." << std::endl;
	}
	//////////////////////////////////////////

	if (aia::use_nline) {
		closest_line = n - 1;
	}

	//////////////////////////////////////////
	// DRAW SELECTED LINE
	//////////////////////////////////////////

	// we draw the first 'n' lines
	cv::Mat img_copy2 = img.clone();
	float rho = lines[closest_line][0];
	float theta = lines[closest_line][1];

	if (theta < aia::PI / 4. || theta > 3. * aia::PI / 4.)
	{ // ~vertical line

		// point of intersection of the line with first row
		cv::Point pt1(rho / cos(theta), 0);
		// point of intersection of the line with last row
		cv::Point pt2((rho - img_copy2.rows * sin(theta)) / cos(theta), img_copy2.rows);
		// draw a white line
		cv::line(img_copy2, pt1, pt2, cv::Scalar(0, 0, 255), 1);

		// Calculate the middle point of the line
		cv::Point mid((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);

		// Get the top right corner of the image
		cv::Point tr(img.cols, 0);

			// draw line from middle point to top right corner
		cv::line(img_copy2, mid, tr, cv::Scalar(255), 2, cv::LINE_AA);
	}
	else
	{ // ~horizontal line

		// point of intersection of the line with first column
		cv::Point pt1(0, rho / sin(theta));
		// point of intersection of the line with last column
		cv::Point pt2(img_copy2.cols, (rho - img_copy2.cols * cos(theta)) / sin(theta));
		// draw a white line
		cv::line(img_copy2, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

		// Calculate the middle point of the line
		cv::Point mid((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);

		// Get the top right corner of the image
		cv::Point tr(img.cols, 0);

		// draw line from middle point to top right corner
		cv::line(img_copy2, mid, tr, cv::Scalar(255), 2, cv::LINE_AA);
	}

	

	cv::imshow("Chosen line", img_copy2);
	//////////////////////////////////////////


	//////////////////////////////////////////
	// SEGMENT PECTORAL MUSCLE
	//////////////////////////////////////////

	//Get the closest line to the top right corner
	cv::Vec2f pectoral_line = lines[closest_line];

	//Use the pectoral line as a mask to segment the mammogram
	cv::Mat mask = cv::Mat::ones(img.size(), CV_8UC1) * 255;
	double rho2 = pectoral_line[0], theta2 = pectoral_line[1];
	double x1 = rho2 / cos(theta2), y1 = 0;
	double x2 = img.cols, y2 = (rho2 - x2 * cos(theta2)) / sin(theta2);

	// Create a mask polygon using points
	std::vector<cv::Point> mask_pts;
	mask_pts.push_back(cv::Point(x1, y1));
	mask_pts.push_back(cv::Point(x2, y2));
	mask_pts.push_back(cv::Point(img.cols, 0));
	cv::fillConvexPoly(mask, mask_pts, cv::Scalar(0));
	aia::mask_hough = mask;
	// Use the mask to segment the mammogram
	cv::Mat segmented;
	orig.copyTo(segmented, mask);
	
	// Display the result
	cv::imshow("Segmented mammogram", segmented);

	aia::segmented_hough = aia::img_fs.clone();
	cv::fillConvexPoly(aia::segmented_hough, cv::Mat( mask_pts ) / scale, cv::Scalar(0));
}