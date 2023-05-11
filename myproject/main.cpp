// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace aia
{
	// local (8-neighborhood) region-growing segmentation based on color difference
	void colorDiffSegmentation(const cv::Mat & img, cv::Scalar colorDiff = cv::Scalar::all(3));
}

void standardize_orientation(cv::Mat &orig);
void appplyCLAHE(cv::Mat &orig, int clip_limit = 6, int tile_size = 10);
void segmentation(cv::Mat &orig);
cv::Mat quantizeImage(cv::Mat _inputImage, int _quantizationColors);

// since we work with a GUI, we need parameters (and the images) to be stored in global variables
namespace aia
{
	// the images we need to keep in memory
	cv::Mat img;            // original image
	cv::Mat imgEdges;       // binary image after edge detection
	cv::Mat orig;					 // original image

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

	// edge detection using gradient / first-order derivatives
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionGrad(int, void*)
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

		cv::imshow("Edge detection (gradient)", imgEdges);
	}


	// line detection using Hough transform
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void Hough(int, void*)
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

		// we draw the first 'n' lines
		cv::Mat img_copy = img.clone();
		for (int k = 0; k < std::min(size_t(n), lines.size()); k++)
		{
			float rho = lines[k][0];
			float theta = lines[k][1];

			if (theta < aia::PI / 4. || theta > 3. * aia::PI / 4.)
			{ // ~vertical line

				// point of intersection of the line with first row
				cv::Point pt1(rho / cos(theta), 0);
				// point of intersection of the line with last row
				cv::Point pt2((rho - img_copy.rows * sin(theta)) / cos(theta), img_copy.rows);
				// draw a white line
				cv::line(img_copy, pt1, pt2, cv::Scalar(0, 0, 255), 1);
			}
			else
			{ // ~horizontal line

				// point of intersection of the line with first column
				cv::Point pt1(0, rho / sin(theta));
				// point of intersection of the line with last column
				cv::Point pt2(img_copy.cols, (rho - img_copy.cols * cos(theta)) / sin(theta));
				// draw a white line
				cv::line(img_copy, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
			}
		}

		cv::imshow("Line detection (Hough)", img_copy);

    // Iterate through the lines and filter out irrelevant ones
    std::vector<cv::Vec2f> filtered_lines;
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec2f line = lines[i];
        float rho = line[0];
        float theta = line[1];
        double angle = theta * 180 / CV_PI;
/*         if (angle < -45 || angle > 45) // filter out horizontal lines
            continue; */

/*         if (rho < img.rows / 2) // filter out lines above the center
            continue; */

        filtered_lines.push_back(line);
    }
		std::cout << "Number of lines: " << filtered_lines.size() << std::endl;

/*     // Select the line that corresponds to the pectoral muscle
    cv::Vec2f pectoral_line;
    double max_length = 0;
    for (size_t i = 0; i < filtered_lines.size(); i++)
    {
        cv::Vec2f line = filtered_lines[i];
        float rho = line[0];
        float theta = line[1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        double x1 = cvRound(x0 + 1000 * (-b)), y1 = cvRound(y0 + 1000 * (a));
        double x2 = cvRound(x0 - 1000 * (-b)), y2 = cvRound(y0 - 1000 * (a));
        double length = cv::norm(cv::Point(x1, y1) - cv::Point(x2, y2));
        if (length > max_length)
        {
            max_length = length;
            pectoral_line = line;
        }
    } */

/*     // Use the pectoral line as a mask to segment the mammogram
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    float rho = pectoral_line[0];
    float theta = pectoral_line[1];
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    double x1 = cvRound(x0 + 1000 * (-b)), y1 = cvRound(y0 + 1000 * (a));
    double x2 = cvRound(x0 - 1000 * (-b)), y2 = cvRound(y0 - 1000 * (a));
    cv::line(mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 10);
		cv::Mat segmented;
    img.copyTo(segmented, mask); */

    // Select the pectoral line that is closest to the top right corner
    cv::Vec2f pectoral_line;
    double min_distance = DBL_MAX;
    for (size_t i = 0; i < std::min(size_t(n), lines.size()); i++)
    {
        cv::Vec2f line = filtered_lines[i];
        double rho = line[0], theta = line[1];
        double x1 = rho / cos(theta), y1 = 0;
        double x2 = (rho - img.cols * sin(theta)) / cos(theta), y2 = img.cols;
        double distance = sqrt((x2 - img.cols) * (x2 - img.cols) + y2 * y2);
        if (distance < min_distance)
        {
            min_distance = distance;
            pectoral_line = line;
        }
    }

/*     // Use the pectoral line as a mask to segment the mammogram
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    double rho = pectoral_line[0], theta = pectoral_line[1];
    double x1 = rho / cos(theta), y1 = 0;
    double x2 = (rho - img.cols * sin(theta)) / cos(theta), y2 = img.cols;
    cv::line(mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 10); */

    // Create the mask to segment the mammogram
    cv::Mat mask = cv::Mat::ones(img.size(), CV_8UC1) * 255;
    double rho = pectoral_line[0], theta = pectoral_line[1];
    double x1 = rho / cos(theta), y1 = 0;
    double x2 = (rho - img.cols * sin(theta)) / cos(theta), y2 = img.cols;

    // Create a mask polygon using points
    std::vector<cv::Point> mask_pts;
    mask_pts.push_back(cv::Point(0, 0));
    mask_pts.push_back(cv::Point(img.cols, 0));
    mask_pts.push_back(cv::Point(x2, y2));
    mask_pts.push_back(cv::Point(x1, y1));
    cv::fillConvexPoly(mask, mask_pts, cv::Scalar(0));

    // Use the mask to segment the mammogram
    cv::Mat segmented;
    orig.copyTo(segmented, mask);

    // Display the result
    cv::imshow("Segmented mammogram", segmented);
	}
}


int main()
{
	cv::Mat img;
	std::string win_name = "CLAHE";
	int clip_limit = 6;
	int tile_size = 10;

	cv::startWindowThread();

	img = cv::imread(std::string(DATASET_PATH) + "/images/20587346_e634830794f5c1bd_MG_R_ML_ANON.tif", cv::IMREAD_GRAYSCALE);
	
	// resize it to 20% of its original size
	cv::resize(img, img, cv::Size(0,0), 0.2, 0.2);

	// standardize orientation
	standardize_orientation(img);

	cv::Mat mammogram = img.clone();

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

// implementation of local (8-neighborhood) region-growing segmentation based on color difference
void segmentation(cv::Mat &orig)
{
	aia::imshow("Orig", orig, false);

	cv::Mat img_ms;
	cv::Mat bgr_image;
	cv::medianBlur(orig, img_ms, 7);
	cv::cvtColor(img_ms, bgr_image, cv::COLOR_GRAY2BGR);
	cv::pyrMeanShiftFiltering(bgr_image, img_ms, 20, 20, 0);

/* 	cv::Mat ts = quantizeImage(img_ms, 4);
	aia::imshow("test", ts, false); */

/* 	// region growing based on color difference
	cv::Mat img_ms_copy = img_ms.clone();
	aia::colorDiffSegmentation(img_ms_copy);
	aia::imshow("Postprocessing", img_ms_copy); */

	// convert back to grayscale
	cv::cvtColor(img_ms, img_ms, cv::COLOR_BGR2GRAY);
	aia::imshow("Mean shift", img_ms, false);

		aia::img = img_ms.clone();
		aia::orig = orig.clone();
			// set default parameters
		aia::stdevX10 = 20;
		aia::threshold = 10;
		aia::alpha0 = 0;
		aia::alpha1 = 360;
		aia::drho = 1;
		aia::dtheta = 1;
		aia::accum = 11;
		aia::n = 30;

		// create a window named 'Edge detection (gradient)' and insert the trackbars
		// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
		cv::namedWindow("Edge detection (gradient)");
		cv::createTrackbar("stdev(x10)", "Edge detection (gradient)", &aia::stdevX10, 100, aia::edgeDetectionGrad);
		cv::createTrackbar("threshold", "Edge detection (gradient)", &aia::threshold, 100, aia::edgeDetectionGrad);
		cv::createTrackbar("alpha0", "Edge detection (gradient)", &aia::alpha0, 360, aia::edgeDetectionGrad);
		cv::createTrackbar("alpha1", "Edge detection (gradient)", &aia::alpha1, 360, aia::edgeDetectionGrad);

		// create another window named 'Line detection (Hough)' and insert the trackbars
		cv::namedWindow("Line detection (Hough)");
		cv::createTrackbar("drho", "Line detection (Hough)", &aia::drho, 100, aia::Hough);
		cv::createTrackbar("dtheta", "Line detection (Hough)", &aia::dtheta, 100, aia::Hough);
		cv::createTrackbar("accum", "Line detection (Hough)", &aia::accum, 100, aia::Hough);
		cv::createTrackbar("n", "Line detection (Hough)", &aia::n, 50, aia::Hough);

		// run edge detection + Hough for the first time with default parameters
		aia::edgeDetectionGrad(1, 0);
		aia::Hough(1, 0);

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

	// select top-right connected component
	unsigned char muscle_intensity = img_ms.at<unsigned char>(10, img_ms.cols-10);
	std::cout << "Muscle intensity: " << (int)muscle_intensity << std::endl;
	cv::inRange(img_ms, muscle_intensity-12, muscle_intensity+12, img_ms);
	std::vector < std::vector <cv::Point> > components;
	cv::findContours(img_ms, components, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	int muscle_component_idx = 0;
	std::cout << "Number of components: " << components.size() << std::endl;
	if(components.size() != 1)
	{
		double maxArea = 0;
		for(int i=0; i<components.size(); i++)
		{
			double compArea = cv::contourArea(components[i]);
			std::cout << "Component " << i << " area: " << compArea << std::endl;

			if(compArea > maxArea)
			{
				maxArea = compArea;
				muscle_component_idx = i;
			}

			// Get the bounding box of the muscle component
			cv::Rect muscle_bbox = cv::boundingRect(components[i]);

			// Print the top-left corner of the bounding box
			std::cout << "Muscle component top-left corner: (" << muscle_bbox.x << ", " << muscle_bbox.y << ")" << std::endl;
			
		}
	}



	// overlay with original image
	cv::Mat selection_layer = orig.clone();
	cv::drawContours(selection_layer, components, muscle_component_idx, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
	aia::imshow("Selection layer", selection_layer, true);
/* 	cv::addWeighted(orig, 0.8, selection_layer, 0.2, 0, orig);
	aia::imshow("Result", orig); */
}

// local (8-neighborhood) region-growing segmentation based on color difference
void aia::colorDiffSegmentation(const cv::Mat &img, cv::Scalar colorDiff)
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

