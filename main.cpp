#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <omp.h>

using namespace cv;
using namespace std;


int erosion_size = 0;
int dialate_size = 0;
Mat element_ero = getStructuringElement(cv::MORPH_RECT,
	cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
	cv::Point(erosion_size, erosion_size));
Mat element_dia = getStructuringElement(cv::MORPH_RECT,
	cv::Size(2 * dialate_size + 1, 2 * dialate_size + 1),
	cv::Point(dialate_size, dialate_size));


string char_to_str(char c)
{
	string s;
	s.push_back(c);
	return s;
}


string execSystemCommand(const char* cmd) 
{
	char buffer[128];
	string result = "";
	FILE* pipe = _popen(cmd, "r");
	if (!pipe) throw std::runtime_error("popen() failed!");
	try {
		while (fgets(buffer, sizeof buffer, pipe) != NULL) {
			result += buffer;
		}
	}
	catch (...) {
		_pclose(pipe);
		throw;
	}
	_pclose(pipe);
	return result;
}



// function to sort contours (insertion sort) based on
// x coordinates from left to right
void sortContours(vector <vector <Point>>& contours)
{
	for (int k = 0; k < contours.size(); k++)
	{
		vector <Point> temp = contours[k];
		int j = k - 1;
		while (j >= 0 && boundingRect(temp).x <= boundingRect(contours[j]).x)
		{
			contours[j + 1] = contours[j];
			j = j - 1;
		}
		contours[j + 1] = temp;
	}
}


string recognizeCharacter(Mat& croppedCharacter)
{
	dilate(croppedCharacter, croppedCharacter, getStructuringElement(MORPH_RECT, Size(3, 3)));
	int borderSizeR = croppedCharacter.rows / 3;
	int borderSizeC = croppedCharacter.cols / 3;
	copyMakeBorder(croppedCharacter, croppedCharacter, borderSizeR, borderSizeR, borderSizeC, borderSizeC, BORDER_ISOLATED, Scalar(255));
	resize(croppedCharacter, croppedCharacter, Size(28, 28));
	// system("cls");

	// parse python command as a string
	// this string will be run as a system command ( using execSystemCommand() )
	// and will look like:
	// python neuralnet.py x1 x2 x3 ... x784
	// where xN denotes the pixel values - passed as args to the python script
	string PythonCommand = "python neuralnet.py "; 
	for (int r = 0; r < croppedCharacter.rows; r++)
	{
		for (int c = 0; c < croppedCharacter.cols; c++)
		{
			croppedCharacter.at <uchar>(r, c) = abs(int(croppedCharacter.at<uchar>(r, c)) - 255); // invert colours
			PythonCommand += to_string(croppedCharacter.at<uchar>(r, c)) + " "; // append inverted pixel value to python command
		}
	}
	PythonCommand.pop_back();
	// cout << PythonCommand << endl;
	string out = execSystemCommand(PythonCommand.c_str());
	out.pop_back();

	return out; // return output character from the neural net
}


void postProcessImg(Mat& dilatedImg, vector <vector <Point>> contours, vector <int> selected_ROI)
{
	char output[10] = {'\0', '\0', '\0' , '\0' , '\0' , '\0' , '\0' , '\0' , '\0' , '\0' };
	Mat col;
	cvtColor(dilatedImg, col, COLOR_GRAY2BGR);

	#pragma omp parallel for
	for (int charCont = 0; charCont < selected_ROI.size(); charCont++)
	{
		Mat croppedCharacter = dilatedImg(boundingRect(contours[selected_ROI[charCont]]));
		output[charCont] = *(recognizeCharacter(croppedCharacter).c_str()); // try to recognize character
		rectangle(col, boundingRect(contours[selected_ROI[charCont]]), Scalar(0, 0, 255), 1);
		putText(col, char_to_str(output[charCont]), boundingRect(contours[selected_ROI[charCont]]).tl(), FONT_HERSHEY_PLAIN, 2, Scalar(150, 50, 255), 2);
		/*imshow("char", croppedCharacter);
		string path = "RecognitionOutput/" + to_string(charCont) + ".jpg";
		imwrite(path, croppedCharacter);
		waitKey(0);*/

	}
	
	string path = "RecognitionOutput/" + string(output) + ".jpg";
	imwrite(path, col);

	/*string imgName = "Cropped Image";
	imshow(imgName, col);
	waitKey(0);*/
}



void processSinglePlate(Mat& croppedPlate)
{
	Mat thres, ero, dia, thresInv, cany;
	threshold(croppedPlate, thres, 100, 255, THRESH_BINARY); // apply binary thresholding
	medianBlur(thres, thres, 5); // apply blur
	erode(thres, ero, element_ero); // erode the "lines" in image
	dilate(ero, dia, element_dia); // dilate the "lines" in image
	threshold(dia, thresInv, 100, 255, THRESH_BINARY_INV); // inverse thresholding
	Canny(thresInv, cany, 100, 200, 3); // appply canny edge detection

	vector <vector <Point>> contours; // vector of vectors to store coordinates making up each contour
	vector <Vec4i> hierarchy; // to store heirarchy of contours (not used in this scenario)
	findContours(cany, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0)); // find contours in the plate image
	sortContours(contours); // sort contours along x axis

	vector <int> selected_ROI;
	Rect ROI;
	Rect prevROI; prevROI.x = prevROI.width = 0;


	for (int cont = 0; cont < contours.size(); cont++)
	{
		ROI = boundingRect(contours[cont]); // select bounding rectangle around contour as reigon of intrest (ROI)
		if (
			ROI.width > 20 &&
			ROI.width < 100 &&
			ROI.height > 35 &&
			ROI.height < 150 &&
			ROI.x > prevROI.x + prevROI.width
			) // if ROI is of acceptable parameters and not overlapping with prevROI
		{
			selected_ROI.push_back(cont);
			prevROI = ROI;
		}

	}

	postProcessImg(dia, contours, selected_ROI); // draws rectangles around letters and labels them with recognized text
}



void processPlatesArray(Mat& frame, Mat& grey, vector <Rect>& plates)
{
	#pragma omp parallel for
	for (int i = 0; i < plates.size(); i++)
	{
		Mat croppedPlate = grey(plates[i]); // crop single plate from grey image

		// rectangle(frame, plates[i].tl(), plates[i].br(), Scalar(0, 255, 0), 4); // rectangle around plate on original image
		/*imshow("Image", frame);
		string path = "RecognitionOutput/" + string("gfdgdfg") + ".jpg";
		imwrite(path, frame);*/

		processSinglePlate(croppedPlate);
	}
}




void main()
{
	int numberOfImages = 0;

	cout << "\n Enter Number of Images in Resources > Plates > test directory : ";
	cin >> numberOfImages;
	
	streambuf* orig_buf = cout.rdbuf();
	cout.rdbuf(NULL);

	double start = omp_get_wtime();
	system("mkdir RecognitionOutput");

	// load video file
	// string path = "Resources/plateVideo2.mp4";
	// VideoCapture video(path);

	Mat frame, grey;
	vector <Rect> plates;


	omp_set_num_threads(8);
	omp_set_dynamic(0);
	omp_set_nested(4);
	#pragma omp parallel for private (plates, frame, grey)
	// for each image in the testing dataset
	for (int img = 0; img < numberOfImages; img++)
	{
		CascadeClassifier plateCascade;
		plateCascade.load("Resources/haarcascade_russian_plate_number.xml");

		if (plateCascade.empty())
			cout << "Error! Harr Cascade XML File Not Loaded" << endl; // print error if file not loaded

		string path = "Resources/Plates/test/plate (" + to_string(img + 1) + ").jpg"; // parse fileName

		frame = imread(path); // read image
		medianBlur(frame, frame, 7); // apply blur
		cvtColor(frame, grey, COLOR_BGR2GRAY); // convert to grayscale

		// run viola-jones plate detection algo
		// on the current image
		// all detected plates returned in plates array
		plateCascade.detectMultiScale(grey, plates, 1.1, 10);

		processPlatesArray(frame, grey, plates);

		/*imshow("Image", frame);
		plates.clear();
		waitKey(0);*/
		destroyAllWindows();
	}



	double end = omp_get_wtime();
	cout.rdbuf(orig_buf);

	cout  << "Execution Time: " << end - start << " seconds" << endl;
}

// para - 18.1869 seconds
// serial - 57.0827 seconds