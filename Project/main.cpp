/// Mesut Bilican
/// Project Beeldinterpretatie
/// Leaf classificator

///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "math.h"
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
using namespace ml;

/// Function Headers
Mat resizeImage(Mat image);
//static void onMouseStrawberries( int event, int x, int y, int, void* );
//static void onMouseBackground( int event, int x, int y, int, void* );

/// Globals
//vector<Point> strawberryPixels;
//vector<Point> backgroundPixels;
//Mat img_strawberry;

int main( int argc, char** argv )
{
    ///######################### Reading image #########################

    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_leaf il  || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_leaf=1013.jpg";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_leaf(parser.get<string>("image_leaf"));

    if (imagepath_leaf.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show leaf
    Mat image_leaf;
    image_leaf = imread(imagepath_leaf);
    namedWindow("leaf", WINDOW_NORMAL);
    imshow("leaf", resizeImage(image_leaf));
    waitKey(0);

    ///######################### Pre-processing #########################

    /// to grayscale
    Mat image_gray;
    cvtColor(image_leaf.clone(), image_gray, COLOR_BGR2GRAY);

    namedWindow("gray", WINDOW_NORMAL);
    imshow("gray", resizeImage(image_gray));
    waitKey(0);

    /// image smoothing
    Mat image_smooth;

    GaussianBlur(image_gray, image_smooth, Size(25,25), 0);

    namedWindow("smooth", WINDOW_NORMAL);
    imshow("smooth", resizeImage(image_smooth));
    waitKey(0);

    /// thresholding
    Mat image_thresh;
    threshold(image_smooth, image_thresh, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);

    namedWindow("threshold", WINDOW_NORMAL);
    imshow("threshold", resizeImage(image_thresh));
    waitKey(0);

    /// closing
    Mat image_closed;
    dilate(image_thresh, image_closed, Mat(), Point(-1,-1), 5);
    erode(image_closed, image_closed, Mat(), Point(-1,-1), 5);

    namedWindow("closed", WINDOW_NORMAL);
    imshow("closed", resizeImage(image_closed));
    waitKey(0);

    /// boundary extraction
    Mat image_contour = Mat::zeros(image_leaf.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> cnt;

    findContours(image_closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    cnt = contours[0];

    for (size_t i = 0; i<contours.size(); i++)
    {
        drawContours(image_contour, contours, -1, 255, 5);
    }

    namedWindow("contour", WINDOW_NORMAL);
    imshow("contour", resizeImage(image_contour));
    waitKey(0);

    Mat image_contour_inv;
    Mat image_origin_cont;

    bitwise_not(image_contour, image_contour_inv);

    image_gray.copyTo(image_origin_cont, image_contour_inv);

    namedWindow("original + contour", WINDOW_NORMAL);
    imshow("original + contour", resizeImage(image_origin_cont));
    waitKey(0);

    ///######################### Feature extraction #########################

    /// Shape based features
    Moments M;
    M = moments(cnt);
    //cerr << "moments: " << M << endl;

    double area;
    area = contourArea(cnt);
    cerr << "area: " << area << endl;

    double perimeter;
    perimeter = arcLength(cnt, true);
    cerr << "perimeter: " << perimeter << endl;

    Rect rect;
    double width, height;
    rect = boundingRect(cnt);
    height = rect.height;
    width = rect.width;

    double rectangularity;
    rectangularity = (width*height)/area;
    cerr << "rectangularity: " << rectangularity << endl;

    double circularity;
    circularity = (perimeter*perimeter)/area;
    cerr << "circularity: " << circularity << endl;

    double equi_diameter;
    equi_diameter = sqrt(4*area/M_PI);
    cerr << "equi_diameter: " << equi_diameter << endl;

    /// Color based features
    Mat image_masked;

    image_leaf.copyTo(image_masked, image_closed);

//    Mat bgr[3];
//    Mat blue, green, red;
//
//    split(image_masked,bgr);
//    blue = bgr[0];
//    green = bgr[1];
//    red = bgr[2];
//
//    namedWindow("blue", WINDOW_NORMAL);
//    imshow("blue", resizeImage(blue));
//    waitKey(0);
//
//    namedWindow("green", WINDOW_NORMAL);
//    imshow("green", resizeImage(green));
//    waitKey(0);
//
//    namedWindow("red", WINDOW_NORMAL);
//    imshow("red", resizeImage(red));
//    waitKey(0);

    Mat image_hsv;
    Mat hsv[3];
    Mat hue, saturation, value;

    cvtColor(image_masked, image_hsv,CV_BGR2HSV);
    split(image_hsv, hsv);
    hue = hsv[0];
    saturation = hsv[1];
    value = hsv[2];

    namedWindow("hue", WINDOW_NORMAL);
    imshow("hue", resizeImage(hue));
    waitKey(0);

    namedWindow("saturation", WINDOW_NORMAL);
    imshow("saturation", resizeImage(saturation));
    waitKey(0);

    namedWindow("value", WINDOW_NORMAL);
    imshow("value", resizeImage(value));
    waitKey(0);

    Scalar hsv_mean, hsv_stddev;

    meanStdDev(image_hsv, hsv_mean, hsv_stddev, image_closed);
    cerr << "hue: " << hsv_mean[0] << "    stddev: " << hsv_stddev[0] << endl;
    cerr << "sat: " << hsv_mean[1] << "    stddev: " << hsv_stddev[1] << endl;
    cerr << "val: " << hsv_mean[2] << "    stddev: " << hsv_stddev[2] << endl;


//    /// Gaussian blur
//    Mat gaussBlur;
//    GaussianBlur(img_strawberry, gaussBlur, Size(5, 5), 0);
//    namedWindow("blurred strawberries", WINDOW_NORMAL);
//    imshow("blurred strawberries", img_strawberry);
//
//    /// Click click click
//    cerr << "##############################################" << endl;
//    cerr << "          Click on strawberries:\n";
//    cerr << "##############################################" << endl;
//    setMouseCallback("blurred strawberries", onMouseStrawberries, 0 );
//    waitKey(0);
//
//    cerr << "##############################################" << endl;
//    cerr << "           Click on background:\n";
//    cerr << "##############################################" << endl;
//    setMouseCallback("blurred strawberries", onMouseBackground, 0 );
//    waitKey(0);
//
//    /// Descriptors
//    Mat img_hsv;
//    cvtColor(img_strawberry, img_hsv, COLOR_BGR2HSV);
//
//    ///foreground training using HSV values as descriptor
//    Mat trainingDataForeground(strawberryPixels.size(), 3, CV_32FC1);
//    Mat labels_fg = Mat::ones(strawberryPixels.size(), 1, CV_32SC1); // Fixed by bug detective Dries en Simon
//
//    for(size_t i=0; i<strawberryPixels.size(); i++)
//    {
//        Vec3b descriptor = img_hsv.at<Vec3b>(strawberryPixels[i].y, strawberryPixels[i].x);
//        trainingDataForeground.at<float>(i,0) = descriptor[0];
//        trainingDataForeground.at<float>(i,1) = descriptor[1];
//        trainingDataForeground.at<float>(i,2) = descriptor[2];
//    }
//
//    ///background training using HSV values as descriptor
//    Mat trainingDataBackground(backgroundPixels.size(), 3, CV_32FC1);
//    Mat labels_bg = Mat::zeros(backgroundPixels.size(), 1, CV_32SC1); // Fixed by bug detective Dries en Simon
//
//    for(size_t i=0; i<backgroundPixels.size(); i++)
//    {
//        Vec3b descriptor = img_hsv.at<Vec3b>(backgroundPixels[i].y, backgroundPixels[i].x);
//        trainingDataBackground.at<float>(i,0) = descriptor[0];
//        trainingDataBackground.at<float>(i,1) = descriptor[1];
//        trainingDataBackground.at<float>(i,2) = descriptor[2];
//    }
//
//    ///group foreground and background
//    Mat trainingData;
//    Mat labels;
//
//    vconcat(trainingDataForeground, trainingDataBackground, trainingData);
//    //cerr << trainingData << endl;
//    vconcat(labels_fg, labels_bg, labels);
//    //cerr << labels << endl;
//    //cout << trainingData.size() << endl;
//    //cout << labels.size() << endl;
//
//
//    ///Training
//    cerr << "Training a 1 Nearest Neighbour Classifier ... " << endl;
//    cerr << "##############################################" << endl;
//    Ptr<KNearest> kNN = KNearest::create();
//    Ptr<TrainData> trainingDataKNN = TrainData::create(trainingData, ROW_SAMPLE, labels);
//    kNN->setIsClassifier(true);
//    kNN->setAlgorithmType(KNearest::BRUTE_FORCE);
//    kNN->setDefaultK(3);
//    kNN->train(trainingDataKNN);
//
//    cerr << "Training a Normal Bayes Classifier ... " << endl;
//    Ptr<NormalBayesClassifier> normalBayes = NormalBayesClassifier::create();
//    normalBayes->train(trainingData, ROW_SAMPLE, labels);
//
//    cerr << "Training a Support Vector Machine Classifier ... " << endl;
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::LINEAR);
//    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//    svm->train(trainingData, ROW_SAMPLE, labels);
//
//    cerr << "Running the classifier on the input image and creating  masks ..." << endl;
//    Mat labels_kNN, labels_normalBayes, labels_SVM;
//    Mat mask_kNN = Mat::zeros(img_strawberry.rows, img_strawberry.cols, CV_8UC1);
//    Mat mask_normalBayes = Mat::zeros(img_strawberry.rows, img_strawberry.cols, CV_8UC1);
//    Mat mask_SVM = Mat::zeros(img_strawberry.rows, img_strawberry.cols, CV_8UC1);
//
//    for(int i=0; i<img_strawberry.rows; i++)
//    {
//        for(int j=0; j<img_strawberry.cols; j++)
//        {
//            Vec3b pixelvalue = img_hsv.at<Vec3b>(i,j);
//            Mat data_test(1,3,CV_32FC1);
//            data_test.at<float>(0,0) = pixelvalue[0];
//            data_test.at<float>(0,1) = pixelvalue[1];
//            data_test.at<float>(0,2) = pixelvalue[2];
//
//            kNN->findNearest(data_test, kNN->getDefaultK(), labels_kNN);
//            normalBayes->predict(data_test, labels_normalBayes);
//            svm->predict(data_test, labels_SVM);
//
//            mask_kNN.at<uchar>(i,j) = labels_kNN.at<float>(0,0);
//            mask_normalBayes.at<uchar>(i,j) = labels_normalBayes.at<int>(0,0);
//            mask_SVM.at<uchar>(i,j) = labels_SVM.at<float>(0,0);
//        }
//    }
//
//    /// Visualize masks
//    namedWindow("Segmentation KNearest", WINDOW_NORMAL);
//    imshow("Segmentation KNearest", mask_kNN*255);
//    namedWindow("Segmentation Normal Bayes", WINDOW_NORMAL);
//    imshow("Segmentation Normal Bayes", mask_normalBayes*255);
//    namedWindow("Segmentation Support Vector Machine", WINDOW_NORMAL);
//    imshow("Segmentation Support Vector Machine", mask_SVM*255);
//    waitKey(0);
//
//    erode(mask_kNN, mask_kNN, Mat(), Point(-1,-1), 1);
//    dilate(mask_kNN, mask_kNN, Mat(), Point(-1,-1), 2);
//
//    erode(mask_normalBayes, mask_normalBayes, Mat(), Point(-1,-1), 1);
//    dilate(mask_normalBayes, mask_normalBayes, Mat(), Point(-1,-1), 2);
//
//    erode(mask_SVM, mask_SVM, Mat(), Point(-1,-1), 1);
//    dilate(mask_SVM, mask_SVM, Mat(), Point(-1,-1), 2);
//
//    /// Visualize resulting foreground pixels
//    Mat result_kNN, result_normalBayes, result_SVM;
//
//    bitwise_and(img_strawberry, img_strawberry, result_kNN, mask_kNN);
//    bitwise_and(img_strawberry, img_strawberry, result_normalBayes, mask_normalBayes);
//    bitwise_and(img_strawberry, img_strawberry, result_SVM, mask_SVM);
//
//    namedWindow("KNearest", WINDOW_NORMAL);
//    imshow("KNearest", result_kNN);
//    waitKey(0);
//    namedWindow("Normal Bayes", WINDOW_NORMAL);
//    imshow("Normal Bayes", result_normalBayes);
//    waitKey(0);
//    namedWindow("Support Vector Machine", WINDOW_NORMAL);
//    imshow("Support Vector Machine", result_SVM);
//    waitKey(0);

    return 0;

}

Mat resizeImage(Mat image)
{
    Mat image_resized;

    resize(image, image_resized, Size(), 0.5, 0.5);

    return image_resized;
}
