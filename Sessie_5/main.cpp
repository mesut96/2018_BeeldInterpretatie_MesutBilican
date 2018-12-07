///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
using namespace ml;

/// Function Headers
static void onMouseStrawberries( int event, int x, int y, int, void* );
static void onMouseBackground( int event, int x, int y, int, void* );

/// Globals
vector<Point> strawberryPixels;
vector<Point> backgroundPixels;
Mat img_strawberry;

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_strawberry is  || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_strawberry=strawberry1.tif";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_strawberry(parser.get<string>("image_strawberry"));

    if (imagepath_strawberry.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show strawberries
    img_strawberry = imread(imagepath_strawberry);
    //namedWindow("strawberries", WINDOW_NORMAL);
    //imshow("strawberries", img_strawberry);
    //waitKey(0);


    /// Gaussian blur
    Mat gaussBlur;
    GaussianBlur(img_strawberry, gaussBlur, Size(5, 5), 0);
    namedWindow("blurred strawberries", WINDOW_NORMAL);
    imshow("blurred strawberries", img_strawberry);

    /// Click click click
    cerr << "##############################################" << endl;
    cerr << "          Click on strawberries:\n";
    cerr << "##############################################" << endl;
    setMouseCallback("blurred strawberries", onMouseStrawberries, 0 );
    waitKey(0);

    cerr << "##############################################" << endl;
    cerr << "           Click on background:\n";
    cerr << "##############################################" << endl;
    setMouseCallback("blurred strawberries", onMouseBackground, 0 );
    waitKey(0);

    /// Descriptors
    Mat img_hsv;
    cvtColor(img_strawberry, img_hsv, COLOR_BGR2HSV);

    ///foreground training using HSV values as descriptor
    Mat trainingDataForeground(strawberryPixels.size(), 3, CV_32FC1);
    Mat labels_fg = Mat::ones(strawberryPixels.size(), 1, CV_32SC1); // Fixed by bug detective Dries en Simon

    for(size_t i=0; i<strawberryPixels.size(); i++)
    {
        Vec3b descriptor = img_hsv.at<Vec3b>(strawberryPixels[i].y, strawberryPixels[i].x);
        trainingDataForeground.at<float>(i,0) = descriptor[0];
        trainingDataForeground.at<float>(i,1) = descriptor[1];
        trainingDataForeground.at<float>(i,2) = descriptor[2];
    }

    ///background training using HSV values as descriptor
    Mat trainingDataBackground(backgroundPixels.size(), 3, CV_32FC1);
    Mat labels_bg = Mat::zeros(backgroundPixels.size(), 1, CV_32SC1); // Fixed by bug detective Dries en Simon

    for(size_t i=0; i<backgroundPixels.size(); i++)
    {
        Vec3b descriptor = img_hsv.at<Vec3b>(backgroundPixels[i].y, backgroundPixels[i].x);
        trainingDataBackground.at<float>(i,0) = descriptor[0];
        trainingDataBackground.at<float>(i,1) = descriptor[1];
        trainingDataBackground.at<float>(i,2) = descriptor[2];
    }

    ///group foreground and background
    Mat trainingData;
    Mat labels;

    vconcat(trainingDataForeground, trainingDataBackground, trainingData);
    //cerr << trainingData << endl;
    vconcat(labels_fg, labels_bg, labels);
    //cerr << labels << endl;
    //cout << trainingData.size() << endl;
    //cout << labels.size() << endl;


    ///Training
    cerr << "Training a 1 Nearest Neighbour Classifier ... " << endl;
    cerr << "##############################################" << endl;
    Ptr<KNearest> kNN = KNearest::create();
    Ptr<TrainData> trainingDataKNN = TrainData::create(trainingData, ROW_SAMPLE, labels);
    kNN->setIsClassifier(true);
    kNN->setAlgorithmType(KNearest::BRUTE_FORCE);
    kNN->setDefaultK(3);
    kNN->train(trainingDataKNN);

    cerr << "Training a Normal Bayes Classifier ... " << endl;
    Ptr<NormalBayesClassifier> normalBayes = NormalBayesClassifier::create();
    normalBayes->train(trainingData, ROW_SAMPLE, labels);

    cerr << "Training a Support Vector Machine Classifier ... " << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labels);

    cerr << "Running the classifier on the input image and creating  masks ..." << endl;
    Mat labels_kNN, labels_normalBayes, labels_SVM;
    Mat mask_kNN = Mat::zeros(img_strawberry.rows, img_strawberry.cols, CV_8UC1);
    Mat mask_normalBayes = Mat::zeros(img_strawberry.rows, img_strawberry.cols, CV_8UC1);
    Mat mask_SVM = Mat::zeros(img_strawberry.rows, img_strawberry.cols, CV_8UC1);

    for(int i=0; i<img_strawberry.rows; i++)
    {
        for(int j=0; j<img_strawberry.cols; j++)
        {
            Vec3b pixelvalue = img_hsv.at<Vec3b>(i,j);
            Mat data_test(1,3,CV_32FC1);
            data_test.at<float>(0,0) = pixelvalue[0];
            data_test.at<float>(0,1) = pixelvalue[1];
            data_test.at<float>(0,2) = pixelvalue[2];

            kNN->findNearest(data_test, kNN->getDefaultK(), labels_kNN);
            normalBayes->predict(data_test, labels_normalBayes);
            svm->predict(data_test, labels_SVM);

            mask_kNN.at<uchar>(i,j) = labels_kNN.at<float>(0,0);
            mask_normalBayes.at<uchar>(i,j) = labels_normalBayes.at<int>(0,0);
            mask_SVM.at<uchar>(i,j) = labels_SVM.at<float>(0,0);
        }
    }

    /// Visualize masks
    namedWindow("Segmentation KNearest", WINDOW_NORMAL);
    imshow("Segmentation KNearest", mask_kNN*255);
    namedWindow("Segmentation Normal Bayes", WINDOW_NORMAL);
    imshow("Segmentation Normal Bayes", mask_normalBayes*255);
    namedWindow("Segmentation Support Vector Machine", WINDOW_NORMAL);
    imshow("Segmentation Support Vector Machine", mask_SVM*255);
    waitKey(0);

    erode(mask_kNN, mask_kNN, Mat(), Point(-1,-1), 1);
    dilate(mask_kNN, mask_kNN, Mat(), Point(-1,-1), 2);

    erode(mask_normalBayes, mask_normalBayes, Mat(), Point(-1,-1), 1);
    dilate(mask_normalBayes, mask_normalBayes, Mat(), Point(-1,-1), 2);

    erode(mask_SVM, mask_SVM, Mat(), Point(-1,-1), 1);
    dilate(mask_SVM, mask_SVM, Mat(), Point(-1,-1), 2);

    /// Visualize resulting foreground pixels
    Mat result_kNN, result_normalBayes, result_SVM;

    bitwise_and(img_strawberry, img_strawberry, result_kNN, mask_kNN);
    bitwise_and(img_strawberry, img_strawberry, result_normalBayes, mask_normalBayes);
    bitwise_and(img_strawberry, img_strawberry, result_SVM, mask_SVM);

    namedWindow("KNearest", WINDOW_NORMAL);
    imshow("KNearest", result_kNN);
    waitKey(0);
    namedWindow("Normal Bayes", WINDOW_NORMAL);
    imshow("Normal Bayes", result_normalBayes);
    waitKey(0);
    namedWindow("Support Vector Machine", WINDOW_NORMAL);
    imshow("Support Vector Machine", result_SVM);
    waitKey(0);

    return 0;

}

static void onMouseStrawberries( int event, int x, int y, int, void* /*param*/)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        strawberryPixels.push_back(Point(x,y));
        std::cerr << Point(x,y) << "\n";
    }
    if(event == EVENT_RBUTTONDOWN)
    {
        if(strawberryPixels.size()==0)
        {
            std::cout << "no points to remove\n";
        }
        else
        {
            strawberryPixels.pop_back();
            std::cout << "removed last point\n";
        }
    }
    if(event == EVENT_MBUTTONDOWN)
    {
        std::cout << "List of strawberry points\n";
        Mat temp = img_strawberry.clone();
        for (size_t i=0; i<strawberryPixels.size(); i++)
        {
            std::cout << strawberryPixels[i] << endl;
            circle(temp,strawberryPixels[i],3, Scalar(0,0,255),-1);
            imshow("blurred strawberries", temp);
        }
    }
}


static void onMouseBackground( int event, int x, int y, int, void* )
{

    if(event == EVENT_LBUTTONDOWN)
    {
        backgroundPixels.push_back(Point(x,y));
        std::cerr << Point(x,y) << endl;
    }
    if(event == EVENT_RBUTTONDOWN)
    {
        if(backgroundPixels.size()==0)
        {
            std::cout << "no points to remove\n";
        }
        else
        {
            backgroundPixels.pop_back();
            std::cout << "removed last point\n";
        }
    }
    if(event == EVENT_MBUTTONDOWN)
    {
        std::cout << "List of background points\n";
        Mat temp = img_strawberry.clone();
        for (size_t i=0; i<backgroundPixels.size(); i++)
        {
            std::cout << backgroundPixels[i] << endl;
            circle(temp,backgroundPixels[i],3, Scalar(255,0,0),-1);
            imshow("blurred strawberries", temp);
        }
    }
}
