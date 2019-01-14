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
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace ml;

/// Function Headers
Mat resizeImage(Mat image);
Ptr<SVM> trainDataset(string path_dataset);
float classifyLeaf(Mat image, Ptr<SVM> svm);
vector<float> featureExtraction(Mat image, bool test);
Mat preProcessing(Mat image);

/// Globals
int SHOW = 0;

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ folder_trainingData tr  || (required) path to folder }"
        "{ folder_testData te  || (required) path to folder }"
        "{ pre_processing pr || (optional) if doing random internet images}"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --folder_trainingData=Trainingdata --folder_testData=Testdata (--pre_processing=1)" << endl;

        return 0;
    }

    /// Collect data from arguments
    string folder_trainingData(parser.get<string>("folder_trainingData"));
    string folder_testData(parser.get<string>("folder_testData"));
    int pre_processingB(parser.get<int>("pre_processing"));

    ifstream test1(folder_trainingData);    // check if folder exists
    if(!test1)
    {
        cerr << "Folder of training data does not exist!" << endl;
        parser.printMessage();
        return -1;
    }

    ifstream test2(folder_testData);        // check if folder exists
    if(!test2)
    {
        cerr << "Folder of test data does not exist!" << endl;
        parser.printMessage();
        return -1;
    }

    if (folder_trainingData.empty()||folder_testData.empty())
    {
        cerr << "Please run with folders.\n";
        parser.printMessage();
        return -1;
    }

    if(pre_processingB!=1)
        pre_processingB = 0;


    ///######################### Training #########################
    Ptr<SVM> svm;
    svm = trainDataset(folder_trainingData);

    map<int, string> leafLookUpTable;
    leafLookUpTable[0] = "Big-fruited Holly";
    leafLookUpTable[1] = "tangerine";
    leafLookUpTable[2] = "Chinese redbud";
    leafLookUpTable[3] = "Japanese maple";
    leafLookUpTable[4] = "goldenrain tree";
    leafLookUpTable[5] = "Nanmu";


    ///######################### Reading image #########################
    while(1)
    {
        string image_path;
        cout << "Please enter path of the image:" << endl;
        cin >> image_path;

        string full_path = folder_testData+"/"+image_path;

        if(image_path=="q")
          return 0;         // Press q to exit

        ifstream test(full_path);
        if(!test)
        {
            cout << "This image does not exist!" << endl;
        }
        else
        {
            Mat image;

            image = imread(full_path);

            if(pre_processingB)
            {
                image = preProcessing(image);
            }

            namedWindow("leaf", WINDOW_NORMAL);
            imshow("leaf", resizeImage(image));

            int klasse;
            klasse = round(classifyLeaf(image, svm));   // classify leaf
            cout << "Leaf: " << leafLookUpTable[klasse] << endl;
            waitKey(1);
        }
    }

    return 0;

}

Ptr<SVM> trainDataset(string path_dataset)
{
//    string path_dataset = "Trainingdata";

    string path_set00 = path_dataset + "/Big-fruited_Holly/*.jpg";
    string path_set01 = path_dataset + "/tangerine/*.jpg";
    string path_set02 = path_dataset + "/Chinese_redbud/*.jpg";
    string path_set03 = path_dataset + "/Japanese_maple/*.jpg";
    string path_set04 = path_dataset + "/goldenrain_tree/*.jpg";
    string path_set05 = path_dataset + "/Nanmu/*.jpg";

    vector<string> path_sets;
    path_sets.push_back(path_set00);
    path_sets.push_back(path_set01);
    path_sets.push_back(path_set02);
    path_sets.push_back(path_set03);
    path_sets.push_back(path_set04);
    path_sets.push_back(path_set05);

    vector<String> fn;
    vector<Mat> data;

    vector<Mat> trainingdata;
    vector<Mat> labeldata;

    cerr << "Extracting features ... " << endl;

    for(size_t i=0; i<path_sets.size(); i++)    // doorloop verschillende mappen in Trainingdata
    {
        glob(path_sets[i], fn, true);           // doorloop alle afbeeldingen in 1 map

        vector<vector<float>> descriptor;
        // de features die we gebruiken zijn het volgende:
        // area, perimeter, width, height, rectangularity, circularity, equidiameter, 3x hsv_mean, 3x hsv_stddev
        // totaal: 13 features

        for(size_t j=0; j<fn.size(); j++)
        {
            Mat im = imread(fn[j]);

            vector<float> descript = featureExtraction(im, 0);  // steek de gevonden features van 1 beeld in een vector
            descriptor.push_back(descript);                     // steek de vector in de vector van de klasse
        }

        Mat trainingdata_single(descriptor.size(), descriptor[0].size(), CV_32FC1);
        Mat labeldata_single = Mat::ones(descriptor.size(), 1, CV_32SC1);

        for(size_t k=0; k<descriptor.size(); k++)
        {
            for(size_t l=0; l<descriptor[0].size(); l++)
            {
                trainingdata_single.at<float>(k,l) = descriptor[k][l];  // vorm de descriptor vector om in een training matrix
            }
        }

//        cerr << "trainingdata_single : " << trainingdata_single << endl;

        trainingdata.push_back(trainingdata_single);    // de training matrix van verschillende klassne steken we in en vector
        labeldata_single *= i;
        labeldata.push_back(labeldata_single);          // hetzelfde doen w emet de labels

//        cerr << "labeldata_single : " << labeldata_single << endl;
        labeldata[i] = labeldata_single;

//        cerr << "descriptor size: " << descriptor.size() << endl;
//        cerr << "trainingdata : " << trainingdata[i] << endl;
//        cerr << "features van klasse: " << labeldata[i] << endl;

        cerr << "Features class " << i << " DONE" << endl;
    }

    // Groepeer alle klassen
    Mat training_data;
    Mat labels;

    vconcat(trainingdata, training_data);
    vconcat(labeldata, labels);

    cerr << "Training a Support Vector Machine Classifier ... " << endl;

    Ptr<SVM> svm = SVM::create();
    //ml::ParamGrid Cgrid
    //svm->trainAuto(training_data, 10, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid);

    svm->setP(1e-3);
//    svm->setC(1000);
//    svm->setDegree(3);
//    svm->setGamma(1);
    svm->setType(SVM::EPS_SVR);
    svm->setKernel(SVM::INTER);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1e5, 1e-6));
    svm->train(training_data, ROW_SAMPLE, labels);
//    svm->trainAuto(ml::TrainData::create(training_data, ROW_SAMPLE, labels));
    cerr << "Training a Support Vector Machine Classifier DONE " << endl;

    return svm;
}

float classifyLeaf(Mat image, Ptr<SVM> svm)
{
    vector<float> features;
    Mat label_test;
    Mat data_test(1,13,CV_32FC1);

    features = featureExtraction(image, SHOW);
    for(size_t t=0; t<features.size(); t++)
    {
        data_test.at<float>(0,t) = features[t];
    }


    svm->predict(data_test, label_test);

    cerr << "Predicted label: " << label_test <<endl;

    return label_test.at<float>(0);

}

vector<float> featureExtraction(Mat image, bool test)
{
    ///######################### Pre-processing #########################

    /// to grayscale
    Mat image_gray;
    cvtColor(image.clone(), image_gray, COLOR_BGR2GRAY);

    if(test)
    {
        namedWindow("gray", WINDOW_NORMAL);
        imshow("gray", resizeImage(image_gray));
        waitKey(0);
    }

    /// image smoothing
    Mat image_smooth;

    GaussianBlur(image_gray, image_smooth, Size(25,25), 0);

    if(test)
    {
        namedWindow("smooth", WINDOW_NORMAL);
        imshow("smooth", resizeImage(image_smooth));
        waitKey(0);
    }


    /// thresholding
    Mat image_thresh;
    threshold(image_smooth, image_thresh, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);

    if(test)
    {
        namedWindow("threshold", WINDOW_NORMAL);
        imshow("threshold", resizeImage(image_thresh));
        waitKey(0);
    }

    /// closing
    Mat image_closed;
    dilate(image_thresh, image_closed, Mat(), Point(-1,-1), 20);
    erode(image_closed, image_closed, Mat(), Point(-1,-1), 20);

    if(test)
    {
        namedWindow("closed", WINDOW_NORMAL);
        imshow("closed", resizeImage(image_closed));
        waitKey(0);
    }

    /// boundary extraction
    Mat image_contour = Mat::zeros(image.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> cnt;

    findContours(image_closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    cnt = contours[0];

    for (size_t i = 0; i<contours.size(); i++)
    {
        drawContours(image_contour, contours, -1, 255, 5);
    }

    if(test)
    {
        namedWindow("contour", WINDOW_NORMAL);
        imshow("contour", resizeImage(image_contour));
        waitKey(0);
    }

    Mat image_contour_inv;
    Mat image_origin_cont;

    bitwise_not(image_contour, image_contour_inv);

    image_gray.copyTo(image_origin_cont, image_contour_inv);

    if(test)
    {
        namedWindow("original + contour", WINDOW_NORMAL);
        imshow("original + contour", resizeImage(image_origin_cont));
        waitKey(0);
    }


    ///######################### Feature extraction #########################

    /// Shape based features
    Moments M;
    M = moments(cnt);

    double area;
    area = contourArea(cnt);

    double perimeter;
    perimeter = arcLength(cnt, true);

    Rect rect;
    double width, height;
    rect = boundingRect(cnt);
    height = rect.height;
    width = rect.width;

    double rectangularity;
    rectangularity = (width*height)/area;

    double circularity;
    circularity = (perimeter*perimeter)/area;

    double equi_diameter;
    equi_diameter = sqrt(4*area/M_PI);

    /// Color based features
    Mat image_masked;

    image.copyTo(image_masked, image_closed);

    Mat image_hsv;
    Mat hsv[3];
    Mat hue, saturation, value;

    cvtColor(image_masked, image_hsv,CV_BGR2HSV);
    split(image_hsv, hsv);
    hue = hsv[0];
    saturation = hsv[1];
    value = hsv[2];


    Scalar hsv_mean, hsv_stddev;

    meanStdDev(image_hsv, hsv_mean, hsv_stddev, image_closed);

    vector<float> features;
    features.push_back((float)area);
    features.push_back((float)perimeter);
    features.push_back((float)width);
    features.push_back((float)height);
    features.push_back((float)rectangularity);
    features.push_back((float)circularity);
    features.push_back((float)equi_diameter);
    features.push_back((float)hsv_mean[0]);
    features.push_back((float)hsv_mean[1]);
    features.push_back((float)hsv_mean[2]);
    features.push_back((float)hsv_stddev[0]);
    features.push_back((float)hsv_stddev[1]);
    features.push_back((float)hsv_stddev[2]);

     if(test)
     {
         cout << "area: " << area <<endl;
         cout << "perimeter: " << perimeter <<endl;
         cout << "width: " << width <<endl;
         cout << "height: " << height <<endl;
         cout << "rectangularity: " << rectangularity <<endl;
         cout << "circularity: " << circularity <<endl;
         cout << "equi_diameter: " << equi_diameter <<endl;
         cout << "average hue: " << hsv_mean[0] <<endl;
         cout << "average saturation: " << hsv_mean[1] <<endl;
         cout << "average value: " << hsv_mean[2] <<endl;
         cout << "std. dev. hue: " << hsv_stddev[0] <<endl;
         cout << "std. dev. saturation: " << hsv_stddev[1] <<endl;
         cout << "std. dev. value: " << hsv_stddev[2] <<endl;
     }

    return features;
}

Mat preProcessing(Mat image) // voor random internet foto's
{
    Mat resized;
    resize(image.clone(), resized, Size(1600,1200));
    namedWindow("resized", WINDOW_NORMAL);
    imshow("resized", resizeImage(resized));
    waitKey(0);

    Mat smooth;
    GaussianBlur(resized, smooth, Size(69,69),0);
    namedWindow("smooth", WINDOW_NORMAL);
    imshow("smooth", resizeImage(smooth));
    waitKey(0);

    Mat hsv_img;    // in hsv-kleurruimte op groen filteren
    Mat hsv[3];
    Mat hue, sat, val;
    cvtColor(smooth, hsv_img, COLOR_BGR2HSV);
    split(hsv_img, hsv);

    hue = hsv[0];
    sat = hsv[1];
    val = hsv[2];

//    namedWindow("hsv_img", WINDOW_NORMAL);
//    imshow("hsv_img", resizeImage(hsv_img));
//    waitKey(0);
//
//    namedWindow("hue", WINDOW_NORMAL);
//    imshow("hue", resizeImage(hue));
//    waitKey(0);
//
//    namedWindow("sat", WINDOW_NORMAL);
//    imshow("sat", resizeImage(sat));
//    waitKey(0);
//
//    namedWindow("val", WINDOW_NORMAL);
//    imshow("val", resizeImage(val));
//    waitKey(0);

    Mat mask;
    inRange(hsv_img, Scalar(50,15,100), Scalar(100,255,255), mask);
    namedWindow("mask", WINDOW_NORMAL);
    imshow("mask", resizeImage(mask));
    waitKey(0);

    Mat closing;
    dilate(mask, closing, Mat(), Point(-1,-1), 33);
    erode(closing, closing, Mat(), Point(-1,-1), 33);
    namedWindow("closing", WINDOW_NORMAL);
    imshow("closing", resizeImage(closing));
    waitKey(0);

    Mat clean = Mat::ones(resized.size(), CV_8UC3);
    resized.copyTo(clean, mask);
    namedWindow("clean", WINDOW_NORMAL);
    imshow("clean", resizeImage(clean));
    waitKey(0);

    Mat inverse_mask;   // zwarte achtergrond wit maken
    bitwise_not(mask, inverse_mask);
    clean.setTo(Scalar(255,255,255), inverse_mask);
    namedWindow("clean", WINDOW_NORMAL);
    imshow("clean", resizeImage(clean));
    waitKey(0);

    return clean;
}

Mat resizeImage(Mat image)
{
    Mat image_resized;

    resize(image, image_resized, Size(), 0.5, 0.5);

    return image_resized;
}
