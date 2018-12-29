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
Ptr<SVM> trainDataset();
int classifyLeaf(Mat image, Ptr<SVM> svm);
vector<float> featureExtraction(Mat image);

/// Globals


int main( int argc, char** argv )
{
    ///######################### Training #########################
    Ptr<SVM> svm;
    svm = trainDataset();

    map<int, string> leafLookUpTable;
    leafLookUpTable[0] = "Big-fruited Holly";
    leafLookUpTable[1] = "castor aralia";
    leafLookUpTable[2] = "Chinese cinnamon";
    leafLookUpTable[3] = "Chinese redbud";
    leafLookUpTable[4] = "deodar";
    leafLookUpTable[5] = "goldenrain tree";
    leafLookUpTable[6] = "Japanese maple";
    leafLookUpTable[7] = "Nanmu";
    leafLookUpTable[8] = "pubescent bamboo";
    leafLookUpTable[9] = "true indigo";
    leafLookUpTable[10] = "tangerine";

    ///######################### Reading image #########################
    while(1)
    {
        string image_path;
        cout << "Please enter path of the image:" << endl;
        cin >> image_path;


        if(image_path=="q")
          return 0;    // Press q to exit

        ifstream test(image_path);
        if(!test)
        {
            cout << "This image does not exist!" << endl;
        }
        else
        {
            Mat image;
            image = imread(image_path);
            namedWindow("leaf", WINDOW_NORMAL);
            imshow("leaf", resizeImage(image));
            waitKey(0);

            int klasse;
            klasse = classifyLeaf(image, svm);

//            cout << "klasse: " << klasse << endl;
            cout << "Leaf: " << leafLookUpTable[klasse] << endl;
        }
    }

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

    return 0;

}

Ptr<SVM> trainDataset()
{
    string path_dataset = "Trainingdata";

    string path_set00 = path_dataset + "/Big-fruited_Holly/*.jpg";
    string path_set01 = path_dataset + "/castor_aralia/*.jpg";
    string path_set02 = path_dataset + "/Chinese_cinnamon/*.jpg";
    string path_set03 = path_dataset + "/Chinese_redbud/*.jpg";
    string path_set04 = path_dataset + "/deodar/*.jpg";
    string path_set05 = path_dataset + "/goldenrain_tree/*.jpg";
    string path_set06 = path_dataset + "/Japanese_maple/*.jpg";
    string path_set07 = path_dataset + "/Nanmu/*.jpg";
    string path_set08 = path_dataset + "/pubescent_bamboo/*.jpg";
    string path_set09 = path_dataset + "/true_indigo/*.jpg";
    string path_set10 = path_dataset + "/tangerine/*.jpg";

    vector<string> path_sets;
    path_sets.push_back(path_set00);
    path_sets.push_back(path_set01);
    path_sets.push_back(path_set02);
//    path_sets.push_back(path_set03);
//    path_sets.push_back(path_set04);
//    path_sets.push_back(path_set05);
//    path_sets.push_back(path_set06);
//    path_sets.push_back(path_set07);
//    path_sets.push_back(path_set08);
//    path_sets.push_back(path_set09);
//    path_sets.push_back(path_set10);

    vector<String> fn;
    vector<Mat> data;

    vector<Mat> trainingdata;
    vector<Mat> labeldata;

    cerr << "Extracting features ... " << endl;

    for(size_t i=0; i<path_sets.size(); i++)    // doorloop verschillende mappen
    {
        glob(path_sets[i], fn, true);           // doorloop alle afbeeldingen in één map

        vector<vector<float>> descriptor;
        // de features die we gebruiken zijn het volgende:
        // area, perimeter, width, height, rectangularity, circularity, equidiameter, 3x hsv_mean, 3x hsv_stddev
        // totaal: 13 features

        for(size_t j=0; j<fn.size(); j++)
        {
            Mat im = imread(fn[j]);

//            imshow("test", resizeImage(im));
//            waitKey(0);

            vector<float> descript = featureExtraction(im);    // steek de gevonden features van één beeld in een vector
            descriptor.push_back(descript);
        }

        Mat trainingdata_single(descriptor.size(), descriptor[0].size(), CV_32FC1);
        Mat labeldata_single = Mat::ones(descriptor.size(), 1, CV_32SC1);

        for(size_t k=0; k<descriptor.size(); k++)
        {
            for(size_t l=0; l<descriptor[0].size(); l++)
            {
                trainingdata_single.at<float>(k,l) = descriptor[k][l];  // steek de descriptoren in een training Mat
            }
        }

//        cerr << "trainingdata_single : " << trainingdata_single << endl;

        trainingdata.push_back(trainingdata_single);
        labeldata_single *= i;
        labeldata.push_back(labeldata_single);

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

    //svm->trainAuto(training_data, 10, )

    Ptr<SVM> svm = SVM::create();
    svm->setP(0.1);
    svm->setType(SVM::EPS_SVR);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(training_data, ROW_SAMPLE, labels);
    cerr << "Training a Support Vector Machine Classifier DONE " << endl;

    return svm;
}

int classifyLeaf(Mat image, Ptr<SVM> svm)
{
    vector<float> features;
    Mat label_test;
    Mat data_test(1,5,CV_32FC1);

    features = featureExtraction(image);
    for(size_t t=0; t<features.size(); t++)
    {
        data_test.at<float>(0,t) = features[t];
    }

    cout << "area: " << features[0] << endl;
    cout << "perimeter: " << features[1] << endl;
    cout << "rect: " << features[2] << endl;
    cout << "circ: " << features[3] << endl;
    cout << "equidiameter: " << features[4] << endl;


    svm->predict(data_test, label_test);

//    cerr << "gevonden label: " << label_test <<endl;

    return label_test.at<float>(0);

}

vector<float> featureExtraction(Mat image)
{
    ///######################### Pre-processing #########################

    /// to grayscale
    Mat image_gray;
    cvtColor(image.clone(), image_gray, COLOR_BGR2GRAY);

//    namedWindow("gray", WINDOW_NORMAL);
//    imshow("gray", resizeImage(image_gray));
//    waitKey(0);

    /// image smoothing
    Mat image_smooth;

    GaussianBlur(image_gray, image_smooth, Size(25,25), 0);

//    namedWindow("smooth", WINDOW_NORMAL);
//    imshow("smooth", resizeImage(image_smooth));
//    waitKey(0);

    /// thresholding
    Mat image_thresh;
    threshold(image_smooth, image_thresh, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);

//    namedWindow("threshold", WINDOW_NORMAL);
//    imshow("threshold", resizeImage(image_thresh));
//    waitKey(0);

    /// closing
    Mat image_closed;
    dilate(image_thresh, image_closed, Mat(), Point(-1,-1), 5);
    erode(image_closed, image_closed, Mat(), Point(-1,-1), 5);

//    namedWindow("closed", WINDOW_NORMAL);
//    imshow("closed", resizeImage(image_closed));
//    waitKey(0);

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

//    namedWindow("contour", WINDOW_NORMAL);
//    imshow("contour", resizeImage(image_contour));
//    waitKey(0);

    Mat image_contour_inv;
    Mat image_origin_cont;

    bitwise_not(image_contour, image_contour_inv);

    image_gray.copyTo(image_origin_cont, image_contour_inv);

//    namedWindow("original + contour", WINDOW_NORMAL);
//    imshow("original + contour", resizeImage(image_origin_cont));
//    waitKey(0);

    ///######################### Feature extraction #########################

    /// Shape based features
    Moments M;
    M = moments(cnt);
    //cerr << "moments: " << M << endl;

    double area;
    area = contourArea(cnt);
//    cerr << "area: " << area << endl;

    double perimeter;
    perimeter = arcLength(cnt, true);
//    cerr << "perimeter: " << perimeter << endl;

    Rect rect;
    double width, height;
    rect = boundingRect(cnt);
    height = rect.height;
    width = rect.width;

    double rectangularity;
    rectangularity = (width*height)/area;
//    cerr << "rectangularity: " << rectangularity << endl;

    double circularity;
    circularity = (perimeter*perimeter)/area;
//    cerr << "circularity: " << circularity << endl;

    double equi_diameter;
    equi_diameter = sqrt(4*area/M_PI);
//    cerr << "equi_diameter: " << equi_diameter << endl;

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

//    namedWindow("hue", WINDOW_NORMAL);
//    imshow("hue", resizeImage(hue));
//    waitKey(0);

//    namedWindow("saturation", WINDOW_NORMAL);
//    imshow("saturation", resizeImage(saturation));
//    waitKey(0);

//    namedWindow("value", WINDOW_NORMAL);
//    imshow("value", resizeImage(value));
//    waitKey(0);

    Scalar hsv_mean, hsv_stddev;

    meanStdDev(image_hsv, hsv_mean, hsv_stddev, image_closed);
//    cerr << "hue: " << hsv_mean[0] << "    stddev: " << hsv_stddev[0] << endl;
//    cerr << "sat: " << hsv_mean[1] << "    stddev: " << hsv_stddev[1] << endl;
//    cerr << "val: " << hsv_mean[2] << "    stddev: " << hsv_stddev[2] << endl;

    vector<float> features;
    features.push_back((float)area);
    features.push_back((float)perimeter);
//    features.push_back((float)width);
//    features.push_back((float)height);
    features.push_back((float)rectangularity);
    features.push_back((float)circularity);
    features.push_back((float)equi_diameter);
//    features.push_back((float)hsv_mean[0]);
//    features.push_back((float)hsv_mean[1]);
//    features.push_back((float)hsv_mean[2]);
//    features.push_back((float)hsv_stddev[0]);
//    features.push_back((float)hsv_stddev[1]);
//    features.push_back((float)hsv_stddev[2]);


    return features;
}

Mat resizeImage(Mat image)
{
    Mat image_resized;

    resize(image, image_resized, Size(), 0.5, 0.5);

    return image_resized;
}
