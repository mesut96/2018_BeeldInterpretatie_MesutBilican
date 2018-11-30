///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

/// Function Headers
void ORB_KeypointsDetection(Mat tmpl, Mat img);
void BRISK_KeypointsDetection(Mat tmpl, Mat img);
void AKAZE_KeypointsDetection(Mat tmpl, Mat img);
void KeypointMatching(string name, Mat tmpl, Mat img, std::vector<KeyPoint> keypoints_tmpl, std::vector<KeyPoint> keypoints_img, Mat descriptor_tmpl, Mat descriptor_img);
static void onMouseStrawberries( int event, int x, int y, int, void* );
static void onMouseBackground( int event, int x, int y, int, void* );

/// Globals
vector<Point> strawberryPixels;
vector<Point> backgroundPixels;

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_strawberry1 s1  || (required) path to image }"
        "{ image_strawberry2 s2 || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_strawberry1=strawberry1.tif --image_strawberry2=strawberry2.tif";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_strawberry1(parser.get<string>("image_strawberry1"));
    string imagepath_strawberry2(parser.get<string>("image_strawberry2"));
    if (imagepath_strawberry1.empty() || imagepath_strawberry2.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show strawberry 1
    Mat img_strawberry1;
    img_strawberry1 = imread(imagepath_strawberry1);
    namedWindow("strawberry1", WINDOW_NORMAL);
    imshow("strawberry1", img_strawberry1);
    waitKey(0);

    /// Read and show strawberry 2
    //Mat img_strawberry2;
    //img_strawberry2 = imread(imagepath_strawberry2);
    //namedWindow("strawberry2", WINDOW_NORMAL);
    //imshow("strawberry2", img_strawberry2);
    //waitKey(0);

    /// Gaussian blur images (5,5)

    /// Click click click
    std::cerr << "Click on strawberries:\n";
    setMouseCallback("strawberry1", onMouseStrawberries, 0 );
    waitKey(0);

    std::cerr << "Click on background:\n";
    setMouseCallback("strawberry1", onMouseBackground, 0 );
    waitKey(0);

    /// Training
    Mat img_hsv;
    cvtColor(img_strawberry1, img_hsv, COLOR_BGR2HSV);

    ///foreground training using HSV values as descriptor
    Mat trainingDataForeground(strawberryPixels.size(), 3, CV_32FC1);
    Mat labels_fg = Mat::ones(strawberryPixels.size(), 1, CV_32FC1);

    for(int i=0; i<strawberryPixels.size(); i++)
    {
        Vec3b descriptor = img_hsv.at<Vec3b>(strawberryPixels[i].y, strawberryPixels[i].x);
        trainingDataForeground.at<float>(i,0) = descriptor[0];
        trainingDataForeground.at<float>(i,1) = descriptor[1];
        trainingDataForeground.at<float>(i,2) = descriptor[2];
    }

    ///background training using HSV values as descriptor
    Mat trainingDataBackground(backgroundPixels.size(), 3, CV_32FC1);
    Mat labels_bg = Mat::zeros(backgroundPixels.size(), 1, CV_32FC1);

    for(int i=0; i<strawberryPixels.size(); i++)
    {
        Vec3b descriptor = img_hsv.at<Vec3b>(strawberryPixels[i].y, strawberryPixels[i].x);
        trainingDataBackground.at<float>(i,0) = descriptor[0];
        trainingDataBackground.at<float>(i,1) = descriptor[1];
        trainingDataBackground.at<float>(i,2) = descriptor[2];
    }

    ///group foreground and background




    /// nearest neighbour classifier
    //Ptr<Knearest> kNN = KNearest::create();
    //Ptr<TrainData> trainingDataKNN = TrainData::create


    //ORB_KeypointsDetection(img_templ.clone(), img_objects.clone());
    //BRISK_KeypointsDetection(img_templ.clone(), img_objects.clone());
    //AKAZE_KeypointsDetection(img_templ.clone(), img_objects.clone());

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
        return;
    }
}


static void onMouseBackground( int event, int x, int y, int, void* )
{

    if(event == EVENT_LBUTTONDOWN)
    {
        backgroundPixels.push_back(Point(x,y));
        std::cerr << Point(x,y) << "\n";
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
        return;
    }
}


void ORB_KeypointsDetection(Mat tmpl, Mat img)
{
    // detect keypoints
    //int nfeatures = 500;

    //ORB
    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints_tmpl;
    std::vector<KeyPoint> keypoints_img;
    Mat descriptor_tmpl;
    Mat descriptor_img;

    detector->detectAndCompute(tmpl, Mat(), keypoints_tmpl, descriptor_tmpl);
    detector->detectAndCompute(img, Mat(), keypoints_img, descriptor_img);

    // draw keypoints
    Mat img_keypoints;
    Mat tmpl_keypoints;

    drawKeypoints(tmpl.clone(), keypoints_tmpl, tmpl_keypoints);
    drawKeypoints(img.clone(), keypoints_img, img_keypoints);

    // show detected keypoints
    namedWindow("ORB detected keypoints template", WINDOW_NORMAL);
    imshow("ORB detected keypoints template", tmpl_keypoints);
    waitKey(0);
    namedWindow("ORB detected keypoints image", WINDOW_NORMAL);
    imshow("ORB detected keypoints image", img_keypoints);
    waitKey(0);

    KeypointMatching("ORB", tmpl, img, keypoints_tmpl, keypoints_img, descriptor_tmpl, descriptor_img);
}

void BRISK_KeypointsDetection(Mat tmpl, Mat img)
{
    // detect keypoints
    //int nfeatures = 500;

    //BRISK
    Ptr<BRISK> detector = BRISK::create();
    std::vector<KeyPoint> keypoints_tmpl;
    std::vector<KeyPoint> keypoints_img;
    Mat descriptor_tmpl;
    Mat descriptor_img;

    detector->detectAndCompute(tmpl, Mat(), keypoints_tmpl, descriptor_tmpl);
    detector->detectAndCompute(img, Mat(), keypoints_img, descriptor_img);

    // draw keypoints
    Mat img_keypoints;
    Mat tmpl_keypoints;

    drawKeypoints(tmpl.clone(), keypoints_tmpl, tmpl_keypoints);
    drawKeypoints(img.clone(), keypoints_img, img_keypoints);

    // show detected keypoints
    namedWindow("BRISK detected keypoints template", WINDOW_NORMAL);
    imshow("BRISK detected keypoints template", tmpl_keypoints);
    waitKey(0);
    namedWindow("BRISK detected keypoints image", WINDOW_NORMAL);
    imshow("BRISK detected keypoints image", img_keypoints);
    waitKey(0);

    KeypointMatching("BRISK", tmpl, img, keypoints_tmpl, keypoints_img, descriptor_tmpl, descriptor_img);
}

void AKAZE_KeypointsDetection(Mat tmpl, Mat img)
{
    // detect keypoints
    //int nfeatures = 500;

    //AKAZE
    Ptr<AKAZE> detector = AKAZE::create();
    std::vector<KeyPoint> keypoints_tmpl;
    std::vector<KeyPoint> keypoints_img;
    Mat descriptor_tmpl;
    Mat descriptor_img;

    detector->detectAndCompute(tmpl, Mat(), keypoints_tmpl, descriptor_tmpl);
    detector->detectAndCompute(img, Mat(), keypoints_img, descriptor_img);

    // draw keypoints
    Mat img_keypoints;
    Mat tmpl_keypoints;

    drawKeypoints(tmpl.clone(), keypoints_tmpl, tmpl_keypoints);
    drawKeypoints(img.clone(), keypoints_img, img_keypoints);

    // show detected keypoints
    namedWindow("AKAZE detected keypoints template", WINDOW_NORMAL);
    imshow("AKAZE detected keypoints template", tmpl_keypoints);
    waitKey(0);
    namedWindow("AKAZE detected keypoints image", WINDOW_NORMAL);
    imshow("AKAZE detected keypoints image", img_keypoints);
    waitKey(0);

    KeypointMatching("AKAZE", tmpl, img, keypoints_tmpl, keypoints_img, descriptor_tmpl, descriptor_img);
}

void KeypointMatching(string name, Mat tmpl, Mat img, std::vector<KeyPoint> keypoints_tmpl, std::vector<KeyPoint> keypoints_img, Mat descriptor_tmpl, Mat descriptor_img)
{
    float GOOD_MATCH_PERCENT = 0.15f;

    // find matches
    BFMatcher matcher(NORM_L2);
    std::vector<DMatch> matches;
    matcher.match(descriptor_tmpl, descriptor_img, matches);

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin()+numGoodMatches, matches.end());

    // draw matches
    Mat img_matches;
    drawMatches(tmpl, keypoints_tmpl, img, keypoints_img, matches, img_matches);

    namedWindow(name, WINDOW_NORMAL);
    imshow(name, img_matches);
    waitKey(0);

    // Extract location of good matches
    std::vector<Point2f> points_tmp, points_img;

    for( size_t i = 0; i < matches.size(); i++ )
    {
      points_tmp.push_back( keypoints_tmpl[ matches[i].queryIdx ].pt );
      points_img.push_back( keypoints_img[ matches[i].trainIdx ].pt );
    }

    Mat homography = findHomography( points_tmp, points_img, RANSAC );

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = CvPoint(tmpl.cols);
    obj_corners[2] = cvPoint(tmpl.cols, tmpl.rows);
    obj_corners[3] = cvPoint(0, tmpl.rows);
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, homography);

    // translate corners over tmpl width
    scene_corners[0].x += tmpl.cols;
    scene_corners[1].x += tmpl.cols;
    scene_corners[2].x += tmpl.cols;
    scene_corners[3].x += tmpl.cols;

    line(img_matches, scene_corners[0], scene_corners[1],Scalar(0,255,0), 3);
    line(img_matches, scene_corners[1], scene_corners[2],Scalar(0,255,0), 3);
    line(img_matches, scene_corners[2], scene_corners[3],Scalar(0,255,0), 3);
    line(img_matches, scene_corners[3], scene_corners[0],Scalar(0,255,0), 3);

    namedWindow("tadaa "+name, WINDOW_NORMAL);
    imshow("tadaa "+name, img_matches);
    waitKey(0);
}
