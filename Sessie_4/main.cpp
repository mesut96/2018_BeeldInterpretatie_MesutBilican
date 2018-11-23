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

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_template it  || (required) path to image }"
        "{ image_objects io || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_template=fitness_object.png --image_objects=fitness_image";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_template(parser.get<string>("image_template"));
    string imagepath_objects(parser.get<string>("image_objects"));
    if (imagepath_template.empty() || imagepath_objects.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show template image
    Mat img_templ;
    img_templ = imread(imagepath_template);
    imshow("Template image", img_templ);
    waitKey(0);

    /// Read and show image with different objects
    Mat img_objects;
    img_objects = imread(imagepath_objects);
    imshow("Multiple objects", img_objects);
    waitKey(0);

    ORB_KeypointsDetection(img_templ.clone(), img_objects.clone());
    BRISK_KeypointsDetection(img_templ.clone(), img_objects.clone());
    AKAZE_KeypointsDetection(img_templ.clone(), img_objects.clone());

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
    imshow("ORB detected keypoints template", tmpl_keypoints);
    waitKey(0);
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
    imshow("BRISK detected keypoints template", tmpl_keypoints);
    waitKey(0);
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
    imshow("AKAZE detected keypoints template", tmpl_keypoints);
    waitKey(0);
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

    line(img_matches, scene_corners[0]+tmpl.width, scene_corners[1],Scalar(0,255,0), 3);
    line(img_matches, scene_corners[1], scene_corners[2],Scalar(0,255,0), 3);
    line(img_matches, scene_corners[2], scene_corners[3],Scalar(0,255,0), 3);
    line(img_matches, scene_corners[3], scene_corners[0],Scalar(0,255,0), 3);

    imshow("tadaa "+name, img_matches);
    waitKey(0);
}


