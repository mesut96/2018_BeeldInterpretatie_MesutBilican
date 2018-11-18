///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

/// Function Headers
void MatchRechtSingle( Mat img, Mat templ );
void MatchRechtMultiple( Mat img, Mat templ );
void MatchRot( Mat img, Mat templ );
void rotate(Mat& src, double angle, Mat& dst);

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_template it  || (required) path to image }"
        "{ image_recht ire || (required) path to image }"
        "{ image_rot iro || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_template=template.jpg --image_recht=recht.jpg --image_rot=rot.jpg";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_template(parser.get<string>("image_template"));
    string imagepath_recht(parser.get<string>("image_recht"));
    string imagepath_rot(parser.get<string>("image_rot"));
    if (imagepath_template.empty() || imagepath_recht.empty() || imagepath_rot.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show template image
    Mat templ;
    templ = imread(imagepath_template);
    imshow("Template image", templ);
    waitKey(0);

    /// Read and show straight image
    Mat img_recht;
    img_recht = imread(imagepath_recht);
    imshow("recht", img_recht);
    waitKey(0);

    /// Read and show rotated image
    Mat img_rot;
    img_rot = imread(imagepath_rot);
    imshow("rot", img_rot);
    waitKey(0);

    /// Template matching
    MatchRechtSingle(img_recht, templ);
    MatchRechtMultiple(img_recht, templ);

}

void MatchRechtSingle( Mat img, Mat templ )
{
    Mat result;

    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    result.create( result_rows, result_cols, CV_32FC1 );

    matchTemplate( img, templ, result, TM_SQDIFF);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    imshow("normalized", result);
    waitKey(0);

    // Localizing the best match with minMaxLoc
    Mat img_display;
    img.copyTo( img_display );

    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    matchLoc = minLoc;

    rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,255,0), 2, 8, 0 );
    rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,255,0), 2, 8, 0 );

    imshow( "norm rechthoek", result );
    waitKey(0);

    imshow( "match recht single", img_display );
    waitKey(0);
}

void MatchRechtMultiple( Mat img, Mat templ )
{
    Mat mask, temp;
    Mat img_display;
    img.copyTo( img_display );
    Mat result;

    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    result.create( result_rows, result_cols, CV_32FC1 );

    matchTemplate( img, templ, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    threshold(result, mask, 0.8, 1, THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    mask = mask*255;
    imshow( "mask ", mask );
    waitKey(0);

    vector<vector<Point>>  contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    //check for local minima/maxima
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    for (size_t i=0; i <contours.size(); i++)
    {
        Rect region = boundingRect(contours[i]);
        Mat temp = result(region);
        Point maxLock;
        Point minLock;
        minMaxLoc(temp, NULL, NULL, NULL, &maxLoc, Mat() );
        matchLoc = maxLoc;
        rectangle(img_display, Point(region.x+ matchLoc.x, region.y + matchLoc.y), Point(matchLoc.x +region.x + templ.cols, matchLoc.y +region.y + templ.rows), Scalar(0,0,255), 2, 8, 0 );
     }

    imshow( "result multiple ", img_display );
    waitKey(0);
}

void MatchRot( Mat img, Mat templ )
{
    int maxrot = 45; // maximum angle we will look after
    int steps = 10; // amount of steps to achieve maxrot
    float stepsize = maxrot/steps; // angle of each step

    Mat img_display;
    img.copyTo( img_display );

}

// Return the rotation matrices for each rotation
// The angle parameter is expressed in degrees!
void rotate(Mat& src, double angle, Mat& dst)
{
    Point2f pt(src.cols/2., src.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}
