///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image i  || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameter: --image=imageColorAdapted.png";
        return 0;
    }

    /// Collect data from arguments
    string imagepath(parser.get<string>("image"));
    if (imagepath.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read an show image
    Mat img;
    img = imread(imagepath);
    imshow("Adapted color image", img);
    waitKey(0);

    /// Segment skin color
    Mat bgr[3];
    Mat BLUE;
    Mat GREEN;
    Mat RED;

    split(img, bgr);
    BLUE = bgr[0];
    GREEN = bgr[1];
    RED = bgr[2];

    Mat mask;
    mask = (RED>95) & (GREEN>40) & (BLUE>20) & ((max(RED,max(GREEN,BLUE)) - min(RED,min(GREEN,BLUE)))>15) & (abs(RED-GREEN)>15) & (RED>GREEN) & (RED>BLUE);

    mask = mask*255;

    imshow("mask", mask);
    waitKey(0);

    /// Clean mask with erode and dilate
    erode(mask, mask, Mat(), Point(-1,-1), 1);
    dilate(mask, mask, Mat(), Point(-1,-1), 1);

    imshow("clean mask ", mask);
    waitKey(0);

    /// connect limbs
    dilate(mask, mask, Mat(), Point(-1,-1), 10);
    erode(mask, mask, Mat(), Point(-1,-1), 10);

    imshow("connected limbs", mask);
    waitKey(0);

    /// Convex hull approach for finding contours
    vector< vector<Point> > contours;
    vector< vector<Point> > hulls;

    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    for (size_t i=0; i<contours.size(); i++)
    {
        vector<Point> hull;
        convexHull(contours[i], hull);
        hulls.push_back(hull);
    }

    //draw filled hulls
    drawContours(mask, hulls, -1, 255, -1);

    imshow("contours", mask);
    waitKey(0);

    /// use mask on original image
    Mat bgr_mask(img.rows, img.cols, CV_8UC3);
    Mat pixels_blue = bgr[0] & mask;
    Mat pixels_green = bgr[1] & mask;
    Mat pixels_red = bgr[2] & mask;

    Mat in[] = { pixels_blue, pixels_green, pixels_red };
    int from_to[] = { 0,0, 1,1, 2,2 };

    mixChannels(in, 3, &bgr_mask, 1, from_to, 3);

    imshow("original + mask", bgr_mask);
    waitKey(0);
}
