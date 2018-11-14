///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int max_value_H = 360/2;
const int max_value = 255;
const String window_trackbar_name = "Draw contours with trackbar";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_trackbar_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_trackbar_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_trackbar_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_trackbar_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_trackbar_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_trackbar_name, high_V);
}

int main(int argc, const char** argv)
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_sign s1  || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_sign=sign.jpg";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_sign(parser.get<string>("image_sign"));
       if (imagepath_sign.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show sign
    Mat img_sign;
    img_sign = imread(imagepath_sign);
    imshow("Sign", img_sign);
    waitKey(0);


    /// Split color image into BGR
    Mat bgr[3];
    Mat BLUE;
    Mat GREEN;
    Mat RED;

    split(img_sign, bgr);
    BLUE = bgr[0];
    GREEN = bgr[1];
    RED = bgr[2];

    imshow("Sign - blue", BLUE);
    waitKey(0);
    imshow("Sign - green", GREEN);
    waitKey(0);
    imshow("Sign - red", RED);
    waitKey(0);

    /// RGB segmentation
    Mat mask1;

    mask1 = Mat::zeros(img_sign.rows, img_sign.cols, CV_8UC1);

    //mask1 = (RED>150) & (RED<255);

    //threshold(RED, maskRGB, 150, 255, THRESH_BINARY);
    inRange(RED, 150, 255, mask1);

    imshow("Mask", mask1);
    waitKey(0);

    /// Show original image with mask
    Mat bgr_mask(img_sign.rows, img_sign.cols, CV_8UC3);
    Mat pixels_blue = bgr[0] & mask1;
    Mat pixels_green = bgr[1] & mask1;
    Mat pixels_red = bgr[2] & mask1;

    Mat in[] = { pixels_blue, pixels_green, pixels_red };
    int from_to[] = { 0,0, 1,1, 2,2 };

    mixChannels(in, 3, &bgr_mask, 1, from_to, 3);


    imshow("Result", bgr_mask);
    waitKey(0);

    /// Split color image into HSV
    Mat img_hsv;

    Mat hsv[3];
    Mat HUE;
    Mat SATURATION;
    Mat VALUE;

    cvtColor(img_sign, img_hsv, COLOR_BGR2HSV);
    split(img_hsv, hsv);

    HUE = hsv[0];
    SATURATION = hsv[1];
    VALUE = hsv[2];

    imshow("Sign - hue", HUE);
    waitKey(0);
    imshow("Sign - saturation", SATURATION);
    waitKey(0);
    imshow("Sign - value", VALUE);
    waitKey(0);

    ///HSV segmentation
    Mat mask_hsv1;
    Mat mask_hsv2;
    Mat mask_hsv;

    Mat mask2;

    mask2 = Mat::zeros(img_sign.rows, img_sign.cols, CV_8UC1);

    inRange(img_hsv, Scalar(168,115,100), Scalar(180, 255, 255), mask_hsv1);
    inRange(img_hsv, Scalar(0,115,100), Scalar(10, 255, 255), mask_hsv2);
    addWeighted(mask_hsv1, 1.0, mask_hsv2, 1.0, 0.0, mask_hsv);

    imshow("Mask HSV", mask_hsv);
    waitKey(0);

    /// connected components
    // erosion & dilation
    dilate(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 7);
    erode(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 7);

    erode(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 4);
    dilate(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 4);

    imshow("Mask HSV erode/dilate", mask_hsv);
    waitKey(0);

    // convex hull approach
    vector< vector<Point> > contours;

    findContours(mask_hsv.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<Point> hull;

    convexHull(contours[0], hull);

    vector<Point> biggest_blob = hull;

    for(size_t i=0; i < contours.size(); i++)
    {
        if(contourArea(contours[i]) > contourArea(biggest_blob))
        {
            convexHull(contours[i], hull);
            biggest_blob = hull;
        }
    }

    vector < vector<Point> > temp;
    Rect box = boundingRect(biggest_blob);
    temp.push_back(biggest_blob);

    drawContours(mask_hsv, temp, -1, 255, -1);

    imshow("Masked HSV contour", mask_hsv);
    waitKey(0);

    Mat img_contour = Mat::zeros(img_sign.size(), CV_8UC3);
    img_sign.copyTo(img_contour, mask_hsv);
    Mat img_contourBox = img_contour.clone();
    rectangle(img_contourBox, box, cv::Scalar(0, 0, 255));
    imshow("Boxed HSV contour", img_contourBox);
    waitKey(0);

    Mat img_cutBox = Mat::zeros(box.size(), img_contour.type());
    Mat ROI(img_contour, box);
    ROI.copyTo(img_cutBox);

    imshow("Leftover sign", img_cutBox);
    waitKey(0);

    /// Trackbars
    namedWindow("Draw contours with trackbar", WINDOW_AUTOSIZE);
    createTrackbar("Low H", window_trackbar_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_trackbar_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_trackbar_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_trackbar_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_trackbar_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_trackbar_name, &high_V, max_value, on_high_V_thresh_trackbar);
    cvtColor(img_sign, img_hsv, COLOR_BGR2HSV);
    Mat img_HSVsegment;

    while (true)
    {
        // mask
        inRange(img_hsv, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), img_HSVsegment);

        // erosion & dilattion for noise reduction and for connecting blobs
        dilate(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 7);
        erode(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 7);

        erode(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 4);
        dilate(mask_hsv, mask_hsv, Mat(), Point(-1, -1), 4);

        // find external contour
        vector< vector<Point> > contours;

        findContours(img_HSVsegment.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        convexHull(contours[0], hull);
        vector<Point> biggest_blob = hull;
        for(size_t i=0; i < contours.size(); i++)
        {
            if(contourArea(contours[i]) > contourArea(biggest_blob))
            {
                convexHull(contours[i], hull);
                biggest_blob = hull;
            }
        }
        vector < vector<Point> > temp;
        temp.push_back(biggest_blob);

        drawContours(mask_hsv, temp, -1, 255, -1);

        // apply mask to image
        Mat img_contour(img_sign.size(), CV_8UC3);
        img_sign.copyTo(img_contour, img_HSVsegment);

        imshow(window_trackbar_name, img_contour);

        char key = (char) waitKey(1);
        if (key == 'q' || key == 27) // ESC
        {
            break;
        }
}
}
