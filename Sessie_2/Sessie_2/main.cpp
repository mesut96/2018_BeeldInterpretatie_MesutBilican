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
    Mat mask_h1;
    Mat mask_h2;
    Mat mask_s;
    Mat mask_v;
    Mat mask2;

    mask2 = Mat::zeros(img_sign.rows, img_sign.cols, CV_8UC1);

    inRange(HUE, 168, 180, mask_h1);
    inRange(HUE, 0, 10, mask_h2);
    inRange(SATURATION, 0, 255, mask_s);
    inRange(VALUE, 0, 255, mask_v);

    mask2 = ((mask_h1|mask_h1));
    imshow("Mask", mask2);
    waitKey(0);


}
