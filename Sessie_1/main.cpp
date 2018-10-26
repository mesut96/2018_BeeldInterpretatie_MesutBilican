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
        "{ image_grey ig  || (required) path to image }"
        "{ image_color ic || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_grey=test.png --image_color=testColor.png";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_grey(parser.get<string>("image_grey"));
    string imagepath_color(parser.get<string>("image_color"));
    if (imagepath_grey.empty() || imagepath_color.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read an show grey image
    Mat img_ig;
    img_ig = imread(imagepath_grey);
    imshow("EAVISE logo - greyscale", img_ig);
    waitKey(0);

    /// Read an show color image
    Mat img_cl;
    img_cl = imread(imagepath_color);
    imshow("EAVISE logo - color", img_cl);
    waitKey(0);

    /// Split color image
    Mat bgr[3];
    Mat img_b;
    Mat img_g;
    Mat img_r;

    split(img_cl, bgr);
    img_b = bgr[0];
    img_g = bgr[1];
    img_r = bgr[2];

    imshow("EAVISE logo - blue", img_b);
    waitKey(0);
    imshow("EAVISE logo - green", img_g);
    waitKey(0);
    imshow("EAVISE logo - red", img_r);
    waitKey(0);

    /// Color image to greyscale
    Mat img_c2g;
    cvtColor(img_cl, img_c2g, COLOR_RGB2GRAY);
    imshow("EAVISE logo - color2grey", img_c2g);
    waitKey(0);

    /// Print geyscale values
    for(int i=0; i<img_c2g.rows; i++)
    {
        for(int j=0; j<img_c2g.cols; j++)
        {
            cerr << (int) img_c2g.at<uchar>(i,j) << " ";
        }
        cerr << "\n";
    }
    waitKey(0);

    /// Canvas with figures
    Mat canvas = Mat::zeros(512, 512, CV_8UC3);
    canvas = Scalar(255, 255, 255);
    rectangle(canvas, Point(100, 200), Point(400, 300), Scalar(0, 0, 255), 10);
    circle(canvas, Point(256, 256), 128, Scalar(255, 0, 0), 5);
    line(canvas, Point(45, 75), Point(80, 400), Scalar(0, 255, 0), 15);
    imshow("Canvas", canvas);
    waitKey(0);
}
