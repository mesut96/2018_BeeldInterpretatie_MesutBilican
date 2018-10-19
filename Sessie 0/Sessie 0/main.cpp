//`pkg-config opencv --libs` in build options, linker

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
        cerr << "help";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_grey(parser.get<string>("image_grey"));
    string imagepath_color(parser.get<string>("image_color"));
    if (imagepath_grey.empty() || imagepath_color.empty())
    {
        cerr << "no arguments";
        parser.printMessage();
        return -1;
    }

    /// Read an show grey image
    Mat img_ig;
    img_ig = imread(imagepath_grey);
    if (imagepath_grey.empty())
    {
        cerr << "image not found";
        parser.printMessage();
        return -1;
    }
    imshow("EAVISE logo - greyscale", img_ig);
    waitKey(0);

    /// Read an show color image
    Mat img_cl;
    img_cl = imread(imagepath_color);
    if (imagepath_color.empty())
    {
        cerr << "image not found";
        parser.printMessage();
        return -1;
    }
    imshow("EAVISE logo - color", img_cl);
    waitKey(0);

}
