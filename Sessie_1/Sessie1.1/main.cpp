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
        "{ image_color ic  || (required) path to image }"
        "{ image_bimodal ib || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_color=imageColor.png --image_bimodal=imageBimodal.png";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_color(parser.get<string>("image_color"));
    string imagepath_bimodal(parser.get<string>("image_bimodal"));
    if (imagepath_color.empty() || imagepath_bimodal.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read an show color image
    Mat img_ic;
    img_ic = imread(imagepath_color);
    imshow("EAVISE logo - greyscale", img_ic);
    waitKey(0);

    /// Read an show bimodal image
    Mat img_ib;
    img_ib = imread(imagepath_bimodal);
    imshow("EAVISE logo - color", img_ib);
    waitKey(0);

    /// Segment skin color
    Mat bgr[3];
    Mat BLUE;
    Mat GREEN;
    Mat RED;

    split(img_ic, bgr);
    BLUE = bgr[0];
    GREEN = bgr[1];
    RED = bgr[2];

    Mat mask_1;
    Mat mask_2;
    mask_1 = Mat::zeros(img_ic.rows, img_ic.cols, CV_8UC1);
    mask_2 = mask_1.clone();

    /// With loops
    for (int row=0; row<img_ic.rows; row++)
    {
        for (int col=0; col<img_ic.cols; col++)
        {
            if((RED.at<uchar>(row,col)>95) && (GREEN.at<uchar>(row,col)>40)
            && (BLUE.at<uchar>(row,col)>20)
            && ((max(RED.at<uchar>(row,col),max(GREEN.at<uchar>(row,col),BLUE.at<uchar>(row,col)))
            - min(RED.at<uchar>(row,col),min(GREEN.at<uchar>(row,col),BLUE.at<uchar>(row,col))))>15)
            && (abs(RED.at<uchar>(row,col)-GREEN.at<uchar>(row,col))>15)
            && (RED.at<uchar>(row,col)>GREEN.at<uchar>(row,col)) && (RED.at<uchar>(row,col)>BLUE.at<uchar>(row,col)))
            {
                mask_1.at<uchar>(row,col) = 255;
            }
        }
    }

    /// With matrix operations
    mask_2 = (RED>95) & (GREEN>40) & (BLUE>20) & ((max(RED,max(GREEN,BLUE))
    - min(RED,min(GREEN,BLUE)))>15) & (abs(RED-GREEN)>15) & (RED>GREEN) & (RED>BLUE);

    imshow("mask", mask_2);
    waitKey(0);

    /// Show original image with mask
    Mat bgr_mask(img_ic.rows, img_ic.cols, CV_8UC3);
    Mat pixels_blue = bgr[0] & mask_2;
    Mat pixels_green = bgr[1] & mask_2;
    Mat pixels_red = bgr[2] & mask_2;

    Mat in[] = { pixels_blue, pixels_green, pixels_red };
    int from_to[] = { 0,0, 1,1, 2,2 };

    mixChannels(in, 3, &bgr_mask, 1, from_to, 3);

    /*
    of met merge
    */

    imshow("original + mask", bgr_mask);
    waitKey(0);

    /// OTSU thresholding
    Mat gray_ticket;
    cvtColor(img_ib, gray_ticket, COLOR_RGB2GRAY);
    imshow("ticket grayscale", gray_ticket);
    waitKey(0);

    Mat otsu;
    threshold(gray_ticket, otsu, 0, 255, THRESH_OTSU | THRESH_BINARY);
    imshow("OTSU", otsu);
    waitKey(0);

    /// Histogram equalization
    Mat gray_ticket_equalized;
    equalizeHist(gray_ticket, gray_ticket_equalized);
    imshow("equalized ticket", gray_ticket_equalized);
    waitKey(0);

    /// OTSU with histogram equalization
    Mat otsu_equalized;
    threshold(gray_ticket_equalized, otsu_equalized, 0, 255, THRESH_OTSU | THRESH_BINARY);
    imshow("OTSU equalized", otsu_equalized);
    waitKey(0);

    /// CLAHE
    Mat gray_ticket_CLAHE;
    Ptr<CLAHE> clahe_pointer = createCLAHE();
    clahe_pointer->setTilesGridSize(Size(15,15));
    clahe_pointer->setClipLimit(1);
    clahe_pointer->apply(gray_ticket, gray_ticket_CLAHE);
    imshow("CLAHE ticket grayscale", gray_ticket_CLAHE);
    waitKey(0);

    /// OTSU thresholfing with CLAHE
    Mat otsu_clahe;
    threshold(gray_ticket_CLAHE, otsu_clahe, 0, 255, THRESH_OTSU | THRESH_BINARY);
    imshow("OTSU clahe", otsu_clahe);
    waitKey(0);
}
