///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

//! [declare]
/// Global Variables
bool use_mask;
Mat img; Mat templ; Mat mask; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;
//! [declare]

/// Function Headers
void MatchingMethod( int, void* );

//int main(int argc, const char** argv)
//{
//    /// Adding a little help option and command line parser input
//    CommandLineParser parser(argc, argv,
//        "{ help h usage ? || show this message }"
//        "{ image_template it  || (required) path to image }"
//        "{ image_recht ir || (required) path to image }"
//    );
//    /// If help is entered
//    if (parser.has("help"))
//    {
//        parser.printMessage();
//        cerr << "use parameters: --image_template=temp[ate.jpg --image_recht=recht.jpg";
//        return 0;
//    }
//
//    /// Collect data from arguments
//    string imagepath_template(parser.get<string>("image_template"));
//    string imagepath_recht(parser.get<string>("image_recht"));
//    if (imagepath_template.empty() || imagepath_recht.empty())
//    {
//        cerr << "image not found\n";
//        parser.printMessage();
//        return -1;
//    }
//
//    /// Read and show template image
//    Mat img_template;
//    img_template = imread(imagepath_template);
//    imshow("Template", img_template);
//    waitKey(0);
//
//    /// Read and show aligned image
//    Mat img_recht;
//    img_recht = imread(imagepath_recht);
//    imshow("EAVISE logo - color", img_recht);
//    waitKey(0);

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ image_template it  || (required) path to image }"
        "{ image_recht ir || (required) path to image }"
    );
    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --image_template=temp[ate.jpg --image_recht=recht.jpg";
        return 0;
    }

    /// Collect data from arguments
    string imagepath_template(parser.get<string>("image_template"));
    string imagepath_recht(parser.get<string>("image_recht"));
    if (imagepath_template.empty() || imagepath_recht.empty())
    {
        cerr << "image not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show template image
    Mat templ;
    templ = imread(imagepath_template);
    imshow("Template", templ);
    waitKey(0);

    /// Read and show aligned image
    Mat img;
    img = imread(imagepath_recht);
    imshow("EAVISE logo - color", img);
    waitKey(0);

    /// Template matching
    Mat result;

    //int result_cols =  img.cols - templ.cols + 1;
    //int result_rows = img.rows - templ.rows + 1;

    //result.create( result_rows, result_cols, CV_32FC1 );

    matchTemplate( img, templ, result, TM_SQDIFF);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    imshow("EAVISE logo - color", result);
    waitKey(0);

    /// Localizing the best match with minMaxLoc
    Mat img_display;
    img.copyTo( img_display );

    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    matchLoc = minLoc;

    rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,255,0), 2, 8, 0 );
    rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,255,0), 2, 8, 0 );

    imshow( image_window, img_display );
    waitKey(0);
    imshow( result_window, result );
    waitKey(0);

    ///meerdere objecten
    Mat mask, temp2;
    Mat img_display2;
    img.copyTo( img_display2 );

    threshold(result, mask, 0.9, 1, THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    mask = mask*255;
    imshow( "mask ", mask );
    waitKey(0);

    vector<vector<Point>>  contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    for (int i=0; i <contours.size(); i++)
    {
        Rect region = boundingRect(contours[i]);
        Mat temp = result(region);
        Point maxLock;
        Point minLock;
        minMaxLoc(temp, NULL, NULL, &minLoc, &maxLoc, Mat() );
        rectangle(img_display2, Point(region.x+ minLoc.x, region.y + minLoc.y),
         Point(minLoc.x +region.x + templ.cols, minLoc.y +region.y + templ.rows), Scalar(0,0,255));
     }

    imshow( "multiple ", img_display2 );
    waitKey(0);




//  if (argc < 3)
//  {
//    cout << "Not enough parameters" << endl;
//    cout << "Usage:\n./MatchTemplate_Demo <image_name> <template_name> [<mask_name>]" << endl;
//    return -1;
//  }
//
//  //! [load_image]
//  /// Load image and template
//  img = imread( argv[1], IMREAD_COLOR );
//  templ = imread( argv[2], IMREAD_COLOR );
//
//  if(argc > 3) {
//    use_mask = true;
//    mask = imread( argv[3], IMREAD_COLOR );
//  }
//
//  if(img.empty() || templ.empty() || (use_mask && mask.empty()))
//  {
//    cout << "Can't read one of the images" << endl;
//    return -1;
//  }
//  //! [load_image]
//
//  //! [create_windows]
//  /// Create windows
//  namedWindow( image_window, WINDOW_AUTOSIZE );
//  namedWindow( result_window, WINDOW_AUTOSIZE );
//  //! [create_windows]
//
//  //! [create_trackbar]
//  /// Create Trackbar
//  const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
//  createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );
////! [create_trackbar]
//
//  MatchingMethod( 0, 0 );
//  waitKey(0);
//  return 0;
}

void MatchingMethod( int, void* )
{
  //! [copy_source]
  /// Source image to display
  Mat img_display;
  img.copyTo( img_display );
  //! [copy_source]

  //! [create_result_matrix]
  /// Create the result matrix
  int result_cols =  img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;

  result.create( result_rows, result_cols, CV_32FC1 );
  //! [create_result_matrix]

  //! [match_template]
  /// Do the Matching and Normalize
  bool method_accepts_mask = (TM_SQDIFF == match_method || match_method == TM_CCORR_NORMED);
  if (use_mask && method_accepts_mask)
    { matchTemplate( img, templ, result, match_method, mask); }
  else
    { matchTemplate( img, templ, result, match_method); }
  //! [match_template]

  //! [normalize]
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
  //! [normalize]

  //! [best_match]
  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  //! [best_match]

  //! [match_loc]
  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == TM_SQDIFF || match_method == TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }
  //! [match_loc]

  //! [imshow]
  /// Show me what you got
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
  rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

  imshow( image_window, img_display );
  imshow( result_window, result );
  //! [imshow]

  return;
}
