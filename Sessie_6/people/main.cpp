///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/objdetect.hpp"

using namespace std;
using namespace cv;

/// Function Headers
void DetectPeople(Mat frame);

/// Globals
vector<Point> traject;

int main( int argc, char** argv )
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ? || show this message }"
        "{ video_people vf  || (required) path to video }"
    );

    /// If help is entered
    if (parser.has("help"))
    {
        parser.printMessage();
        cerr << "use parameters: --video_people=people.mp4";
        return 0;
    }

    /// Collect data from arguments
    string videopath_people(parser.get<string>("video_people"));
    if (videopath_people.empty())
    {
        cerr << "video not found\n";
        parser.printMessage();
        return -1;
    }

    /// Read and show faces.mp4
    VideoCapture cap(videopath_people); //Create a VideoCapture object and open the input file

    /// Check if camera opened successfully
    if(!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1)
    {
        Mat frame;

        cap >> frame; // Capture frame-by-frame

        if(frame.empty())
            break; // If the frame is empty, break immediately

        //imshow( "Frame", frame ); // Display the resulting frame

        DetectPeople(frame.clone());

        char c=(char)waitKey(25);
        if(c==27)
          break;    // Press  ESC on keyboard to exit
    }

    cap.release(); // When everything done, release the video capture object
    destroyAllWindows(); // Closes all the frames

    return 0;
}

void DetectPeople(Mat frame)
{
    HOGDescriptor hog;

    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);
    vector<Rect> found, found_filtered;
    hog.detectMultiScale(gray, found, 0, Size(8,8), Size(8,8), 1.05, 0.05, true);

    size_t i,j;
    for(i=0; i<found.size(); i++)
    {
        Rect r = found[i];
        for(j=0; j<found.size(); j++)
        {
            if(j!=i && (r & found[j])==r)
            {
                break;
            }
        }
        if(j==found.size())
        {
            found_filtered.push_back(r);
        }
    }

    for(i=0; i<found_filtered.size(); i++)
    {
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.06);
        r.height = cvRound(r.height*0.9);
        rectangle(frame, r.tl(), r.br(), Scalar(0,0,255), 2);
        traject.push_back(Point(r.x, r.y+r.height/2));
    }

    for(size_t t=0; t<traject.size(); t++)
    {
        circle(frame, traject[t], 3, Scalar(255,128,128),-1); // teken traject
    }

    imshow( "face detection", frame );
}
