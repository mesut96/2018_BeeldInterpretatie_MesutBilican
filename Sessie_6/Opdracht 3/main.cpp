///`pkg-config opencv --libs` in build options, linker

#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

/// Function Headers
void DetectPeople(Mat frame);

/// Globals
CascadeClassifier haarcascade;
CascadeClassifier lbpcascade;
string haarcascade_name = "haarcascade_frontalface_alt.xml";
string lbpcascade_name =  "lbpcascade_frontalface_improved.xml";


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
    string videopath_video(parser.get<string>("video_people"));
    if (videopath_video.empty())
    {
        cerr << "video not found\n";
        parser.printMessage();
        return -1;
    }

    if(!haarcascade.load(haarcascade_name))
    {
        printf("--(!)Error loading haar cascade\n");
        return -1;
    }
    if(!lbpcascade.load(lbpcascade_name))
    {
        printf("--(!)Error loading lbp cascade\n");
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

        DetectFaces(frame.clone());

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
    std::vector<Rect> faces_haar;
    std::vector<Rect> faces_lbp;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    vector<int> haarScore;
    vector<int> lpbScore;

    //-- Detect faces
    haarcascade.detectMultiScale( frame_gray.clone(), faces_haar, haarScore, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    lbpcascade.detectMultiScale( frame_gray.clone(), faces_lbp, lpbScore, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for( size_t i = 0; i < faces_haar.size(); i++ )
    {
        Point center( faces_haar[i].x + faces_haar[i].width/2, faces_haar[i].y + faces_haar[i].height/2 );
        ellipse( frame, center, Size( faces_haar[i].width/2, faces_haar[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        putText(frame, to_string(haarScore[i]), Point(center.x+40, center.y+40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,255));
        Mat faceROI = frame_gray( faces_haar[i] );
    }
    for( size_t i = 0; i < faces_lbp.size(); i++ )
    {
        Point center( faces_lbp[i].x + faces_lbp[i].width/2, faces_lbp[i].y + faces_lbp[i].height/2 );
        ellipse( frame, center, Size( faces_lbp[i].width/2, faces_lbp[i].height/2), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
        putText(frame, to_string(lpbScore[i]), Point(center.x-30, center.y-30),FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255));
        Mat faceROI = frame_gray( faces_lbp[i] );
    }
    //-- Show what you got
    imshow( "face detection", frame );
}
