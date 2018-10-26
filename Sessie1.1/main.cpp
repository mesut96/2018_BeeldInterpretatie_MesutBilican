#include <iostream>

using namespace std;

int main()
{
    cout << "Hello world!" << endl;
    return 0;

     mask = (RED>95) & (GREEN>40) & (BLUE>20) & ((max(RED,max(GREEN,BLUE))
    - min(RED,min(GREEN,BLUE)))>15) & (abs(RED-GREEN)>15) & (RED>GREEN) & (RED>BLUE);

    mask = mask*255;

    erode(mask, mask, Mat(), Point(-1,-1), 2);
    dilate(mask, mask, Mat(), Point(-1,-1, 2));

    /// Convex hull approach
    vector< vector<Point> > contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector< vector<Point> > hulls;
    for (size_t i=0; i<contours.size(); i++)
    {
        vector<Point> hull;
        convexHull(contours[i], hull);
        hulls.push_back(hull)
    }
    // draw the filled hulls
    drawContours(mask, hulls, -1, 255, -1)
}
