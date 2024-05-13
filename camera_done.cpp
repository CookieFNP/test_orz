#include <opencv2/opencv.hpp>  
#include <vector>  
#include <string>  

using namespace std;
using namespace cv;

int main() {
    int camera_id = 1;
    Size boardSize(11, 8);
    float squareSize = 15.0f;
    vector<vector<Point3f>> objPoints;
    vector<vector<Point2f>> imgPoints;

    VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Point3f> objp;
    for (int y = 0; y < boardSize.height; y++) {
        for (int x = 0; x < boardSize.width; x++) {
            objp.push_back(Point3f(x * squareSize, y * squareSize, 0));
        }
    }

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    bool needCalibration = true;
    int requiredFrames = 10; 

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners);

        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

            objPoints.push_back(objp);
            imgPoints.push_back(corners);
            if (objPoints.size() >= requiredFrames && needCalibration) {
                double rms = calibrateCamera(objPoints, imgPoints, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
                cout << "RMS: " << rms << endl;
                needCalibration = false;
            }

            drawChessboardCorners(frame, boardSize, corners, found);
        }

        if (!distCoeffs.empty()) {
            Mat undistortedFrame;
            undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);
            imshow("Undistorted Camera Feed", undistortedFrame);
        }
        else {
            imshow("Camera Feed", frame);
        }

        int key = waitKey(1);
        if (key == 'q' || key == 27) { 
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}