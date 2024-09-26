#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// Global variables
cv::CascadeClassifier face_cascade;

int main(int argc, char** argv)
{
    // Initialize ROS node
    ros::init(argc, argv, "face_detection_node");
    ros::NodeHandle nh;

    // Load the face detection model (Haar Cascade)
    std::string face_cascade_name = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    if (!face_cascade.load(face_cascade_name))
    {
        ROS_ERROR("--(!)Error loading face cascade model.");
        return -1;
    }

    // Open laptop's integrated camera (device 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        ROS_ERROR("--(!)Error opening video capture.");
        return -1;
    }

    // Create a window to display the images
    cv::namedWindow("Face Detection");

    // Main loop to process video frames
    while (ros::ok())
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            ROS_ERROR("Failed to capture an image from the camera.");
            break;
        }

        // Convert the image to grayscale for face detection
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_frame, gray_frame);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.1, 3, 0, cv::Size(30, 30));

        // Draw rectangles around the faces
        for (size_t i = 0; i < faces.size(); i++)
        {
            cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
        }

        // Display the result
        cv::imshow("Face Detection", frame);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
    }

    // Release the camera and destroy the window
    cap.release();
    cv::destroyWindow("Face Detection");

    return 0;
}
