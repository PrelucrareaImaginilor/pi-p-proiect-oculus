#include <sl/Camera.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
#ifdef HAVE_CUDA
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input);
#endif

int main(int argc, char** argv) {
    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA; // Use ULTRA depth mode
    init_parameters.coordinate_units = UNIT::MILLIMETER; // Use millimeter units (for depth measurements)

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program." << endl;
        return EXIT_FAILURE;
    }

    // Create OpenCV window
    const string window_name = "Normalized Depth View";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    // Variables for images and loop control
    sl::Mat image, depth;
    char key = ' ';

    while (key != 'q') { // press 'q' to exit the loop
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            zed.retrieveImage(image, VIEW::LEFT);
            zed.retrieveMeasure(depth, MEASURE::DEPTH);
            cv::Mat depth_ocv = slMat2cvMat(depth);

            //  normalize depth
            cv::Mat depth_normalized;
            double min_val, max_val;
            cv::minMaxLoc(depth_ocv, &min_val, &max_val);
            depth_ocv.convertTo(depth_normalized, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val)); //convertesc pentru a putea 

            //color map
            cv::Mat depth_colored;
            cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET);
            cv::imshow(window_name, depth_colored);

            // Check for key presses
            key = cv::waitKey(1);
            if (key == 's') { // Press 's' to save 100 depth maps
                for (int i = 0; i < 100; i++) {
                    if (zed.grab() == ERROR_CODE::SUCCESS) {

                        zed.retrieveMeasure(depth, MEASURE::DEPTH);

                        //save depth
                        string file_name = "raw_depth_frame_" + to_string(i) + ".png";
                        depth.write(file_name.c_str());
                        cout << "Saved: " << file_name << endl;
                    }
                }
            }
        }
    }

    // close the camera and destroy the window
    zed.close();
    cv::destroyWindow(window_name);
    return EXIT_SUCCESS;
}

cv::Mat slMat2cvMat(Mat& input) {
    return cv::Mat(input.getHeight(), input.getWidth(), CV_32FC1, input.getPtr<sl::float1>(MEM::CPU));
}

#ifdef HAVE_CUDA
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input) {
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), CV_32FC1, input.getPtr<sl::float1>(MEM::GPU));
}
#endif
