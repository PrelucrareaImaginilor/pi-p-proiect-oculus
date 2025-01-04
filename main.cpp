///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#include <sl/Camera.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>


//cv::Mat slMat2cvMat(Mat& input);

using namespace std;
using namespace sl;
cv::Mat slMat2cvMat(Mat& input);
#ifdef HAVE_CUDA
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input);
#endif // HAVE_CUDA

int main(int argc, char **argv) {
    //QApplication a(argc, argv);

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

    // Capture 50 images and depth, then stop
    int i = 0;
    sl::Mat image, depth, point_cloud;

    while (i < 100) {
        // A new image is available if grab() returns ERROR_CODE::SUCCESS
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(image, VIEW::LEFT);
            // Retrieve depth map. Depth is aligned on the left image
            zed.retrieveMeasure(depth, MEASURE::DEPTH);
            // Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA);

            // Get and print distance value in mm at the center of the image
            // We measure the distance camera - object using Euclidean distance
            int x = image.getWidth() / 2;
            int y = image.getHeight() / 2;
            sl::float4 point_cloud_value;
            point_cloud.getValue(x, y, &point_cloud_value);

            if(std::isfinite(point_cloud_value.z)){
                float distance = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);
                cout<<"Distance to Camera at {"<<x<<";"<<y<<"}: "<<distance<<"mm"<<endl;
            }else
                cout<<"The Distance can not be computed at {"<<x<<";"<<y<<"}"<<endl;

                
                

                
                // Convertire sl::Mat la cv::Mat (float32)
                sl::MAT_TYPE data_type = depth.getDataType();
                cv::Mat cv_depth(depth.getHeight(), depth.getWidth(), CV_32FC1, depth.getPtr<sl::float1>(MEM::CPU));
                cv::Mat image_ocv = slMat2cvMat(depth);
                
                // Salvare ca EXR
                vector<int> compression_params;
                compression_params.push_back(cv::ImwriteEXRTypeFlags::IMWRITE_EXR_TYPE_FLOAT);
                compression_params.push_back(cv::ImwriteEXRCompressionFlags::IMWRITE_EXR_COMPRESSION_NO);
                //depth.write(depth_map.exr)
               // cv::imshow("ferestrea", cv_depth);
                //sl::imshow("fereatra", depth);
               // cv::imwrite("depth_map.exr", image_ocv);
                float depth_value = 0;
                
                std::string baza_nume = "imagineadanc";
                std::string extensie = ".png";
                std::string nume_fisier = baza_nume + std::to_string(i) + extensie;
                const char* normal_string = nume_fisier.c_str();
                depth.write(normal_string);
                cv::waitKey(0);
                
                cout << "Harta de adâncime salvată ca depth_map.exr" << endl;
                image_ocv.release();
                

            // Increment the loop
            i++;
        }
    }
    // Close the camera
    zed.close();
    return EXIT_SUCCESS;
}
// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
    case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
    case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
    case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
    case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
    case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
    case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
    case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
    case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
    default: break;
    }
    return cv_type;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

#ifdef HAVE_CUDA
/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}
#endif

/**
* This function displays help in console
**/
void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}
