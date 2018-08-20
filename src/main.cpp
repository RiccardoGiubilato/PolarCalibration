#include <iostream>
#include <stdio.h>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "polarcalibration.h"

using namespace std;

class CalibParser {
public:
  CalibParser(std::string& filename) {
    file_ = cv::FileStorage(filename, cv::FileStorage::READ);
    file_["K"] >> K;
    file_["d"] >> d;
  };
  CalibParser() { };
  bool open(const std::string& filename) {
    file_ = cv::FileStorage(filename, cv::FileStorage::READ);
    if (file_.isOpened()) {
      file_["camera_matrix"] >> K;
      file_["distortion_coefficients"] >> d;
      std::cout << K << std::endl;
      std::cout << d << std::endl;
      return true;
    } else {
      return false;
    }
  };
  cv::Mat getCalib() {return K;};
  cv::Mat getDist() {return d;};

private:
  cv::FileStorage file_;
  cv::Mat K, d;
};

struct StereoParams {
  int minDisparity;
  int numDisparities;
  int SADWindowSize;
  int P1;
  int P2;
  int disp12MaxDiff;
  int preFilterCap;
  int uniquenessRatio;
  int speckleWindowSize;
  int speckleRange;
  bool fullDP;
};

class StereoMatcher {
public:
  StereoMatcher(std::string& filename) {
    file_ = cv::FileStorage(filename, cv::FileStorage::READ);
    if (file_.isOpened()) {
      /* Store stereo params */
      file_["minDisparity"] >> params_.minDisparity;
      file_["numDisparities"] >> params_.numDisparities;
      file_["SADWindowSize"] >> params_.SADWindowSize;
      file_["P1"] >> params_.P1;
      file_["P2"] >> params_.P2;
      file_["disp12MaxDiff"] >> params_.disp12MaxDiff;
      file_["preFilterCap"] >> params_.preFilterCap;
      file_["uniquenessRatio"] >> params_.uniquenessRatio;
      file_["speckleWindowSize"] >> params_.speckleWindowSize;
      file_["speckleRange"] >> params_.speckleRange;
      file_["fullDP"] >> params_.fullDP;

      matcher_ = cv::StereoSGBM(params_.minDisparity, params_.numDisparities, params_.SADWindowSize,
      params_.P1, params_.P2, params_.disp12MaxDiff, params_.preFilterCap, params_.uniquenessRatio,
      params_.speckleWindowSize, params_.speckleRange, params_.fullDP);
    }
  }

  bool computeDisparity(cv::Mat& im1, cv::Mat& im2) {
    matcher_(im1, im2, disparity_);
    std::cout << "generated disparity of type: " << disparity_.type() << std::endl;
    disparity_raw_ = cv::Mat(disparity_.rows, disparity_.cols, CV_16S);
    disparity_.copyTo(disparity_raw_);
    disparity_.convertTo(disparity_vis_, CV_8U, 255/(params_.numDisparities*16.));
  };
  cv::Mat getDisparity() {
    return disparity_vis_;};
  cv::Mat getDisparityRaw() {return disparity_raw_;};
  void showDisparity() {
    cv::namedWindow("disparity");
    cv::imshow("disparity", disparity_);
    char k = cv::waitKey(1);

  }

private:
  cv::FileStorage file_;
  cv::StereoSGBM matcher_;
  cv::Mat disparity_, disparity_vis_, disparity_raw_;
  StereoParams params_;
};

int main(int argc, char * argv[]) {

    cv::Mat showImg1, showImg2;

    cv::namedWindow("showImg1");
    cv::namedWindow("showImg2");

    PolarCalibration calibrator;
    calibrator.toggleShowCommonRegion(false);
    calibrator.toggleShowIterations(false);

    std::string stereo_params_filename(argv[4]);
    StereoMatcher matcher(stereo_params_filename);

    cv::Mat img1distorted = cv::imread(argv[1], 0);
    cv::Mat img2distorted = cv::imread(argv[2], 0);

    CalibParser parser;
    parser.open(std::string(argv[3]));

    cv::Mat cameraMatrix1  = parser.getCalib();
    cv::Mat cameraMatrix2 = cameraMatrix1.clone();
    cv::Mat distCoeffs1 = parser.getDist();
    cv::Mat distCoeffs2 = distCoeffs1.clone();

    if (!calibrator.compute(img1distorted, img2distorted, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2)) {
      std::cout << "calibrator failed to compute" << std::endl;
      return 0;
    }

    // Visualization
    cv::Mat img1, img2, rectified1, rectified2;
    cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
    cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
    calibrator.getRectifiedImages(img1, img2, rectified1, rectified2);

    cv::Mat img_orig1, img_orig2;
    calibrator.getOriginalImages(rectified1, rectified2, img_orig1, img_orig2);
    std::cout << "size original: " << img_orig1.size() << std::endl;
    std::cout << "size original: " << img_orig2.size() << std::endl;
    imshow("orig_1", img_orig1);
    imshow("orig_2", img_orig2);

    cv::Mat scaled1, scaled2;
    cv::Size newSize;
    if (rectified1.cols > rectified1.rows) {
        newSize = cv::Size(600, 600 * rectified1.rows / rectified1.cols);
    } else {
        newSize = cv::Size(600 * rectified1.cols / rectified1.rows, 600);
    }

    cout << "prevSize " << rectified1.size() << endl;
    cout << "newSize " << newSize << endl;

    cv::resize(rectified1, scaled1, newSize);
    cv::resize(rectified2, scaled2, newSize);

    matcher.computeDisparity(rectified1, rectified2);
    cv::Mat disp = matcher.getDisparityRaw();
    cv::Mat disp_view = matcher.getDisparity();
    //std::cout << disp << std::endl;
    cv::Mat disp_scaled;
    cv::resize(disp_view, disp_scaled, newSize);

    cv::imshow("showImg1", scaled1);
    cv::imshow("showImg2", scaled2);
    cv::imshow("showDisp", disp_view);
    cv::moveWindow("showImg2", 700, 0);

    calibrator.compute3DMap(disp);

    uint8_t keycode = cv::waitKey(0);
    switch (keycode) {
        case 'q':
            exit(0);
            break;
        default:
            ;
    }

  return 0;
}
