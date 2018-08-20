/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <boost/concept_check.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <omp.h>

#include "polarcalibration.h"


PolarCalibration::PolarCalibration() {
    m_hessianThresh = 20;
    m_stepSize = STEP_SIZE;
    m_showCommonRegion = true;
    m_showIterations = false;
}

PolarCalibration::~PolarCalibration() {

}

bool PolarCalibration::compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted,
                               const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1,
                               const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2,
                               const uint32_t method) {

    cv::Mat img1(img1distorted.rows, img1distorted.cols, CV_8UC1);
    cv::Mat img2(img2distorted.rows, img2distorted.cols, CV_8UC1);

    cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
    cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);

    Kmat = cameraMatrix1;

    return compute(img1, img2);
}


bool PolarCalibration::compute(const cv::Mat& img1, const cv::Mat& img2, cv::Mat F,
                               vector< cv::Point2f > points1, vector< cv::Point2f > points2, const uint32_t method)
{
//     cv::Mat F;
    cv::Point2d epipole1, epipole2;
    if (! findFundamentalMat(img1, img2, F, points1, points2, epipole1, epipole2, method))
        return false;

    Fund = F;

    // Determine common region
    vector<cv::Vec3f> initialEpilines, finalEpilines;
    vector<cv::Point2f> epipoles(2);
    epipoles[0] = epipole1;
    epipoles[1] = epipole2;
    determineCommonRegion(epipoles, cv::Size(img1.cols, img1.rows), F);

    getTransformationPoints(img1.size(), epipole1, epipole2, F);
    doTransformation(img1, img2, epipole1, epipole2, F);

    return true;
}


inline bool PolarCalibration::findFundamentalMat(const cv::Mat& img1, const cv::Mat& img2, cv::Mat & F,
                                                 vector<cv::Point2f> points1, vector<cv::Point2f> points2,
                                                 cv::Point2d& epipole1, cv::Point2d& epipole2, const uint32_t method)
{
    if (F.empty()) {
        switch(method) {
            case FMAT_METHOD_OFLOW:
                findPairsOFlow(img1, img2, points1, points2);
                break;
            case FMAT_METHOD_SURF:
                findPairsSURF(img1, img2, points1, points2);
                break;
        }

        if (points1.size() < 8)
            return false;

        F = cv::findFundamentalMat(points1, points2, CV_FM_RANSAC, 1, 0.999);

        /* OPTIMIZE THE TWO VIEW GEOMETRY! */

        if (cv::countNonZero(F) == 0) {
            std::cout << "Failed to compute fundamental matrix..!" << std::endl;
            return false;
        };
    }

    // We obtain the epipoles
    getEpipoles(F, epipole1, epipole2);

    checkF(F, epipole1, epipole2, points1[0], points2[0]);

    /// NOTE: Remove. Just for debugging (begin)
//     {
//         cout << "***********************" << endl;
//         cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat_I.xml", cv::FileStorage::READ);
// //         cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/results/mats/lastMat_0925.xml", cv::FileStorage::READ);
// //         cv::FileStorage file("/tmp/results/mats/lastMat_0039.xml", cv::FileStorage::READ);
//
//         file["F"] >> F;
//         file.release();
//         cout << "F (from file)\n" << F << endl;
//         getEpipoles(F, epipole1, epipole2);
//         cout << "epipole1 " << epipole1 << endl;
//         cout << "epipole2 " << epipole2 << endl;
//     }
    /// NOTE: Remove. Just for debugging (end)

    /// NOTE: Remove. Just for debugging (begin)
//     cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat_O.xml", cv::FileStorage::WRITE);
//     file << "F" << F;
//     file.release();
    /// NOTE: Remove. Just for debugging (end)

    if (SIGN(epipole1.x) != SIGN(epipole2.x) &&
        SIGN(epipole1.y) != SIGN(epipole2.y)) {

        epipole2 *= -1;
    }

    return true;
}

void PolarCalibration::optimizeTwoView(cv::Mat& R, cv::Mat& t, std::vector<cv::Point3f>& p,
                      const std::vector<cv::Point2f>& f1, const std::vector<cv::Point2f>& f2,
                      const cv::Mat& K, const cv::Mat& d) {

	// check if size of f1 and size of f2 are equal, else throw warning
	if( !( (f1.size() == f2.size() ) && (f1.size() == p.size() ) ) ) std::cout << "WARNING: size point != size features" << std::endl;

    // generate arrays of mutable data that will be passed to cost functors
	double* camera1   = new double[6];
	double* camera2   = new double[6];
	double* intrinsic = new double[9];
	double* points    = new double[3*p.size()];
  double* old_intrinsic = new double[9];

    // camera 1
	for(size_t i=0; i<6; i++){
		camera1[i] = 0;
	}

	// camera 2
	cv::Vec3f angles = rotationMatrixToEulerAngles(R);
	for(size_t i=0; i<3; i++){
		camera2[i] = angles[i];
	}
	for(size_t i=0; i<3; i++){
		camera2[i+3] = t.at<double>(i,0);
	}

	// intrinsics
	intrinsic[0] = K.at<double>(0,0);
	intrinsic[1] = K.at<double>(1,1);
	intrinsic[2] = K.at<double>(0,2);
	intrinsic[3] = K.at<double>(1,2);

	intrinsic[4] = 0.0; // undistorted points
	intrinsic[5] = 0.0;
	intrinsic[6] = 0.0;
	intrinsic[7] = 0.0;
	intrinsic[8] = 0.0;

  for(size_t i=0; i<9; i++) old_intrinsic[i] = intrinsic[i];

	// points
	for(size_t i=0; i<p.size(); i++){
		points[3*i + 0] = p[i].x;
		points[3*i + 1] = p[i].y;
		points[3*i + 2] = p[i].z;
	}

	// optimization
	ceres::Problem TwoViewOptimizer;

		// Add a cost function for each observations of camera 1
		for(size_t i=0; i<f1.size(); i++){
			ceres::CostFunction* cost_function =
				ReprojectionError::Create(
  									 f1[i].x,
                                     f1[i].y);
				TwoViewOptimizer.AddResidualBlock(cost_function,
						NULL,
						camera1,
						intrinsic,
						points+(3*i)   );
				TwoViewOptimizer.SetParameterBlockConstant( intrinsic );
				TwoViewOptimizer.SetParameterBlockConstant( camera1 );
		}

		// Add a cost function for each observations of camera 2
		for(size_t i=0; i<f2.size(); i++){
			ceres::CostFunction* cost_function =
				ReprojectionError::Create(
  									 f2[i].x,
                                     f2[i].y);
				TwoViewOptimizer.AddResidualBlock(cost_function,
						NULL,
						camera2,
						intrinsic,
						points+(3*i)   );
				TwoViewOptimizer.SetParameterBlockConstant( intrinsic );
		}

	ceres::Solver::Options options;
  	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  	options.minimizer_progress_to_stdout = true;
  	options.max_num_iterations = 20;

  	ceres::Solver::Summary summary;
  	ceres::Solve(options, &TwoViewOptimizer, &summary);
  	std::cout << summary.FullReport() << "\n";

  	// write optimized points in std::vector<cv::Points3f>
	for(size_t i=0; i<p.size(); i++){
		p[i].x = points[3*i + 0];
		p[i].y = points[3*i + 1];
		p[i].z = points[3*i + 2];
	}

	// write optimized matrices in R, t
	cv::Vec3f angleVector(camera2[0],camera2[1],camera2[2]);
	R = eulerAnglesToRotationMatrix( angleVector );
	t.at<double>(0,0) = camera2[3];
	t.at<double>(1,0) = camera2[4];
	t.at<double>(2,0) = camera2[5];

  // deallocate
  delete[] camera1;
  delete[] camera2;
  delete[] intrinsic;
  delete[] points;
  delete[] old_intrinsic;

};



PolarCalibration::ReprojectionError::ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool PolarCalibration::ReprojectionError::operator()(
  				  const T* const camera,
  				  const T* const intrinsics,
            const T* const point,
            T* residuals) const {

  	T rMat[9];
	  T eul[3] = {180/3.14*camera[0],
    	    	    180/3.14*camera[1],
    	    	    180/3.14*camera[2]};
    ceres::EulerAnglesToRotationMatrix(eul,3,rMat);

    // [R|r] world -> camera
    Eigen::Matrix<T,3,3> R;
    Eigen::Matrix<T,3,1> t;
    R << rMat[0], rMat[1], rMat[2],
         rMat[3], rMat[4], rMat[5],
         rMat[6], rMat[7], rMat[8];
    t << camera[3], camera[4], camera[5];

	// point -> Eigen
    Eigen::Matrix<T,3,1> point_eig;
    point_eig << point[0], point[1], point[2];

    // transform point to local
    Eigen::Matrix<T,3,3> Rl = R.transpose();
    Eigen::Matrix<T,3,1> tl = -Rl*t;
    Eigen::Matrix<T,3,1> p  = Rl*point_eig + tl;

    // Normalize the point coordinates
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Apply distortion coefficients. Distortion model found in:
	// http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
	// coefficients
	const T& k1 = intrinsics[4];
	const T& k2 = intrinsics[5];
	const T& p1 = intrinsics[6];
	const T& p2 = intrinsics[7];
	const T& k3 = intrinsics[8];

	T r = xp*xp + yp*yp; // r = r^2 !
	T xp2 = xp * (T(1) + k1*r + k2*r*r + k3*r*r*r) + T(2)*p1*xp*yp + p2*(r + T(2)*xp*xp);
	T yp2 = yp * (T(1) + k1*r + k2*r*r + k3*r*r*r) + T(2)*p2*xp*yp + p1*(r + T(2)*yp*yp);

    // Compute final projected point position.
    const T& fSx = intrinsics[0];
    const T& fSy = intrinsics[1];
    const T& cx  = intrinsics[2];
    const T& cy  = intrinsics[3];
    T predicted_x = fSx * xp2 + cx;
    T predicted_y = fSy * yp2 + cy;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  ceres::CostFunction* PolarCalibration::ReprojectionError::Create(
  									 const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 9, 3>(
                new ReprojectionError(observed_x, observed_y)));
  }


cv::Vec3f PolarCalibration::rotationMatrixToEulerAngles(cv::Mat &R)
{


    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        z = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        x = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        z = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        x = 0;
    }
    return cv::Vec3f(x, y, z);

}

cv::Mat PolarCalibration::eulerAnglesToRotationMatrix(cv::Vec3f &theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );

    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,                1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );

    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);


    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;

}

inline void PolarCalibration::findPairsOFlow(const cv::Mat & img1, const cv::Mat & img2,
                                             vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2) {

    // We look for correspondences using Optical flow
    // vector of keypoints
    vector<cv::KeyPoint> keypoints1;
    cv::FastFeatureDetector fastDetector(20);
    fastDetector.detect(img1, keypoints1);

    if (keypoints1.size() == 0)
        return;

    vector<cv::Point2f> points1(keypoints1.size()), points2, points1B;
    {
        uint32_t idx = 0;
        for (vector<cv::KeyPoint>::iterator it = keypoints1.begin(); it != keypoints1.end(); it++, idx++) {
            points1[idx] = it->pt;
        }
    }
    // Optical flow
    vector<uint8_t> status, statusB;
    vector<float_t> error, errorB;

    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, cv::Size(15, 15), 3);
    cv::calcOpticalFlowPyrLK(img2, img1, points2, points1B, statusB, errorB, cv::Size(15, 15), 3);

    vector<cv::Point2f> pointsA(points1.size()), pointsB(points2.size());
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < points1.size(); i++) {
            if ((status[i] == 1) && (statusB[i] == 1)) {
                if (cv::norm(points1[i] - points1B[i]) < 1.0) {
                    pointsA[idx] = points1[i];
                    pointsB[idx] = points2[i];
                }
            }
            idx++;
        }
        pointsA.resize(idx);
        pointsB.resize(idx);
    }

    outPoints1 = pointsA;
    outPoints2 = pointsB;

}

inline void PolarCalibration::findPairsSURF(const cv::Mat & img1, const cv::Mat & img2,
                                                     vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2) {

    // We look for correspondences using SURF

    // vector of keypoints
    vector<cv::KeyPoint> keypoints1, keypoints2;

    cv::SurfFeatureDetector surf(m_hessianThresh);
    surf.detect(img1, keypoints1);
    surf.detect(img2, keypoints2);

    // Descriptors are extracted
    cv::SurfDescriptorExtractor surfDesc;
    cv::Mat descriptors1, descriptors2;
    surfDesc.compute(img1, keypoints1, descriptors1);
    surfDesc.compute(img2, keypoints2, descriptors2);

    // Descriptors are matched
    cv::FlannBasedMatcher matcher;
    vector<cv::DMatch> matches;

    matcher.match(descriptors1, descriptors2, matches);

    nth_element(matches.begin(), matches.begin()+24, matches.end());
    matches.erase(matches.begin()+25, matches.end());

//     cv::Mat imageMatches;
//     cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imageMatches, cv::Scalar(0,0,255));
//     cv::namedWindow("Matched");
//     cv::imshow("Matched", imageMatches);

    // Fundamental matrix is found
    vector<cv::Point2f> points1, points2;

    points1.resize(matches.size());
    points2.resize(matches.size());

    {
        uint32_t idx2 = 0;
        for (int idx = 0; idx < matches.size(); idx++) {
            const cv::Point2f & p1 = keypoints1[matches[idx].queryIdx].pt;
            const cv::Point2f & p2 = keypoints2[matches[idx].trainIdx].pt;

            if (fabs(p1.x - p2.x < 10.0) && fabs(p1.y - p2.y < 10.0)) {
                points1[idx2] = p1;
                points2[idx2] = p2;
                idx2++;
            }
        }
        points1.resize(idx2);
        points2.resize(idx2);
    }

    outPoints1 = points1;
    outPoints2 = points2;
}

inline void PolarCalibration::getEpipoles(const cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2) {
    cv::SVD svd(F);

    cv::Mat e1 = svd.vt.row(2);
    cv::Mat e2 = svd.u.col(2);

    epipole1 = cv::Point2d(e1.at<double>(0, 0) / e1.at<double>(0, 2), e1.at<double>(0, 1) / e1.at<double>(0, 2));
    epipole2 = cv::Point2d(e2.at<double>(0, 0) / e2.at<double>(2, 0), e2.at<double>(1, 0) / e2.at<double>(2, 0));
}

inline void PolarCalibration::checkF(cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2, const cv::Point2d & m, const cv::Point2d & m1) {
    cv::Vec3f line = GET_LINE_FROM_POINTS(epipole1, m);
    vector<cv::Point2f> points(1);
    points[0] = epipole1;
    vector<cv::Vec3f> lines;
    cv::computeCorrespondEpilines(points, 1, F, lines);
    cv::Vec3f line1 = lines[0];

    cv::Mat L(1, 3, CV_64FC1);
    L.at<double>(0, 0) = line[0];
    L.at<double>(0, 1) = line[1];
    L.at<double>(0, 2) = line[2];

    cv::Mat L1(1, 3, CV_64FC1);
    L1.at<double>(0, 0) = line1[0];
    L1.at<double>(0, 1) = line1[1];
    L1.at<double>(0, 2) = line1[2];

    cv::Mat M(3, 1, CV_64FC1);
    M.at<double>(0, 0) = m.x;
    M.at<double>(1, 0) = m.y;
    M.at<double>(2, 0) = 1.0;

    cv::Mat M1(3, 1, CV_64FC1);
    M1.at<double>(0, 0) = m1.x;
    M1.at<double>(1, 0) = m1.y;
    M1.at<double>(2, 0) = 1.0;

    cv::Mat fl = L * M;
    cv::Mat fl1 = L1 * M1;


    if (SIGN(fl.at<double>(0,0)) != SIGN(fl1.at<double>(0,0))) {
        F = -F;

        getEpipoles(F, epipole1, epipole2);
    }
}

inline bool PolarCalibration::lineIntersectsSegment(const cv::Vec3d & line, const cv::Point2d & p1, const cv::Point2d & p2, cv::Point2d * intersection) {
    const cv::Vec3d segment = GET_LINE_FROM_POINTS(p1, p2);

    if (intersection != NULL)
        *intersection = cv::Point2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

    // Lines are represented as ax + by + c = 0, so
    // y = -(ax+c)/b. If y1=y2, then we have to obtain x, which is
    // x = (b1 * c2 - b2 * c1) / (b2 * a1 - b1 * a2)
    if ((segment[1] * line[0] - line[1] * segment[0]) == 0)
        return false;
    double x = (line[1] * segment[2] - segment[1] * line[2]) / (segment[1] * line[0] - line[1] * segment[0]);
    double y = -(line[0] * x + line[2]) / line[1];

    if (((int32_t)round(x) >= (int32_t)min(p1.x, p2.x)) && ((int32_t)round(x) <= (int32_t)max(p1.x, p2.x))) {
        if (((int32_t)round(y) >= (int32_t)min(p1.y, p2.y)) && ((int32_t)round(y) <= (int32_t)max(p1.y, p2.y))) {
            if (intersection != NULL)
                *intersection = cv::Point2d(x, y);

            return true;
        }
    }

    return false;
}

inline bool PolarCalibration::lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions, cv::Point2d * intersection) {
    return lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), intersection) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), intersection) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), intersection) ||
            lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), intersection);
}

inline bool PolarCalibration::isTheRightPoint(const cv::Point2d & epipole, const cv::Point2d & intersection, const cv::Vec3d & line,
                                       const cv::Point2d * lastPoint)
{
    if (lastPoint != NULL) {
        cv::Vec3f v1(lastPoint->x - epipole.x, lastPoint->y - epipole.y, 0.0);
        v1 /= cv::norm(v1);
        cv::Vec3f v2(intersection.x - epipole.x, intersection.y - epipole.y, 0.0);
        v2 /= cv::norm(v2);

        if (fabs(acos(v1.dot(v2))) > CV_PI / 2.0)
            return false;
        else
            return true;
    } else {
        if ((line[0] > 0) && (epipole.y < intersection.y)) return false;
        if ((line[0] < 0) && (epipole.y > intersection.y)) return false;
        if ((line[1] > 0) && (epipole.x > intersection.x)) return false;
        if ((line[1] < 0) && (epipole.x < intersection.x)) return false;

        return true;
    }
    return false;
}

inline
void PolarCalibration::getBorderIntersections(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions,
                                                     vector<cv::Point2d> & intersections) {

    cv::Point2d intersection(-1, -1);
    intersections.reserve(2);

    if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &intersection)) {
        intersections.push_back(intersection);
    }
    if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &intersection)) {
        intersections.push_back(intersection);
    }
    if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &intersection)) {
        intersections.push_back(intersection);
    }
    if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &intersection)) {
        intersections.push_back(intersection);
    }
}

inline
cv::Point2d PolarCalibration::getNearestIntersection(const cv::Point2d& oldEpipole, const cv::Point2d& newEpipole,
                                                    const cv::Vec3d& line, const cv::Point2d& oldPoint,
                                                    const cv::Size & imgDimensions)
{
    vector<cv::Point2d> intersections;
    getBorderIntersections(newEpipole, line, imgDimensions, intersections);

    double minAngle = std::numeric_limits<double>::max();
    cv::Point2d point(-1, -1);

    cv::Vec3d v1(oldPoint.x - oldEpipole.x, oldPoint.y - oldEpipole.y, 0.0);
    v1 /= cv::norm(v1);

    for (uint32_t i = 0; i < intersections.size(); i++) {
        cv::Vec3d v(intersections[i].x - newEpipole.x, intersections[i].y - newEpipole.y, 0.0);
        v /= cv::norm(v);

        const double & angle = fabs(acos(v.dot(v1)));

        if (angle < minAngle) {
            minAngle = angle;
            point = intersections[i];
        }
    }

    return point;
}


inline
cv::Point2d PolarCalibration::getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions,
                                                            const cv::Point2d * lastPoint) {

    cv::Point2d intersection(-1, -1);

    if (IS_INSIDE_IMAGE(epipole, imgDimensions)) {
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
    } else {
        double maxDist = std::numeric_limits<double>::min();
        cv::Point2d tmpIntersection(-1, -1);
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
                                 (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);

            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);

            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);

            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);

            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        return intersection;
    }
}

inline void PolarCalibration::getExternalPoints(const cv::Point2d &epipole, const cv::Size imgDimensions,
                                                vector<cv::Point2f> &externalPoints) {

    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[1] = cv::Point2f(0, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[1] = cv::Point2f(0, 0);
        } else { // Case 3
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(0, 0);
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(0, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            externalPoints.resize(4);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[2] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
            externalPoints[3] = cv::Point2f(0, imgDimensions.height - 1);
        } else { // Case 6
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
        }
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
        } else { // Case 9
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
        }
    }
}

inline void PolarCalibration::computeEpilines(const vector<cv::Point2f> & points, const uint32_t &whichImage,
                                            const cv::Mat & F, const vector <cv::Vec3f> & oldlines, vector <cv::Vec3f> & newLines) {

    cv::computeCorrespondEpilines(points, whichImage, F, newLines);

    for (uint32_t i = 0; i < oldlines.size(); i++) {
        if ((SIGN(oldlines[i][0]) != SIGN(newLines[i][0])) &&
            (SIGN(oldlines[i][1]) != SIGN(newLines[i][1]))) {
            newLines[i] *= -1;
        }
    }
}

inline cv::Point2d PolarCalibration::getBorderIntersectionFromOutside(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions)
{
    vector<cv::Point2d> intersections;
    getBorderIntersections(epipole, line, imgDimensions, intersections);
    double maxDist = std::numeric_limits<double>::min();
    cv::Point2d returnPoint;
    for (uint32_t i = 0; i < intersections.size(); i++) {
        const double dist = cv::norm(epipole - intersections[i]);
        if (dist > maxDist) {
            maxDist = dist;
            returnPoint = intersections[i];
        }
    }

    return returnPoint;
}

/**
 * This function is more easily understandable after reading section 3.4 of
 * ftp://cmp.felk.cvut.cz/pub/cmp/articles/matousek/Sandr-TR-2009-04.pdf
 * */
inline void PolarCalibration::determineCommonRegion(const vector<cv::Point2f> &epipoles,
                                             const cv::Size imgDimensions, const cv::Mat & F) {

    vector<cv::Point2f> externalPoints1, externalPoints2;
    getExternalPoints(epipoles[0], imgDimensions, externalPoints1);
    getExternalPoints(epipoles[1], imgDimensions, externalPoints2);

    determineRhoRange(epipoles[0], imgDimensions, externalPoints1, m_minRho1, m_maxRho1);
    determineRhoRange(epipoles[1], imgDimensions, externalPoints2, m_minRho2, m_maxRho2);

    if (!IS_INSIDE_IMAGE(epipoles[0], imgDimensions) && !IS_INSIDE_IMAGE(epipoles[1], imgDimensions)) {
        // CASE 1: Both outside
        const cv::Vec3f line11 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
        const cv::Vec3f line12 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[1]);

        const cv::Vec3f line23 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[0]);
        const cv::Vec3f line24 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[1]);

        vector <cv::Vec3f> inputLines(2), outputLines;
        inputLines[0] = line23;
        inputLines[1] = line24;
        computeEpilines(externalPoints2, 2, F, inputLines, outputLines);
        const cv::Vec3f line13 = outputLines[0];
        const cv::Vec3f line14 = outputLines[1];

        inputLines[0] = line11;
        inputLines[1] = line12;
        computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
        const cv::Vec3f line21 = outputLines[0];
        const cv::Vec3f line22 = outputLines[1];

        // Beginning and ending lines
        m_line1B = lineIntersectsRect(line13, imgDimensions)? line13 : line11;
        m_line1E = lineIntersectsRect(line14, imgDimensions)? line14 : line12;
        m_line2B = lineIntersectsRect(line21, imgDimensions)? line21 : line23;
        m_line2E = lineIntersectsRect(line22, imgDimensions)? line22 : line24;

        // Beginning and ending lines intersection with the borders
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[0], m_line1B, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[0] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_b1 = intersections[i];
                }
            }
        }
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[1], m_line2B, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[1] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_b2 = intersections[i];
                }
            }
        }
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[0], m_line1E, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[0] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_e1 = intersections[i];
                }
            }
        }
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[1], m_line2E, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[1] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_e1 = intersections[i];
                }
            }
        }

        if (m_showCommonRegion) {
            showCommonRegion(epipoles[0], line11, line12, line13, line14, m_line1B, m_line1E, m_b1, m_e1, imgDimensions,
                            externalPoints1, std::string("leftCommonRegion"));
            showCommonRegion(epipoles[1], line23, line24, line21, line22, m_line2B, m_line2E, m_b2, m_e2, imgDimensions,
                            externalPoints2, std::string("rightCommonRegion"));
        }

    } else if (IS_INSIDE_IMAGE(epipoles[0], imgDimensions) && IS_INSIDE_IMAGE(epipoles[1], imgDimensions)) {
        // CASE 2: Both inside
        m_line1B = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
        m_line1E = m_line1B;

        vector <cv::Vec3f> inputLines(1), outputLines;
        inputLines[0] = m_line1B;
        computeEpilines(externalPoints1, 1, F, inputLines, outputLines);

        m_line2B = outputLines[0];
        m_line2E = outputLines[0];

        m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
        m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);

        m_b2 = m_e2 = getNearestIntersection(epipoles[0], epipoles[1], m_line2B, m_b1, imgDimensions);

        if (m_showCommonRegion) {
            showCommonRegion(epipoles[0], m_line1B, m_line1E, m_line2B, m_line2E, m_line1B, m_line1E, m_b1, m_e1, imgDimensions,
                            externalPoints1, std::string("leftCommonRegion"));
            showCommonRegion(epipoles[1], m_line2B, m_line2E, m_line1B, m_line1E, m_line2B, m_line2E, m_b2, m_e2, imgDimensions,
                            externalPoints2, std::string("rightCommonRegion"));
        }
    } else {
        // CASE 3: One inside and one outside
        if (IS_INSIDE_IMAGE(epipoles[0], imgDimensions)) {
            // CASE 3.1: Only the first epipole is inside

            const cv::Vec3f line23 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[0]);
            const cv::Vec3f line24 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[1]);

            vector <cv::Vec3f> inputLines(2), outputLines;
            inputLines[0] = line23;
            inputLines[1] = line24;
            computeEpilines(externalPoints2, 2, F, inputLines, outputLines);
            const cv::Vec3f & line13 = outputLines[0];
            const cv::Vec3f & line14 = outputLines[1];

            m_line1B = line13;
            m_line1E = line14;
            m_line2B = line23;
            m_line2E = line24;

            m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
            m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);

            m_b1 = getNearestIntersection(epipoles[1], epipoles[0], m_line1B, m_b2, imgDimensions);
            m_e1 = getNearestIntersection(epipoles[1], epipoles[0], m_line1E, m_e2, imgDimensions);

            if (m_showCommonRegion) {
                showCommonRegion(epipoles[0], line13, line14, line13, line14, m_line1B, m_line1E, m_b1, m_e1, imgDimensions,
                                externalPoints1, std::string("leftCommonRegion"));
                showCommonRegion(epipoles[1], line23, line24, line23, line24, m_line2B, m_line2E, m_b2, m_e2, imgDimensions,
                                externalPoints2, std::string("rightCommonRegion"));
            }
        } else {
            // CASE 3.2: Only the second epipole is inside
            const cv::Vec3f line11 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
            const cv::Vec3f line12 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[1]);

            vector <cv::Vec3f> inputLines(2), outputLines;
            inputLines[0] = line11;
            inputLines[1] = line12;
            computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
            const cv::Vec3f & line21 = outputLines[0];
            const cv::Vec3f & line22 = outputLines[1];

            m_line1B = line11;
            m_line1E = line12;
            m_line2B = line21;
            m_line2E = line22;

            m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
            m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);

            m_b2 = getNearestIntersection(epipoles[0], epipoles[1], m_line2B, m_b1, imgDimensions);
            m_e2 = getNearestIntersection(epipoles[0], epipoles[1], m_line2E, m_e1, imgDimensions);

            if (m_showCommonRegion) {
                showCommonRegion(epipoles[0], line11, line12, line11, line12, m_line1B, m_line1E, m_b1, m_e1, imgDimensions,
                                externalPoints1, std::string("leftCommonRegion"));
                showCommonRegion(epipoles[1], line21, line22, line21, line22, m_line2B, m_line2E, m_b2, m_e2, imgDimensions,
                                externalPoints2, std::string("rightCommonRegion"));
            }
        }
    }

    if (m_showCommonRegion) {
        cv::moveWindow("rightCommonRegion", imgDimensions.width + 10, 0);
    }
}

inline void PolarCalibration::determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions,
                       const vector<cv::Point2f> &externalPoints, double & minRho, double & maxRho) {
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            minRho = sqrt(epipole.x * epipole.x + epipole.y * epipole.y);         // Point A
            maxRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                          ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point D
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            minRho = -epipole.y;
            maxRho = max(sqrt(epipole.x * epipole.x +
                              ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y)),        // Point C
                         sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                              ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y))        // Point D
                        );
        } else { // Case 3
            minRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                          epipole.y * epipole.y);        // Point B
            maxRho = sqrt(epipole.x * epipole.x +
                          ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point C
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            minRho = -epipole.x;
            maxRho = max(
                         sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                              ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y)),        // Point D
                         sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                              epipole.y * epipole.y)        // Point B
                     );
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            minRho = 0;
            maxRho = max(
                         max(
                             sqrt(epipole.x * epipole.x + epipole.y * epipole.y),        // Point A
                             sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                                  epipole.y * epipole.y)        // Point B
                         ),
                         max(
                             sqrt(epipole.x * epipole.x +
                                  ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y)),        // Point C
                             sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                                  ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y))        // Point D
                         )
                     );
        } else { // Case 6
            minRho = epipole.x - (imgDimensions.width - 1);
            maxRho = max(
                         sqrt(epipole.x * epipole.x + epipole.y * epipole.y),        // Point A
                         sqrt(epipole.x * epipole.x +
                              ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y))        // Point C
                     );
        }
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            minRho = sqrt(epipole.x * epipole.x +
                          ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point C
            maxRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                          epipole.y * epipole.y);        // Point B
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            minRho = epipole.y - (imgDimensions.height - 1);
            maxRho = max(
                         sqrt(epipole.x * epipole.x + epipole.y * epipole.y),        // Point A
                         sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                              epipole.y * epipole.y)        // Point B

                     );
        } else { // Case 9
            minRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) +
                          ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point D
            maxRho = sqrt(epipole.x * epipole.x + epipole.y * epipole.y);        // Point A
        }
    }
}

inline void PolarCalibration::getNewPointAndLineSingleImage(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions,
                                    const cv::Mat & F, const uint32_t & whichImage, const cv::Point2d & pOld1, const cv::Point2d & pOld2,
                                   cv::Vec3f & prevLine, cv::Point2d & pNew1, cv::Vec3f & newLine1,
                                   cv::Point2d & pNew2, cv::Vec3f & newLine2) {


    // We obtain vector v
    cv::Vec2f v;

    cv::Vec3f vBegin(m_b1.x - epipole1.x, m_b1.y - epipole1.y, 0.0);
    cv::Vec3f vCurr(pOld1.x - epipole1.x, pOld1.y - epipole1.y, 0.0);
    cv::Vec3f vEnd(m_e1.x - epipole1.x, m_e1.y - epipole1.y, 0.0);

    vBegin /= cv::norm(vBegin);
    vCurr /= cv::norm(vCurr);
    vEnd /= cv::norm(vEnd);

    if (IS_INSIDE_IMAGE(epipole1, imgDimensions)) {
        if (IS_INSIDE_IMAGE(epipole2, imgDimensions)) {
            v = cv::Vec2f(vCurr[1], -vCurr[0]);
        } else {
            vBegin = cv::Vec3f(m_b2.x - epipole2.x, m_b2.y - epipole2.y, 0.0);
            vCurr = cv::Vec3f(pOld2.x - epipole2.x, pOld2.y - epipole2.y, 0.0);
            vEnd = cv::Vec3f(m_e2.x - epipole2.x, m_e2.y - epipole2.y, 0.0);

            vBegin /= cv::norm(vBegin);
            vCurr /= cv::norm(vCurr);
            vEnd /= cv::norm(vEnd);

            const cv::Vec3f vCross = vBegin.cross(vEnd);

            v = cv::Vec2f(vCurr[1], -vCurr[0]);
            if (vCross[2] > 0.0) {
                v = -v;
            }
        }
    } else {
        const cv::Vec3f vCross = vBegin.cross(vEnd);

        v = cv::Vec2f(vCurr[1], -vCurr[0]);
        if (vCross[2] > 0.0) {
            v = -v;
        }
    }

    pNew1 = cv::Point2d(pOld1.x + v[0] * m_stepSize, pOld1.y + v[1] * m_stepSize);
    newLine1 = GET_LINE_FROM_POINTS(epipole1, pNew1);

    if (! IS_INSIDE_IMAGE(epipole1, imgDimensions)) {
        pNew1 = getBorderIntersection(epipole1, newLine1, imgDimensions, &pOld1);
    } else {
        pNew1 = getNearestIntersection(epipole1, epipole1, newLine1, pOld1, imgDimensions);
    }

    vector<cv::Point2f> points(1);
    points[0] = pNew1;
    vector<cv::Vec3f> inLines(1);
    inLines[0] = newLine1;
    vector<cv::Vec3f> outLines(1);
    computeEpilines(points, whichImage, F, inLines, outLines);
    newLine2 = outLines[0];

    if (! IS_INSIDE_IMAGE(epipole2, imgDimensions)) {
        cv::Point2d tmpPoint = getBorderIntersection(epipole2, newLine2, imgDimensions, &pOld2);
        pNew2 = tmpPoint;
    } else {
        vector <cv::Point2d> intersections;
        getBorderIntersections(epipole2, newLine2, imgDimensions, intersections);
        pNew2 = intersections[0];

        double minDist = std::numeric_limits<double>::max();
        for (uint32_t i = 0; i < intersections.size(); i++) {
            const double dist = (pOld2.x - intersections[i].x) * (pOld2.x - intersections[i].x) +
                                (pOld2.y - intersections[i].y) * (pOld2.y - intersections[i].y);
            if (minDist > dist) {
                minDist = dist;
                pNew2 = intersections[i];
            }
        }
    }

}

inline void PolarCalibration::getNewEpiline(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions,
                                     const cv::Mat & F, const cv::Point2d pOld1, const cv::Point2d pOld2,
                                     cv::Vec3f prevLine1, cv::Vec3f prevLine2,
                                     cv::Point2d & pNew1, cv::Point2d & pNew2, cv::Vec3f & newLine1, cv::Vec3f & newLine2) {

    getNewPointAndLineSingleImage(epipole1, epipole2, imgDimensions, F, 1, pOld1, pOld2, prevLine1, pNew1, newLine1, pNew2, newLine2);

    //TODO If the distance is too big in image 2, we do it in the opposite sense
//     double distImg2 = (pOld2.x - pNew2.x) * (pOld2.x - pNew2.x) + (pOld2.y - pNew2.y) * (pOld2.y - pNew2.y);
//     if (distImg2 > m_stepSize * m_stepSize)
//         getNewPointAndLineSingleImage(epipole2, epipole1, imgDimensions, F, 2, pOld2, pOld1, prevLine2, pNew2, newLine2, pNew1, newLine1);

    if (m_showIterations) {
        showNewEpiline(epipole1, m_line1B, m_line1E, newLine1, pOld1, pNew1, imgDimensions, std::string("newEpiline1"));
        showNewEpiline(epipole2, m_line2B, m_line2E, newLine2, pOld2, pNew2, imgDimensions, std::string("newEpiline2"));
        cv::moveWindow("newEpiline2", imgDimensions.width +10, 0);
    }
}

inline void PolarCalibration::transformLine(const cv::Point2d& epipole, const cv::Point2d& p2, const cv::Mat& inputImage,
                                            const uint32_t & thetaIdx, const double &minRho, const double & maxRho,
                                            cv::Mat& mapX, cv::Mat& mapY, cv::Mat& inverseMapX, cv::Mat& inverseMapY)
{
    cv::Vec2f v(p2.x - epipole.x, p2.y - epipole.y);
    double maxDist = cv::norm(v);
    v /= maxDist;

    {
        uint32_t rhoIdx = 0;
        for (double rho = minRho; rho <= min(maxDist, maxRho); rho += 1.0, rhoIdx++) {
            cv::Point2d target(v[0] * rho + epipole.x, v[1] * rho + epipole.y);
            if ((target.x >= 0) && (target.x < inputImage.cols) &&
                (target.y >= 0) && (target.y < inputImage.rows)) {

                mapX.at<float>(thetaIdx, rhoIdx) = target.x;
                mapY.at<float>(thetaIdx, rhoIdx) = target.y;

                inverseMapX.at<float>(target.y, target.x) = rhoIdx;
                inverseMapY.at<float>(target.y, target.x) = thetaIdx;
            }
        }
    }
}

void PolarCalibration::getTransformationPoints(const cv::Size& imgDimensions, const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Mat& F)
{
    cv::Point2d p1 = m_b1, p2 = m_b2;
    cv::Vec3f line1 = m_line1B, line2 = m_line2B;

    m_thetaPoints1.clear();
    m_thetaPoints2.clear();
    m_thetaPoints1.reserve(2 * (imgDimensions.width + imgDimensions.height));
    m_thetaPoints2.reserve(2 * (imgDimensions.width + imgDimensions.height));

    int32_t crossesLeft = 0;
    if (IS_INSIDE_IMAGE(epipole1, imgDimensions) && IS_INSIDE_IMAGE(epipole2, imgDimensions))
        crossesLeft++;

    uint32_t thetaIdx = 0;
    double lastCrossProd = 0;

    while (true) {
        m_thetaPoints1.push_back(p1);
        m_thetaPoints2.push_back(p2);
//         transformLine(epipole1, p1, img1, thetaIdx, m_minRho1, m_maxRho1, m_mapX1, m_mapY1, m_inverseMapX1, m_inverseMapY1);
//         transformLine(epipole2, p2, img2, thetaIdx, m_minRho2, m_maxRho2, m_mapX2, m_mapY2, m_inverseMapX2, m_inverseMapY2);

        cv::Vec3f v0(p1.x - epipole1.x, p1.y - epipole1.y, 1.0);
        v0 /= cv::norm(v0);
        cv::Point2d oldP1 = p1;

        getNewEpiline(epipole1, epipole2, imgDimensions, F, p1, p2, line1, line2, p1, p2, line1, line2);

        // Check if we reached the end
        cv::Vec3f v1(p1.x - epipole1.x, p1.y - epipole1.y, 0.0);
        v1 /= cv::norm(v1);
        cv::Vec3f v2(m_e1.x - epipole1.x, m_e1.y - epipole1.y, 0.0);
        v2 /= cv::norm(v2);
        cv::Vec3f v3(oldP1.x - epipole1.x, oldP1.y - epipole1.y, 0.0);
        v3 /= cv::norm(v3);

        double crossProd = v1.cross(v2)[2];

        if (thetaIdx != 0) {
            if ((SIGN(lastCrossProd) != SIGN(crossProd)) || (fabs(acos(v1.dot(-v3))) < 0.01) || (p1 == cv::Point2d(-1, -1)))
                crossesLeft--;

            if ((crossesLeft < 0)) {
                break;
            }
        }
        lastCrossProd = crossProd;
        thetaIdx++;

        if (m_showIterations) {
            int keycode = cv::waitKey(0);

            // q: exit
            if (keycode == 113) {
                exit(0);
            }
            // 1: stepSize = 1
            if (keycode == 49) {
                m_stepSize = 1;
            }
            // 2: stepSize = 50
            if (keycode == 50) {
                m_stepSize = 10;
            }
            // 3: stepSize = 50
            if (keycode == 51) {
                m_stepSize = 50;
            }
            // 4: stepSize = 100
            if (keycode == 51) {
                m_stepSize = 100;
            }
            // n: next image
            if (keycode == 110) {
                break;
            }
            // r: reset current image
            if (keycode == 114) {
                p1 = m_b1;
                p2 = m_b2;
                line1 = m_line1B;
                line2 = m_line2B;
            }
        }
    }
    m_thetaPoints1.pop_back();
    m_thetaPoints2.pop_back();
}

void PolarCalibration::doTransformation(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Mat & F) {

    const double rhoRange1 = m_maxRho1 - m_minRho1 + 1;
    const double rhoRange2 = m_maxRho2 - m_minRho2 + 1;

    const double rhoRange = max(rhoRange1, rhoRange2);

    m_mapX1 = cv::Mat::ones(m_thetaPoints1.size(), rhoRange, CV_32FC1) * -1;
    m_mapY1 = cv::Mat::ones(m_thetaPoints1.size(), rhoRange, CV_32FC1) * -1;
    m_mapX2 = cv::Mat::ones(m_thetaPoints2.size(), rhoRange, CV_32FC1) * -1;
    m_mapY2 = cv::Mat::ones(m_thetaPoints2.size(), rhoRange, CV_32FC1) * -1;

    m_inverseMapX1 = cv::Mat::ones(img1.rows, img1.cols, CV_32FC1) * -1;
    m_inverseMapY1 = cv::Mat::ones(img1.rows, img1.cols, CV_32FC1) * -1;
    m_inverseMapX2 = cv::Mat::ones(img1.rows, img1.cols, CV_32FC1) * -1;
    m_inverseMapY2 = cv::Mat::ones(img1.rows, img1.cols, CV_32FC1) * -1;

    for (uint32_t thetaIdx = 0; thetaIdx < m_thetaPoints1.size(); thetaIdx++) {
        transformLine(epipole1, m_thetaPoints1[thetaIdx], img1, thetaIdx, m_minRho1, m_maxRho1, m_mapX1, m_mapY1, m_inverseMapX1, m_inverseMapY1);
        transformLine(epipole2, m_thetaPoints2[thetaIdx], img2, thetaIdx, m_minRho2, m_maxRho2, m_mapX2, m_mapY2, m_inverseMapX2, m_inverseMapY2);
    }

    cv::remap(img1, m_rectified1, m_mapX1, m_mapY1, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
    cv::remap(img2, m_rectified2, m_mapX2, m_mapY2, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
}

void PolarCalibration::getRectifiedImages(const cv::Mat& img1, const cv::Mat& img2,
                                          cv::Mat& rectified1, cv::Mat& rectified2, int interpolation) {

    cv::remap(img1, rectified1, m_mapX1, m_mapY1, interpolation, cv::BORDER_TRANSPARENT);
    cv::remap(img2, rectified2, m_mapX2, m_mapY2, interpolation, cv::BORDER_TRANSPARENT);
}

void PolarCalibration::getOriginalImages(const cv::Mat& rectified1, const cv::Mat& rectified2,
                                          cv::Mat& img_orig1, cv::Mat& img_orig2, int interpolation) {

    cv::remap(rectified1, img_orig1, m_inverseMapX1, m_inverseMapY1, interpolation, cv::BORDER_TRANSPARENT);
    cv::remap(rectified2, img_orig2, m_inverseMapX2, m_inverseMapY2, interpolation, cv::BORDER_TRANSPARENT);

    display1 = img_orig1;
    display2 = img_orig2;
}

void PolarCalibration::rectifyAndStoreImages(const cv::Mat& img1, const cv::Mat& img2, int interpolation)
{
    getRectifiedImages(img1, img2, m_rectified1, m_rectified2, interpolation);
}

bool PolarCalibration::getStoredRectifiedImages(cv::Mat& img1, cv::Mat& img2)
{
    img1 = m_rectified1;
    img2 = m_rectified2;
}

void PolarCalibration::getMaps(cv::Mat& mapX, cv::Mat& mapY, const uint8_t& whichImage)
{
    if (whichImage == 1) {
        mapX = m_mapX1;
        mapY = m_mapY1;
    } else if (whichImage == 2) {
        mapX = m_mapX2;
        mapY = m_mapY2;
    }
}

void PolarCalibration::getInverseMaps(cv::Mat& mapX, cv::Mat& mapY, const uint8_t& whichImage)
{
    if (whichImage == 1) {
        mapX = m_inverseMapX1;
        mapY = m_inverseMapY1;
    } else if (whichImage == 2) {
        mapX = m_inverseMapX2;
        mapY = m_inverseMapY2;
    }
}

void PolarCalibration::transformPoints(const vector< cv::Point2d >& points,
                                       vector< cv::Point2d >& transformedPoints, const uint8_t & whichImage)
{
    cv::Mat * pMapX, * pMapY;
    /*
    if (whichImage == 1) {
        pMapX = &m_inverseMapX1;
        pMapY = &m_inverseMapY1;
    } else if (whichImage == 2) {
        pMapX = &m_inverseMapX2;
        pMapY = &m_inverseMapY2;
    }
    */
    if (whichImage == 1) {
        pMapX = &m_mapX1;
        pMapY = &m_mapY1;
    } else if (whichImage == 2) {
        pMapX = &m_mapX2;
        pMapY = &m_mapY2;
    }

    transformedPoints.clear();
    transformedPoints.reserve(points.size());
    for (uint32_t i = 0; i < points.size(); i++) {
        const cv::Point2d & p = points[i];
        transformedPoints.push_back(
                    cv::Point2d(  (pMapX->at<float>(floor(p.y), floor(p.x)) +
                                  pMapX->at<float>(floor(p.y), ceil(p.x)) +
                                  pMapX->at<float>(ceil(p.y), floor(p.x)) +
                                  pMapX->at<float>(ceil(p.y), ceil(p.x)) )/4.0,

                                  (pMapY->at<float>(floor(p.y), floor(p.x)) +
                                  pMapY->at<float>(floor(p.y), ceil(p.x)) +
                                  pMapY->at<float>(ceil(p.y), floor(p.x)) +
                                  pMapY->at<float>(ceil(p.y), ceil(p.x)) )/4.0
                                ));

//         cout << points[i] << " --> " << transformedPoints[i] << endl;
    }
//     exit(0);
}

bool PolarCalibration::compute3DMap(cv::Mat& disparity) {

  vector<cv::Point2d> p_left, p_right;

  std::cout << "disparity rows: " << disparity.rows << std::endl;
  std::cout << "disparity cols: " << disparity.cols << std::endl;

  if (disparity.type() == CV_16SC1) {
    for(int i=0; i<disparity.rows; i++)
      for(int j=0; j<disparity.cols; j++) {
        if(disparity.at<short>(i,j) > 5*16) {
          p_left.push_back(cv::Point2d(double(j), double(i)));
          p_right.push_back(cv::Point2d(double(j - (float(disparity.at<short>(i,j)) / 16.0)), double(i)));
        }
      }
  } else if (disparity.type() == CV_32FC1) {
    for(int i=0; i<disparity.rows; i++)
      for(int j=0; j<disparity.cols; j++) {
        if(disparity.at<short>(i,j) > 5*16) {
          p_left.push_back(cv::Point2d(double(j), double(i)));
          p_right.push_back(cv::Point2d(double(j - disparity.at<float>(i,j)), double(i)));
        }
      }
  } else {
    std::cerr << "No expected disparity type! Type: " << disparity.type() << std::endl;
    return false;
  }

  /* Transform points in image space */
  vector<cv::Point2d> p_left_im, p_right_im;
  transformPoints(p_left, p_left_im, 1);
  transformPoints(p_right, p_right_im, 2);

  cv::Mat rect, rectRGB;
  cv::addWeighted(m_rectified2, 0.5, m_rectified1, 0.5, 0.0, rect);
  cv::cvtColor(rect, rectRGB, CV_GRAY2RGB);


  vector<cv::Point2d> p_left_im_filt, p_right_im_filt;
  vector<cv::Point2d> p_left_filt, p_right_filt;
  cv::Mat display, displayRGB;
  cv::addWeighted(display1, 0.5, display2, 0.5, 0.0, display);
  cv::cvtColor(display, displayRGB, CV_GRAY2RGB);

  for (int i=0; i<p_right_im.size(); i=i+100) {
    if(p_right_im[i].x < disparity.cols && p_right_im[i].x > 10.0 &&
       // p_right_im[i].y < disparity.rows && p_right_im[i].y > 10.0 &&
     ( (p_right_im[i].y - p_left_im[i].y)*(p_right_im[i].y - p_left_im[i].y) + (p_right_im[i].x - p_left_im[i].x)*(p_right_im[i].x - p_left_im[i].x) ) < 50*50 &&
     ( (p_right_im[i].y - p_left_im[i].y)*(p_right_im[i].y - p_left_im[i].y) + (p_right_im[i].x - p_left_im[i].x)*(p_right_im[i].x - p_left_im[i].x) ) > 10*10 ) {
      p_left_filt.push_back(p_left[i]);
      p_right_filt.push_back(p_right[i]);
      p_left_im_filt.push_back(p_left_im[i]);
      p_right_im_filt.push_back(p_right_im[i]);
      /*
      std::cout << "point" << std::endl;
      std::cout << p_left[i] << std::endl;
      std::cout << p_right[i] << std::endl;
      std::cout << p_left_im[i] << std::endl;
      std::cout << p_right_im[i] << std::endl; */
      cv::circle(displayRGB, p_right_im[i], 3, cv::Scalar(0,255,0));
      cv::line(displayRGB, p_left_im[i] , p_right_im[i], cv::Scalar(200,100,50));

      cv::circle(rectRGB, p_right[i], 3, cv::Scalar(0,255,0));
      cv::line(rectRGB, p_left[i] , p_right[i], cv::Scalar(200,100,50));
    }
  }

  cv::imshow("ciccio", displayRGB);
  cv::imshow("ciccio2", rectRGB);

  /* Decompose essential matrix in roto translations (not necessary if poses are already known) */
  cv::Mat E = Kmat.t() * Fund * Kmat;
  std::cout << "Fundamental matrix: " << std::endl << Fund << std::endl;

  cv::Mat u,w,vt;
  cv::SVD::compute(E,w,u,vt);
  cv::Mat t;
  u.col(2).copyTo(t);

  t=t/cv::norm(t);

  cv::Mat W(3,3,CV_64F,cv::Scalar(0));
  W.at<double>(0,1)=-1;
  W.at<double>(1,0)=1;
  W.at<double>(2,2)=1;

  cv::Mat R1 = u*W*vt;
  if(cv::determinant(R1)<0)
      R1=-R1;

  std::cout << R1 << std::endl;

  cv::Mat R2 = u*W.t()*vt;
  if(cv::determinant(R2)<0)
      R2=-R2;

  std::cout << R2 << std::endl;

  R1.convertTo(R1, CV_64FC1);
  R2.convertTo(R2, CV_64FC1);
  t.convertTo(t, CV_64FC1);
  Kmat.convertTo(Kmat, CV_64FC1);

  /* Triangulate points */
  cv::Mat t1 = cv::Mat::zeros(3,1,CV_64FC1);
  cv::Mat R = cv::Mat::eye(3,3,CV_64FC1);

  cv::Mat rt1;
  cv::hconcat(R, t1, rt1);

  cv::Mat rt2;
  std::cout << R1 << std::endl;

  std::cout << t << std::endl;
  cv::hconcat(R1, t, rt2);

  cv::Mat proj1 = Kmat * rt1;
  cv::Mat proj2 = Kmat * rt2;

  std::cout << proj1 << std::endl;
  std::cout << proj2 << std::endl;

  cv::Mat points1(1, p_left_filt.size(), CV_64FC2);
  cv::Mat points2(1, p_right_filt.size(), CV_64FC2);

  for(int i=0; i<p_left_filt.size(); i++) {
    points1.at<cv::Vec2d>(i)[0] = p_left_im_filt[i].x;
    points1.at<cv::Vec2d>(i)[1] = p_left_im_filt[i].y;
  }

  for(int i=0; i<p_right_filt.size(); i++) {
    points2.at<cv::Vec2d>(i)[0] = p_right_im_filt[i].x;
    points2.at<cv::Vec2d>(i)[1] = p_right_im_filt[i].y;
  }

  cv::Mat p3D(1,p_right_filt.size(),CV_64FC4);
  cv::triangulatePoints(proj1, proj2, points1, points2, p3D);

  /* Open visualizer */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int i=0; i<p_right_filt.size(); i++) {

    /*std::cout << "test: " << std::endl;
    std::cout << points1.at<cv::Vec2d>(i)[0] << " " << points1.at<cv::Vec2d>(i)[1] << " -> " << points2.at<cv::Vec2d>(i)[0] << " " << points2.at<cv::Vec2d>(i)[1] << std::endl;
    std::cout << p3D.at<double>(0, i) << " " <<
                 p3D.at<double>(1, i) << " " <<
                 p3D.at<double>(2, i) << " " <<
                 p3D.at<double>(3, i) << std::endl;*/

    pcl::PointXYZRGB p;
    /*std::cout << p3D.at<cv::Vec4d>(i)[0] << " " <<
                 p3D.at<cv::Vec4d>(i)[1] << " " <<
                 p3D.at<cv::Vec4d>(i)[2] << " " <<
                 p3D.at<cv::Vec4d>(i)[3] << std::endl; */
    p.x = p3D.at<double>(0, i) / p3D.at<double>(3, i);
    p.y = p3D.at<double>(1, i) / p3D.at<double>(3, i);
    p.z = p3D.at<double>(2, i) / p3D.at<double>(3, i);
    p.r = 200;
    p.g = 200;
    p.b = 200;
    if (abs(p.x) < 10000 && abs(p.y) < 10000 && abs(p.z) < 10000) cloud->push_back(p);
  }

  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  viewer->spin();






};

/*inline
bool PolarCalibration::isVectorBetweenVectors(const cv::Vec3f& v, const cv::Vec3f& v1, const cv::Vec3f& v2)
{
    if (SIGN(v1.cross(v)[2]) == SIGN(v.cross(v2)[2])) {
        if (fabs(acos(v1.dot(v2))) > fabs(acos(v1.dot(v)))) {
            return true;
        }
    }
    return false;
}

bool PolarCalibration::findThetaIdx(const cv::Point2d& epipole, const cv::Point2d &queryPoint,
                                    const vector<cv::Point2d> & thetaPoints,
                                    const uint32_t& minIdx, const uint32_t& maxIdx, double & thetaIdx)
{
//     cout << "queryPoint " << queryPoint << endl;
//     cout << "minPoint " << thetaPoints[minIdx] << endl;
//     cout << "maxPoint " << thetaPoints[maxIdx] << endl;
//     cout << "m_b1 " << m_b1 << endl;
//     cout << "m_e1 " << m_e1 << endl;

    cv::Vec3f v(queryPoint.x - epipole.x, queryPoint.y - epipole.y, 0.0);
    cv::Vec3f v1(thetaPoints[minIdx].x - epipole.x, thetaPoints[minIdx].y - epipole.y, 0.0);
    cv::Vec3f v2(thetaPoints[maxIdx].x - epipole.x, thetaPoints[maxIdx].y - epipole.y, 0.0);

    v /= cv::norm(v);
    v1 /= cv::norm(v1);
    v2 /= cv::norm(v2);

    if (! isVectorBetweenVectors(v, v1, v2))
        return false;

    if (maxIdx - minIdx == 1) {
        const double & dist1 = cv::norm(queryPoint - thetaPoints[minIdx]);
        const double & dist2 = cv::norm(queryPoint - thetaPoints[maxIdx]);

//         cout << dist1 << endl;
//         cout << dist2 << endl;
//         cout << dist1 << endl;

        thetaIdx = minIdx + (dist1 / (dist1 + dist2));
//         if (dist1 < dist2)
//             thetaIdx = minIdx;
//         else
//             thetaIdx = maxIdx;

        return true;
    }

    const uint32_t minIdx1 = minIdx;
    const uint32_t maxIdx1 = minIdx + ((maxIdx - minIdx + 1) / 2);
    const uint32_t minIdx2 = maxIdx1;
    const uint32_t maxIdx2 = maxIdx;

    if (findThetaIdx(epipole, queryPoint, thetaPoints, minIdx1, maxIdx1, thetaIdx))
        return true;
    if (findThetaIdx(epipole, queryPoint, thetaPoints, minIdx2, maxIdx2, thetaIdx))
        return true;
}

bool PolarCalibration::transformPoints(const cv::Mat& Fin, const vector< cv::Point2f >& ctrlPoints1,
                                       const vector< cv::Point2f >& ctrlPoints2, const cv::Size & imgDimensions,
                                       const vector< cv::Point2d >& points1, const vector< cv::Point2d >& points2,
                                       vector< cv::Point2d >& transformedPoints1, vector< cv::Point2d >& transformedPoints2)
{
    cv::Point2d epipole1, epipole2;
    const cv::Mat img1(2, 2, CV_8UC3), img2(2, 2, CV_8UC3);
    cv::Mat F = Fin;
    if (! findFundamentalMat(img1, img2, F, ctrlPoints1, ctrlPoints2, epipole1, epipole2, FMAT_METHOD_OFLOW))
        return false;

//     Determine common region
    vector<cv::Vec3f> initialEpilines, finalEpilines;
    vector<cv::Point2f> epipoles(2);
    epipoles[0] = epipole1;
    epipoles[1] = epipole2;
    determineCommonRegion(epipoles, imgDimensions, F);

    getTransformationPoints(img1.size(), epipole1, epipole2, F);

    transformedPoints1.reserve(points1.size());
    transformedPoints2.reserve(points2.size());

    for (uint32_t i = 0; i < points1.size(); i++) {
        double thetaIdx;
        if (! findThetaIdx(epipole1, points1[i], m_thetaPoints1, 0, m_thetaPoints1.size() / 2, thetaIdx))
            findThetaIdx(epipole1, points1[i], m_thetaPoints1, m_thetaPoints1.size() / 2, m_thetaPoints1.size() - 1, thetaIdx);
        uint32_t rhoIdx = round(cv::norm(epipole1 - points1[i]) - m_minRho1);

        cout << "p " << points1[i] << endl;
        cout << "e " << epipole1 << endl;
        cout << "d " << cv::norm(epipole1 - points1[i]) << endl;
        cout << "r " << m_minRho1 << endl;

//         exit(0);

//         transformedPoints1.push_back(cv::Point2d(rhoIdx, thetaIdx + 1));
        transformedPoints1.push_back(cv::Point2d(cv::norm(epipole1 - points1[i]) - m_minRho1, thetaIdx + 1));
    }

    return true;

}*/
