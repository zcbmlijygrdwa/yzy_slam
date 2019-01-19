//for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
// for opencv 
#include <opencv2/core/core.hpp>

#include <vector>

#include "Keypoint.hpp"

class Frame
{

    public:
        //std::vector<Keypoint> features;
        cv::Mat features_desp;
        std::vector<cv::Point2d> features_2d;
        std::vector<cv::Point3d> features_3d;
        Eigen::Isometry3d T;   //transform relative to the world origin

        Frame()
        {
            T = Eigen::Isometry3d::Identity(); //init as identity matrix
        }
        
        void add_features_desp(cv::Mat features_desp)
        {
            Frame::features_desp = features_desp;
        }

        void add_features_2d(std::vector<cv::Point2d> features_2d)
        {
            Frame::features_2d = features_2d;
        }

        void add_features_3d(std::vector<cv::Point3d> features_3d)
        {
            Frame::features_3d = features_3d;
        }

        int feature_count()
        {
            if(features_3d.size()==features_2d.size())
            {
                return features_3d.size();
            }
            else
            {
                return -1;
            }
        }

};
