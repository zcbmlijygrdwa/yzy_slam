#ifndef Pose3D2D_h
#define Pose3D2D_f

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../cvminecraft/pose_estimation/case_3D_2D/SolvePnpCeres.hpp"

class Pose3D2D
{
    public:
        cv::Mat R_new;
        cv::Mat t_new;

        bool if_use_ceres = false;

        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;

        cv::Mat K_mat; //camera instrinsics

        void setUseCeres(bool in)
        {
            if_use_ceres = in;
        }

        template <class T>
            void setPoints3d(T data)
            {
                std::cout<<"data.size() = "<<data.size()<<std::endl;
                for(uint i = 0 ; i<data.size() ; i++)
                {
                    //std::cout<<"i = "<<i<<std::endl;
                    pts_3d.push_back(cv::Point3d(data[i].x,data[i].y,data[i].z));
                }
            }



        template <class T>
            void setPoints2d(T data)
            {
                for(uint i = 0 ; i<data.size() ; i++)
                {
                    pts_2d.push_back(cv::Point2d(data[i].x,data[i].y));
                }
            }

        void setCameraIntrinsic(cv::Mat K_mat_in)
        {
            K_mat = K_mat_in;
        }


        cv::Mat rotation()
        {
            return R_new;
        }

        cv::Mat translation()
        {
            return t_new;
        }

        void solve()
        {
            //std::cout<<"1"<<endl;
            if(pts_3d.size()==0||pts_2d.size()==0)
            {
                std::cout<<"zero size 3D points or 2D points. Solve failed."<<std::endl;
            }
            else if(pts_3d.size()!=pts_2d.size())
            {
                std::cout<<"Different sizes of 3D points or 2D points. Solve failed."<<std::endl;
            }


            //first use opencv's solvepnp
            solvePnP(pts_3d,pts_2d,K_mat,cv::Mat(),R_new,t_new,false,CV_ITERATIVE);


            //then use ceres to optimize
            if(if_use_ceres)
            {
                SolvePnpCeres solvepnpceres;
                solvepnpceres.init();
                solvepnpceres.setInputs(pts_3d,pts_2d,K_mat);
                solvepnpceres.optimize(&R_new,&t_new,false);
            }
        }
}; 

#endif
