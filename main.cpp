/**
 * BA Example 
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 * 
 * 在这个程序中，我们读取两张图像，进行特征匹配。然后根据匹配得到的特征，计算相机运动以及特征点的位置。这是一个典型的Bundle Adjustment，我们用g2o进行优化。
 */


// for std
#include <iostream>

//for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
// for opencv 
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/concept_check.hpp>


//PCL
#include <pcl/visualization/cloud_viewer.h>

#include "Frame.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;


bool is_slam_init = false;
Frame ref_frame;

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)   {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize=Size(21,21);                                                               
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    int status_size = status.size();
    for( int i=0; i<status_size; i++)
    {  Point2f pt = points2.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))    {
            if((pt.x<0)||(pt.y<0))    {
                status.at(i) = 0;
            }
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}


int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& desp2, cv::Mat& img1_with_features);
//const int MAX_FEATURES = 500;
const int MAX_FEATURES = 500;
// 相机内参
double cx = 239.961714;
double cy = 256.842130;
double fx = 814.660678;
double fy = 815.013833;

clock_t deltaTime = 0;
unsigned int frames = 0;
int frameCount = 0;
double  frameRate = 30;

Mat traj_image = Mat::zeros( 800, 800, CV_8UC1);

double clockToMilliseconds(clock_t ticks){
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
}



void visulizePose2d(Mat& traj_image,Isometry3d& pose_in)
{
    Vector3d translation = pose_in.translation();

    double drawX = -translation(0);
    double drawY = translation(1);
    //drawX*=50;
    //drawY*=50;
    drawX = (int)drawX+traj_image.cols/2;
    drawY = (int)drawY+traj_image.rows/2;
    //cout<<"drawX = "<<drawX<<", drawY = "<<drawY<<endl;
    Point drawP = Point(drawX,drawY);
    line(traj_image,drawP,drawP,Scalar(255,255,255),3,8);
    imshow("traj_image", traj_image);
}


void vis_3d_points(vector<Point3f> pts)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr;
    pcl::PointCloud<pcl::PointXYZ> pointCloud;

    for(int i = 0 ; i< pts.size() ; i++)
    {
        pcl::PointXYZ point;
        point.x = pts[i].x;
        point.y = pts[i].y;
        point.z = pts[i].z;
        pointCloud.points.push_back(point);

    }

    cout<<"asdf"<<endl;
    pointCloudPtr = pointCloud.makeShared();
    cout<<"2asdf"<<endl;
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (pointCloudPtr);
    while (!viewer.wasStopped ())
    {

    }


}


int main( int argc, char** argv )
{



    VideoCapture cap;

    if(argc!=3)
    {
        cout<<"Useage:"<<endl<<"./yzy_vo cam [cameraIdex]"<<endl<<"./yzy_vo video [pathTovideo]"<<endl;
        return -1;
    }

    if(strcmp(argv[1],"video")==0)
    {
        cout<<"Run as video input mode"<<endl;
        cap = VideoCapture(argv[2]);
    }
    else if(strcmp(argv[1],"cam")==0)
    {
        int cameraIdx = atoi(argv[2]);

        cout<<"Run as web_cam mode, preparing camera["<<cameraIdx<<"]"<<endl;
        cap = VideoCapture(cameraIdx);
    }
    else
    {
        cout<<"Useage:"<<endl<<"./yzy_vo cam [cameraIdex]"<<endl<<"./yzy_vo video [pathTovideo]"<<endl;

        return -1;
    }

    if(!cap.isOpened())  // check if we succeeded
    {
        cout<<"camera not open"<<endl;
        return -1;
    }


    cv::Mat img1; 
    cv::Mat img2; 


    //create Isometry object to keep tracking of pose
    Isometry3d pose_global = Isometry3d::Identity();

    pose_global.rotate(AngleAxisd(3.141592653*0.5,Vector3d::UnitX()));

    vector<uchar> status;

    vector<Point2f> prevFeatures;
    vector<Point2f> currFeatures;



    Mat img1_with_features;
    cap>>img1;
    //resize(img1, img1, cv::Size(), resizeFactor, resizeFactor);

    cx = img1.cols/2;
    cy = img1.rows/2;
    cout<<"cx = "<<cx<<endl;
    cout<<"cy = "<<cy<<endl;

    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    vector<cv::KeyPoint> kp1;
    cv::Mat desp1;
    orb->detectAndCompute( img1, cv::Mat(), kp1, desp1 );
    vector<Point2f> tempPoints1;
    cout<<"kp1.size() = "<<kp1.size()<<endl;
    for(auto tempKp:kp1)
    { 
        tempPoints1.push_back( tempKp.pt );
    }
    prevFeatures = tempPoints1;


    while(cap.isOpened())
    {
        clock_t beginFrame = clock();


        cap>>img2;

        //get a frame every 10 frames
        //doing this to make sure feature points move enough distance on image, if distance too small, the camera small motion estimation will not be accurate.
        if(frameCount%5!=0)
        {
            frameCount++;
            continue;
        }

        //resize(img2, img2, cv::Size(), resizeFactor, resizeFactor);

        // 找到对应点
        vector<cv::Point2f> pts1, pts2;
        cv::Mat desp2;


        if ( findCorrespondingPoints( img1, img2, pts1, pts2, desp2, img1_with_features) == false )
        {
            //imshow("img1", img1);
            //imshow("img2", img2);
            img2.copyTo(img1);
            cout<<"Insufficient matching!"<<endl;
            continue;
        }
        //cout<<"找到了"<<pts1.size()<<"组对应特征点。"<<endl;

        //prevFeatures = pts1;

        vector<Point2f> currFeatures;
        currFeatures = pts2; 

        //feature tracking

        //cout<<"prevFeatures.size() = "<<prevFeatures.size()<<endl;
        //cout<<"currFeatures.size() = "<<currFeatures.size()<<endl;
        //featureTracking(img1, img2, prevFeatures, currFeatures, status);
        //cout<<"2prevFeatures.size() = "<<prevFeatures.size()<<endl;
        //cout<<"2currFeatures.size() = "<<currFeatures.size()<<endl;

        //use epipolar constrain
        cv::Mat mask;
        cv::Mat e_mat;
        e_mat = cv::findEssentialMat(pts1,pts2,fx,cv::Point2f(cx,cy),cv::RANSAC, 0.999, 1.f,mask);
        //e_mat = cv::findEssentialMat(prevFeatures,currFeatures,fx,cv::Point2f(cx,cy),cv::RANSAC, 0.999, 1.f,mask);
        //cout << "E:" << endl << e_mat/e_mat.at<double>(2,2) << endl;
        cv::Mat R, t;
        cv::recoverPose(e_mat, pts1, pts2, R, t,fx,cv::Point2f(cx,cy),mask);

        Mat K_mat = Mat::zeros(3,3,CV_64F);
        K_mat.at<double>(0,0) = fx;
        K_mat.at<double>(1,1) = fy;

        K_mat.at<double>(0,2) = cx;
        K_mat.at<double>(1,2) = cy;

        K_mat.at<double>(2,2) = 1;

        cout<<"K_mat = "<<endl<<K_mat<<endl;

        if(!is_slam_init)
        {
            //calculate 3D points
            cout<<"Initializing SLAM, setting the first two frames translation as 1."<<endl;


            Mat T1_cv;
            //the rotation of T1 is not ZERO ROTATION matrix, need to rotate roll so that the Z axis point forward(camera)

            float pi = 3.141592653;
            
            cout<<"AngleAxisf(pi*0.5f,Vector3f::UnitX()) = "<<AngleAxisf(pi*0.5f,Vector3f::UnitX()).toRotationMatrix()<<endl;
            
            Matrix3f rot_x;
            rot_x = AngleAxisf(pi*0.5f,Vector3f::UnitX());
            Mat rot_x_mat;
            eigen2cv(rot_x,rot_x_mat);
            rot_x_mat.convertTo(rot_x_mat,CV_64F);
            cout<<"rot_x_mat = "<<endl<<rot_x_mat<<endl;
            cout<<"R = "<<endl<<R<<endl;
        
            hconcat(rot_x_mat,Mat::zeros(3, 1, CV_64F),T1_cv);

            Mat T2_cv;
            hconcat(R,t,T2_cv);

            cout<<"T1_cv = "<<endl<<T1_cv<<endl;
            cout<<"T2_cv = "<<endl<<T2_cv<<endl;

            Mat proj_cam_1;
            Mat proj_cam_2;

            proj_cam_1 = K_mat*T1_cv;
            proj_cam_2 = K_mat*T2_cv;

            cout<<"proj_cam_1 = "<<endl<<proj_cam_1<<endl;
            cout<<"proj_cam_2 = "<<endl<<proj_cam_2<<endl;
            //call triangulatePoints from opencv
            Mat results;
            triangulatePoints(proj_cam_1,proj_cam_2,pts1,pts2,results);
            //transpose
            results = results.t();

            MatrixXd results_eigen;
            cv2eigen(results,results_eigen);


            vector<Point3f> points3d;
            for(int i = 0; i < results_eigen.rows() ; i++)
            {
                results_eigen.row(i) /= results_eigen(i,3);
                points3d.push_back(Point3f(results_eigen(i,0),results_eigen(i,1),results_eigen(i,2)));


            }
            cout<<"results_eigen = "<<endl<<results_eigen<<endl;

            vis_3d_points(points3d);

            ref_frame.add_features_2d(pts2);
            ref_frame.add_features_3d(points3d);
            ref_frame.add_features_desp(desp2);

            cout<<"ref_frame.featresize()"<<ref_frame.feature_count()<<endl;

            //use PnP to verify

            cv::Mat R_new, t_new;

            solvePnP(points3d,pts2,K_mat,Mat(),R_new,t_new,false,CV_ITERATIVE);

            cout<<"R_new = "<<endl<<R_new<<endl;
            cout<<"t_new = "<<endl<<t_new<<endl;

            cout<<"try with pts1"<<endl;            
            solvePnP(points3d,pts1,K_mat,Mat(),R_new,t_new,false,CV_ITERATIVE);

            cout<<"R_new = "<<endl<<R_new<<endl;
            cout<<"t_new = "<<endl<<t_new<<endl;


            is_slam_init = true;

            return 0;

        }
        else
        {

            //use PnP to esitmaate trasform
            cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
            vector<cv::KeyPoint> kp;
            cv::Mat desp;
            orb->detectAndCompute( img2, cv::Mat(), kp, desp);
            cout<<"分别找到了"<<kp.size()<<"个特征点"<<endl;

            if(kp.size()==0)
            {
                return 1;
            }

            cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");

            double knn_match_ratio=0.7;
            vector< vector<cv::DMatch> > matches_knn;
            matcher->knnMatch( ref_frame.features_desp, desp, matches_knn, 2 );
            vector< cv::DMatch > matches;
            for ( size_t i=0; i<matches_knn.size(); i++ )
            {
                if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
                    matches.push_back( matches_knn[i][0] );
            }

            cout<<"matches.size() = "<<matches.size()<<endl;
            if (matches.size() <= 20) //匹配点太少
                return false;
            vector<Point2f> new_pts;
            vector<Point3f> pre_pts_3d = ref_frame.features_3d;
            vector<Point2f> pre_pts_2d = ref_frame.features_2d;
            vector<Point3f> pre_pts_3d_match;
            vector<Point2f> pre_pts_2d_match;
            for ( auto m:matches )
            {
                pre_pts_3d_match.push_back( pre_pts_3d[m.queryIdx]);
                pre_pts_2d_match.push_back( pre_pts_2d[m.queryIdx]);
                new_pts.push_back( kp[m.trainIdx].pt );
            }

            for(int ii = 0 ; ii<pre_pts_2d_match.size();ii++)
            {
                cout<<pre_pts_2d_match[ii]<<" from pre is match to "<<new_pts[ii]<<endl;
            }
            //3d from ref frame are matched with 2d from curr frame
            // then need to calculate R,t with PnP

            cv::Mat R_new, t_new;


            cout<<"if solvePnP with pre_pts_2d_match:"<<endl;
            solvePnP(pre_pts_3d_match,pre_pts_2d_match,K_mat,Mat(),R_new,t_new,false,CV_ITERATIVE);

            cout<<"R_new = "<<endl<<R_new<<endl;
            cout<<"t_new = "<<endl<<t_new<<endl;

            cout<<"if solvePnP with new_pts:"<<endl;
            solvePnP(pre_pts_3d_match,new_pts,K_mat,Mat(),R_new,t_new,false,CV_ITERATIVE);

            cout<<"R_new = "<<endl<<R_new<<endl;
            cout<<"t_new = "<<endl<<t_new<<endl;

            cout<<"ref_frame.features_desp size "<<ref_frame.features_desp.rows<<", "<<ref_frame.features_desp.cols<<endl;




            Mat img2_with_features;
            drawKeypoints(img2,kp, img2_with_features, Scalar::all(-1),DrawMatchesFlags::DEFAULT);
            //imshow("img1_with_features",img1_with_features);
            //imshow("img2_with_features",img2_with_features);
            //waitKey(0);

            //return 0;
            //end of using epipolar constrain
            //imshow("img1", img1);
            //imshow("img2", img2);
            imshow("img1_with_features", img1_with_features);





            //accumulating transformation
            Matrix3d rot_mat;
            Vector3d t_mat;
            cv::cv2eigen(R_new,rot_mat);
            //cout<<"rot_mat = "<<rot_mat<<endl;
            cv::cv2eigen(t_new,t_mat);
            //cout<<"t_mat = "<<t_mat<<endl;
            Isometry3d pose_temp = Isometry3d::Identity();


            //************************************************
            //* the resulted t is normalized, (t(0)^2+t(1)^2+t(2)^2 = 1)
            //* need scale for each frame to recover actual t
            //************************************************

            //double tempNorm = t_mat.squaredNorm();
            //cout<<"tempNorm = "<<tempNorm<<endl;



            if(frameCount!=-160)
            {
                pose_temp.rotate(rot_mat);
                pose_temp.pretranslate(t_mat);
            }
            else
            {
                rot_mat = AngleAxisd(0.25f,Vector3d::UnitY());
                t_mat <<0,0,0;
                cout<<"test rot_mat = "<<rot_mat<<endl;
                cout<<"test t_mat = "<<t_mat<<endl;
                pose_temp.rotate(rot_mat);
                pose_temp.pretranslate(t_mat);
            }



            //pose_global = pose_global*pose_temp;
            pose_global = pose_temp;

            //pose_global.rotate(rot_mat);
            //pose_global.pretranslate(t_mat);



            //pose_global.rotate(pose_temp.rotation());
            //pose_global.pretranslate(pose_temp.translation());

            Vector3d ea = pose_temp.rotation().eulerAngles(0, 1, 2);
            cout<<"pose_temp[R t] = ["<<ea.transpose()<<","<<pose_temp.translation().transpose()<<"]"<<endl;

            ea = pose_global.rotation().eulerAngles(0, 1, 2);
            cout<<"pose_global[R t] = ["<<ea.transpose()<<","<<pose_global.translation().transpose()<<"]"<<endl;
            //draw 2globald trajectory onto mat
            visulizePose2d(traj_image,pose_global);


            cout<<"[frame"<<frameCount<<"]FPS = "<<frameRate<<endl; 

            //if(waitKey(30) >= 0) break;
            waitKey(1);



            img2.copyTo(img1);
            prevFeatures = currFeatures;

            clock_t endFrame = clock();
            deltaTime += endFrame - beginFrame;
            frames ++;
            frameCount++;
            //if you really want FPS
            if( clockToMilliseconds(deltaTime)>1000.0)
            { //every second
                frameRate = (double)frames*0.5 +  frameRate*0.5; //more stable
                frames = 0;
                deltaTime -= CLOCKS_PER_SEC;
            }
        }
    }
    return 0;
}


int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& desp2_out, cv::Mat& img1_with_features)
{
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desp1, desp2;
    orb->detectAndCompute( img1, cv::Mat(), kp1, desp1 );
    orb->detectAndCompute( img2, cv::Mat(), kp2, desp2 );
    //cout<<"分别找到了"<<kp1.size()<<"和"<<kp2.size()<<"个特征点"<<endl;

    if(kp1.size()==0||kp2.size()==0)
    {
        return false;
    }


    drawKeypoints(img2,kp2, img1_with_features, Scalar::all(-1),DrawMatchesFlags::DEFAULT);

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");

    double knn_match_ratio=0.8;
    vector< vector<cv::DMatch> > matches_knn;
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector< cv::DMatch > matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            matches.push_back( matches_knn[i][0] );
    }

    cout<<"matches.size() = "<<matches.size()<<endl;
    if (matches.size() <= 20) //匹配点太少
        return false;

    bool isStart = false;
    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );
        points2.push_back( kp2[m.trainIdx].pt );
        if(!isStart)
        {
            isStart = true;
            desp2_out = desp2.row(m.trainIdx);
        }
        else
        {
            vconcat(desp2_out,desp2.row(m.trainIdx),desp2_out);
        }
    }

    return true;
}



