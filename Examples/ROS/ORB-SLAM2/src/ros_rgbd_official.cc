/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <yolov5_ros_msgs/BoundingBoxes.h>
#include <yolov5_ros_msgs/BoundingBox.h>

#include <opencv2/core/core.hpp>

#include "../../../include/System.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System *pSLAM) : mpSLAM(pSLAM) {}

    void GrabRGBD(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD, const yolov5_ros_msgs::BoundingBoxesConstPtr &box);

    ORB_SLAM2::System *mpSLAM;
};

//时间计算
std::vector<double> vTimesTrack;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if (argc != 3)
    {
        cerr << endl
             << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames. //false即关闭viewer
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<yolov5_ros_msgs::BoundingBoxes> box_sub(nh, "/yolov5/BoundingBoxes", 100);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/yolov5/rgb_image", 100);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image", 100);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, yolov5_ros_msgs::BoundingBoxes> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub, depth_sub, box_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD, &igb, _1, _2, _3));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    sort(vTimesTrack.begin(), vTimesTrack.end());
    double totaltime = 0;
    for (int ni = 0; ni < vTimesTrack.size(); ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    std::cout << "Total tracking time: " << totaltime << " s" << std::endl;
    std::cout << "Average Tracking time: " << totaltime / vTimesTrack.size() << " s" << std::endl;
    std::cout <<"接收到的话题数" <<  vTimesTrack.size() << endl;
    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD, const yolov5_ros_msgs::BoundingBoxesConstPtr &box)
{
    cv::Mat depth;
    std::vector<std::vector<float>> m_boundingboxes;
    for (size_t i = 0; i < box->bounding_boxes.size(); i++)
    {
        std::vector<float> m_boundingbox;
        m_boundingbox.push_back(box->bounding_boxes[i].xmin);
        m_boundingbox.push_back(box->bounding_boxes[i].ymin);
        m_boundingbox.push_back(box->bounding_boxes[i].xmax);
        m_boundingbox.push_back(box->bounding_boxes[i].ymax);
        m_boundingboxes.push_back(m_boundingbox);
    }
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImagePtr cv_ptrD;
    try
    {
        ros::Time time = msgD->header.stamp;
        double time1 = time.toSec();
        //string name = "/home/jnu/桌面/TUM/rgbd_dataset_freiburg3_walking_xyz/depth/" + to_string(time1) + ".png";
        //depth = cv::imread(name, CV_LOAD_IMAGE_UNCHANGED);
        cv_ptrD = cv_bridge::toCvCopy(msgD, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // cv::Mat imRGB = cv_ptrRGB->image;
    // for (size_t i = 0; i < m_boundingboxes.size(); i++)
    // {
    //     cv::rectangle(imRGB,cv::Point(m_boundingboxes[i][0],m_boundingboxes[i][1]),cv::Point(m_boundingboxes[i][2],m_boundingboxes[i][3]),cv::Scalar(255,255,0),2,5);
    // }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    mpSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, cv_ptrRGB->header.stamp.toSec(), m_boundingboxes);
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count();
    cout << "SLAM TrackRGBD all time =" << ttrack * 1000 << endl;
    vTimesTrack.push_back(ttrack);
}