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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

/*     static bool CompareSmall(std::pair<int, ExtractorNode *> &p1, std::pair<int, ExtractorNode *> &p2)
    {
        return (p1.first < p2.first);
    }    //bug fix */

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.

///////////////////////////////////////////////////////
    void CheckMovingKeyPoints(const cv::Mat &imGray, std::vector<cv::KeyPoint> &mvKeysT, std::vector<cv::Point2f> T, int level,const cv::Mat mask);
    void CheckMovingKeyPoints_finall(const cv::Mat &imGray, std::vector<std::vector<cv::KeyPoint>> &mvKeysT, std::vector<cv::Point2f> T, const cv::Mat mask);
////////////////////////////////////////////

    void operator()( cv::InputArray image, cv::InputArray mask,const cv::Mat &imD,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors,std::vector<std::vector<float>>& box,std::vector<cv::Point2f> T_M);

    void operator()(cv::InputArray image, cv::InputArray mask, const cv::Mat &img,const cv::Mat &imD,
                    std::vector<cv::KeyPoint> &keypoints,
                    cv::OutputArray descriptors, std::vector<std::vector<float>> &box, std::vector<cv::Point2f> T_M,cv::Mat &mask_result,std::vector<int> blur_flag);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

   // int ID_Image = 1;


protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, std::vector<std::vector<cv::KeyPoint> >& allKeypoints_copy, std::vector<cv::Point2f> T_M,bool flag_area,const cv::Mat mask);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

