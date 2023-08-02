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
#include "ORBextractor.h"

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

using namespace cv;

///////////////////////////////////////////
cv::Mat imGrayPre;
//std::vector<std::vector<float>> box_pre;
std::vector<cv::Point2f> prepoint, nextpoint;
std::vector<cv::Point2f> F_prepoint, F_nextpoint;
std::vector<cv::Point2f> F2_prepoint, F2_nextpoint;
std::vector<uchar> state;
std::vector<float> err;
std::vector<std::vector<cv::KeyPoint>> mvKeysPre;
//////////////////////////////////////////

namespace ORB_SLAM2
{

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame()
    {
    }

    //Copy Constructor
    Frame::Frame(const Frame &frame)
        : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
          mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
          mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
          mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
          mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
          mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
          mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
          mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
          mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
          mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
          mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), keyframe_flag(frame.keyframe_flag), blur_flag(frame.blur_flag)
    {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mpReferenceKF(static_cast<KeyFrame *>(NULL))
    {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        // thread threadLeft(&Frame::ExtractORB,this,0,imLeft);  biscuit
        // thread threadRight(&Frame::ExtractORB,this,1,imRight);
        // threadLeft.join();
        // threadRight.join();

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoMatches();

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imLeft);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    //RGBD
    Frame::Frame(const cv::Mat &img,const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, std::vector<std::vector<float>> &box)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
    {
/*         fstream fid_laplace;
        fstream fid_brenner;
        fstream fid_energy;
        fstream fid_ten;
        fid_laplace.open("/home/jnu/桌面/trajectory/模糊程度laplacian.txt", ios::app);
        fid_brenner.open("/home/jnu/桌面/trajectory/模糊程度Brenner.txt", ios::app);
        fid_energy.open("/home/jnu/桌面/trajectory/模糊程度energy.txt", ios::app);
        fid_ten.open("/home/jnu/桌面/trajectory/模糊程度ten.txt", ios::app); */
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        ////////////////////////////////////////////////////////
        cv::Mat imGrayT = imGray;
        cv::Mat imGray_copy = imGray.clone();
        bool blur = 0;
        keyframe_flag = true;
        vector<int>().swap(blur_flag);
        // Calculate the dynamic abnormal points and output the T matrix
        if (imGrayPre.data)
        {
            ProcessMovingObject(imGray, box);
            double cast1 = 0;
            //double brennerlocal = 0;
            //double tenlocal = 0;
            //double energylocal = 0;
            for (size_t i = 0; i < box.size(); i++)
            {
                Mat image = imGray_copy(Rect(int(box[i][0]), int(box[i][1]), int(box[i][2] - box[i][0]), int(box[i][3] - box[i][1]))).clone();
                cast1 = detect_laplacian(image);   //4.2      walk_rpy丢120帧  rmse：34多
                //brennerlocal = detect_Brenner(image);  //350 walk_rpy丢120帧  rmse：四十多
                //tenlocal = detect_Tenengard(image);   //59    walk_rpy丢100帧  rmse:42  
                //energylocal = detect_Energy(image); //36.5 walk_rpy丢40帧  rmse:32.3     walk_static rmse:76   walk_half有丢帧情况   walk-rpy也有丢帧严重的情况
/*                 fid_brenner << brennerlocal << endl;
                fid_energy << energylocal << endl;
                fid_laplace << cast1 << endl;
                fid_ten << tenlocal << endl;
                fid_brenner.close();
                fid_energy.close();
                fid_laplace.close();
                fid_ten.close(); */
                //cout << "brenner:    " << brennerlocal << "     ten:   " << tenlocal << "         energy: " << energylocal << "cast:   " << cast1 << endl;
                //cout << "X坐标：    "<< int(box[i][0]) << endl;
                //cout << "ten:   " << tenwhole << "         energy: " << energywhole << "cast:   " << cast2<< endl;
                if (cast1 < 4.2)
                {
                    /*                 if ((brennerwhole - brennerlocal) > 200&&brennerlocal < 300) //模糊参数
                { */
                    blur_flag.push_back(1);
                   //cout << "laplacian low:  " << cast1 <<endl;
                   //cv::circle(imGray, Point2f(((box[i][2] - box[i][0]) / 2 + box[i][0]), (box[i][3] - box[i][1]) / 2 + box[i][1]), 5, cv::Scalar(255, 0, 0), 2);////！！！！这一句会导致程序崩溃，是坐标超出图像范围了？
                }
                else
                {
                    //cout << "laplacian high:  " << cast1 <<endl;
                    blur_flag.push_back(0);
                }
            }
            std::swap(imGrayPre, imGrayT);
        }
        else
        {
            blur_flag.push_back(0);
            blur_flag.push_back(0);
            std::swap(imGrayPre, imGrayT);
        }
        
     ///biscuit test depth
        // ORB extraction
        ExtractORB(0, img, imGray, imDepth, box);

        N = mvKeys.size();
        //cout << "最终特征点的个数"  << N << endl;
        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoFromRGBD(imDepth);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
    {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        //ExtractORB(0,imGray); biscuit

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        // Set no stereo information
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    /**
 * Epipolar constraints and output the T matrix.
 * Save outliers to T_M
 */
    void Frame::ProcessMovingObject(const cv::Mat &imgray, std::vector<std::vector<float>> &box)
    {

        double sum_total = 0;
        double count_total = 0;
        double sum = 0;
        double count = 0;
        // Clear the previous data
        //cv::Mat image;
        //image = imGrayPre;   //上一帧的灰度图
        F_prepoint.clear();
        F_nextpoint.clear();
        F2_prepoint.clear();
        F2_nextpoint.clear();
        T_M.clear();

/*         cv::Mat image_color = imgray.clone();
        cv::cvtColor(image_color, image_color, CV_GRAY2RGB); */

        // Detect dynamic target and ultimately optput the T matrix
        //biscuit

        cv::goodFeaturesToTrack(imGrayPre, prepoint, 1000, 0.01, 8, cv::Mat(), 3, true, 0.04);
        cv::cornerSubPix(imGrayPre, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
        cv::calcOpticalFlowPyrLK(imGrayPre, imgray, prepoint, nextpoint, state, err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
        for (int i = 0; i < state.size(); i++)
        {
            if (state[i] != 0)
            {
                int dx[10] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
                int dy[10] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
                int x1 = prepoint[i].x, y1 = prepoint[i].y;
                int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
                if ((x1 < limit_edge_corner || x1 >= imgray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= imgray.cols - limit_edge_corner || y1 < limit_edge_corner || y1 >= imgray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= imgray.rows - limit_edge_corner))
                {
                    state[i] = 0;
                    continue;
                }
                double sum_check = 0;
                for (int j = 0; j < 9; j++)
                    sum_check += abs(imGrayPre.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgray.at<uchar>(y2 + dy[j], x2 + dx[j]));
                if (sum_check > limit_of_check)
                    state[i] = 0;
                if (state[i])
                {
                    // if (*mohu_mask.ptr<uchar>(int(nextpoint[i].y), int(nextpoint[i].x)) != 0)
                    //  {  如果这里只用检测框外的光流点做运动一致性容易导致系统崩溃，推测是光流点不足。
                    F_prepoint.push_back(prepoint[i]);
                    F_nextpoint.push_back(nextpoint[i]);
                    // }
                }
                //cv::circle(image,prepoint[i],5,cv::Scalar(255,0,0),2);
            }
        }
        //cv::imshow("特征点",image);
        cv::Mat ima = imgray.clone();
        //cvtColor(ima,ima,CV_GRAY2BGR);
        // F-Matrix
        cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
        cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);
        // cout << "nextpoint.size " << nextpoint.size() << endl;
        for (int i = 0; i < prepoint.size(); i++)
        {
            if (state[i] != 0)
            {
                double A = F.at<double>(0, 0) * prepoint[i].x + F.at<double>(0, 1) * prepoint[i].y + F.at<double>(0, 2);
                double B = F.at<double>(1, 0) * prepoint[i].x + F.at<double>(1, 1) * prepoint[i].y + F.at<double>(1, 2);
                double C = F.at<double>(2, 0) * prepoint[i].x + F.at<double>(2, 1) * prepoint[i].y + F.at<double>(2, 2);
                double dd = fabs(A * nextpoint[i].x + B * nextpoint[i].y + C) / sqrt(A * A + B * B);
                if (dd <= 1) //默认值：1.0  极线约束参数
                    continue;
                T_M.push_back(nextpoint[i]);
                //cv::circle(imgray,nextpoint[i],5,cv::Scalar(0,0,200),2);
            }
        }
        //imshow("运动一致性",image_color);
        //waitKey(20);
        //cout << "总：   " << (sum_total / (count_total + 1))<<" 检测框"<< (sum / (count + 1)) << endl;
        // imshow("动态点",ima);
        // cv::waitKey(3);
        //cout << "动态点" << T_M.size() << endl;

    }

   
    void Frame::AssignFeaturesToGrid()
    {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++)
        {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractORB(int flag, const cv::Mat &img, const cv::Mat &im, const cv::Mat &imD, std::vector<std::vector<float>> &box)
    {
        if (flag == 0)
            (*mpORBextractorLeft)(im, cv::Mat(), img, imD, mvKeys, mDescriptors, box, T_M, mask_frame, blur_flag);
        else
            (*mpORBextractorRight)(im, cv::Mat(),img, imD, mvKeysRight, mDescriptorsRight, box, T_M, mask_frame, blur_flag);
    }

    /*
void Frame::ExtractORB(int flag, const cv::Mat &im,const cv::Mat &imD,std::vector<std::vector<float>>& box)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),imD,mvKeys,mDescriptors,box,T_M);
    else
        (*mpORBextractorRight)(im,cv::Mat(),imD,mvKeysRight,mDescriptorsRight,box,T_M);
}
*/

    void Frame::SetPose(cv::Mat Tcw)
    {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices()
    {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
    {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
    {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels)
                    {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void Frame::ComputeBoW()
    {
        if (mBowVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints()
    {
        if (mDistCoef.at<float>(0) == 0.0)
        {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++)
        {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++)
        {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft)
    {
        if (mDistCoef.at<float>(0) != 0.0)
        {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
        }
        else
        {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void Frame::ComputeStereoMatches()
    {
        mvuRight = vector<float>(N, -1.0f);
        mvDepth = vector<float>(N, -1.0f);

        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

        //Assign keypoints to row table
        vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

        for (int i = 0; i < nRows; i++)
            vRowIndices[i].reserve(200);

        const int Nr = mvKeysRight.size();

        for (int iR = 0; iR < Nr; iR++)
        {
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);

            for (int yi = minr; yi <= maxr; yi++)
                vRowIndices[yi].push_back(iR);
        }

        // Set limits for search
        const float minZ = mb;
        const float minD = 0;
        const float maxD = mbf / minZ;

        // For each left keypoint search a match in the right image
        vector<pair<int, int>> vDistIdx;
        vDistIdx.reserve(N);

        for (int iL = 0; iL < N; iL++)
        {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const vector<size_t> &vCandidates = vRowIndices[vL];

            if (vCandidates.empty())
                continue;

            const float minU = uL - maxD;
            const float maxU = uL - minD;

            if (maxU < 0)
                continue;

            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = mDescriptors.row(iL);

            // Compare descriptor to right keypoints
            for (size_t iC = 0; iC < vCandidates.size(); iC++)
            {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];

                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;

                const float &uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU)
                {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }

            // Subpixel match by correlation
            if (bestDist < thOrbDist)
            {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);

                // sliding window search
                const int w = 5;
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
                IL.convertTo(IL, CV_32F);
                IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);

                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                    continue;

                for (int incR = -L; incR <= +L; incR++)
                {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    IR.convertTo(IR, CV_32F);
                    IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestincR = incR;
                    }

                    vDists[L + incR] = dist;
                }

                if (bestincR == -L || bestincR == L)
                    continue;

                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];

                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

                if (deltaR < -1 || deltaR > 1)
                    continue;

                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

                float disparity = (uL - bestuR);

                if (disparity >= minD && disparity < maxD)
                {
                    if (disparity <= 0)
                    {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                }
            }
        }

        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--)
        {
            if (vDistIdx[i].first < thDist)
                break;
            else
            {
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
    {

        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++)
        {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0)
            {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i)
    {
        const float z = mvDepth[i];
        if (z > 0)
        {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        }
        else
            return cv::Mat();
    }

    // 图像锐化
    Mat Frame::imgSharpen(const Mat &img, char *arith) //arith为3*3模板算子
    {
        int rows = img.rows;                  //原图的行
        int cols = img.cols * img.channels(); //原图的列
        int offsetx = img.channels();         //像素点的偏移量

        Mat dst = Mat::ones(img.rows - 2, img.cols - 2, img.type());

        for (int i = 1; i < rows - 1; i++)
        {
            const uchar *previous = img.ptr<uchar>(i - 1);
            const uchar *current = img.ptr<uchar>(i);
            const uchar *next = img.ptr<uchar>(i + 1);
            uchar *output = dst.ptr<uchar>(i - 1);
            for (int j = offsetx; j < cols - offsetx; j++)
            {
                output[j - offsetx] =
                    saturate_cast<uchar>(previous[j - offsetx] * arith[0] + previous[j] * arith[1] + previous[j + offsetx] * arith[2] +
                                         current[j - offsetx] * arith[3] + current[j] * arith[4] + current[j + offsetx] * arith[5] +
                                         next[j - offsetx] * arith[6] + next[j] * arith[7] + next[j - offsetx] * arith[8]);
            }
        }
        return dst;
    }

    Mat Frame::ruihua(string name, const Mat img, const std::vector<float> &box)
    {
        Mat out_image = img.clone();
        Mat image = img(Rect(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))).clone();
        //imwrite("test_image/模糊.jpg",image);
        //cout << image.size() << endl;

        char arith[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; //使用拉普拉斯算子

        Mat dst1 = imgSharpen(image, arith);
        //imwrite("test_image/锐化.jpg",dst1);
        //cout << dst1.size() << endl;
        dst1.copyTo(out_image(Rect(int(box[0]), int(box[1]), int(box[2] - box[0]) - 2, int(box[3] - box[1]) - 2)));
        //cout << out_image.size() << endl;
        //namedWindow(name, WINDOW_NORMAL);
        //imshow(name, dst1);
        return out_image;
    }

    double Frame::detect_laplacian(cv::Mat image)
    {
        cv::Mat gray_image, lap_image;
        gray_image = image.clone();
        cv::Laplacian(gray_image, lap_image, CV_16U);
        lap_image = cv::abs(lap_image);
        double cast = cv::mean(lap_image)[0];
        return cast;
    }

    double Frame::detect_Brenner(cv::Mat image)
    {
        /**         
* Brenner梯度方法 
*   
* Inputs:   
* @param image:  
* Return: double    
*/

        assert(image.empty());
        cv::Mat gray_img;
        gray_img = image.clone();
        //cout << gray_img.channels() << endl;
        double result = 0;
        for (int i = 0; i < gray_img.rows; ++i)
        {
            uchar *data = gray_img.ptr<uchar>(i);
            for (int j = 0; j < gray_img.cols - 2; ++j)
            {
                result += pow(data[j + 2] - data[j], 2);
            }
        }
        return result / gray_img.total();
    }

    double Frame::detect_Tenengard(cv::Mat image)
    {

        assert(image.empty());

        cv::Mat gray_img, sobel_x, sobel_y, G;
        gray_img = image.clone();

        //分别计算x/y方向梯度
        cv::Sobel(gray_img, sobel_x, CV_32FC1, 1, 0);
        cv::Sobel(gray_img, sobel_y, CV_32FC1, 0, 1);
        cv::multiply(sobel_x, sobel_x, sobel_x);
        cv::multiply(sobel_y, sobel_y, sobel_y);
        cv::Mat sqrt_mat = sobel_x + sobel_y;
        cv::sqrt(sqrt_mat, G);

        return cv::mean(G)[0];
    }

    double Frame::detect_Energy(cv::Mat image)
    {

        assert(image.empty());

        cv::Mat gray_img, smd_image_x, smd_image_y, G;
        gray_img = image.clone();
        cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
        kernel_x.at<float>(1, 2) = -1.0;
        kernel_x.at<float>(1, 1) = 1.0;
        cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
        kernel_y.at<float>(1, 1) = 1.0;
        kernel_y.at<float>(2, 1) = -1.0;
        cv::filter2D(gray_img, smd_image_x, gray_img.depth(), kernel_x);
        cv::filter2D(gray_img, smd_image_y, gray_img.depth(), kernel_y);

        cv::multiply(smd_image_x, smd_image_x, smd_image_x);
        cv::multiply(smd_image_y, smd_image_y, smd_image_y);
        G = smd_image_x + smd_image_y;

        return cv::mean(G)[0];
    }


} //namespace ORB_SLAM
