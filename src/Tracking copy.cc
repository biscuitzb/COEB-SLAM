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

#include<unistd.h>
#include "Tracking.h"
#include "ORBextractor.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{

///////////////////////////////dynaslam修改
    vAllPixels = cv::Mat(640*480,2,CV_32F);
    int m(0);
    for (int i(0); i < 640; i++){
        for (int j(0); j < 480; j++){
            vAllPixels.at<float>(m,0) = i;
            vAllPixels.at<float>(m,1) = j;
            m++;
        }
    }
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp,std::vector<std::vector<float>>& box)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;
    cv::Mat imDOut;
    cv::Mat _imRGB = imRGB; 
    cv::Mat imMask;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

////////////////////////////////////////////构建语义分割的mask
    cv::Mat __mask = cv::Mat::ones(480,640,CV_8U);
    for (size_t i = 0; i < box.size(); i++)
    {
        __mask(cv::Rect(int(box[i][0]),int(box[i][1]),int(box[i][2]-box[i][0]),int(box[i][3]-box[i][1]))).setTo(0);
    }
    mCurrentFrame = Frame(mImGray, imDepth, __mask, _imRGB, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, box);
    ///////////////////////////////////  inpainting
    cv::Mat imRGBOut = imRGB;

    //mCurrentFrame = Frame(mImGray, imDepth, imMask, imRGBOut, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth,box);
    // string name = "/home/jnu/wzb/ORB_SLAM2/test_image/1.jpg";
    // cv::imwrite(name,mCurrentFrame.mImMask);
    Track();
    if (!mCurrentFrame.mTcw.empty())
    {
            //标记
        //cout<<"test:    " << mDB.mNumElem   << endl;
        if (mDB.mNumElem > 0)
        {
            //cout << (int)mDB.mvDataBase[0].mImMask.at<uchar>(0,0) << endl;
            InpaintFrames(mCurrentFrame, mImGray, imDepth, imRGBOut, __mask);
        }
    }

    if (mCurrentFrame.mIsKeyFrame)
    {
        //cout << "insert test" << endl;
        mDB.InsertFrame2DB(mCurrentFrame);
        string name = "/home/jnu/wzb/ORB_SLAM2/test_image/" + to_string(rand()) + ".jpg";
        cv::imwrite(name, imRGBOut);
    }
    imDOut = imDepth;
    imDepth.convertTo(imDOut, CV_16U, 1. / mDepthMapFactor);
    cv::Mat imaskOut = __mask;
    // cv::imshow("test",imRGBOut);
    // cv::waitKey(3);
    //string name = "/home/jnu/wzb/ORB_SLAM2/test_image/"+to_string(rand())+".jpg";
    // //cv::imshow("bounding box",image_bounding_box);
    // //cv::waitKey(3);
    //cv::imwrite(name,imRGBOut);
    // string name2 = "/home/jnu/wzb/ORB_SLAM2/test_image/2.jpg";
    // cv::imwrite(name2,imRGBOut);
    ////////////////////////////
    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

 void Tracking::InpaintFrames(const ORB_SLAM2::Frame &currentFrame,
                             cv::Mat &imGray, cv::Mat &imDepth,
                             cv::Mat &imRGB, cv::Mat &mask){
    FillRGBD(currentFrame,mask,imGray,imDepth,imRGB);
} 

void Tracking::FillRGBD(const ORB_SLAM2::Frame &currentFrame,cv::Mat &mask,cv::Mat &imGray,cv::Mat &imDepth,cv::Mat &imRGB){

    //cout << "test" << endl;
    cv::Mat imGrayAccumulator = imGray.mul(mask);
    imGrayAccumulator.convertTo(imGrayAccumulator,CV_32F);
    cv::Mat bgr[3];
    cv::split(imRGB,bgr);
    cv::Mat imRAccumulator = bgr[2].mul(mask);
    imRAccumulator.convertTo(imRAccumulator,CV_32F);
    cv::Mat imGAccumulator = bgr[1].mul(mask);
    imGAccumulator.convertTo(imGAccumulator,CV_32F);
    cv::Mat imBAccumulator = bgr[0].mul(mask);
    imBAccumulator.convertTo(imBAccumulator,CV_32F);
    cv::Mat imCounter;
    mask.convertTo(imCounter,CV_32F);
    cv::Mat imDepthAccumulator = imDepth.mul(imCounter);
    imDepthAccumulator.convertTo(imDepthAccumulator,CV_32F);
    cv::Mat imMinDepth = cv::Mat::zeros(imDepth.size(),CV_32F)+100.0;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    cv::Mat K1 = cv::Mat::eye(3,3,CV_8U);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    for (int i(0); i < mDB.mNumElem; i++){

        ORB_SLAM2::Frame ref = mDB.mvDataBase[i];
        ORB_SLAM2::Frame refFrame = mDB.mvDataBase[i];
        cv::Mat m_imd = mDB.mvDataBase[i].mImDepth;
        cv::Mat m_imm = mDB.mvDataBase[i].mImMask;
        cv::Mat m_imRGB = mDB.mvDataBase[i].mImRGB;
        // cout << "size:  " << mDB.mvDataBase[i].mImDepth.size() << endl;
        // cout << "type:  " << mDB.mvDataBase[i].mImDepth.type() << endl;
        // cout << "chan:  " << mDB.mvDataBase[i].mImDepth.channels() << endl;
        // cout << "size:  " << m_imd.size() << endl;
        // cout << "type:  " << m_imd.type() << endl;
        // cout << "chan:  " << m_imd.channels() << endl;

        cv::Mat bgr[3];
        cv::split(m_imRGB,bgr);
        cv::Mat imR = bgr[2];
        cv::Mat imG = bgr[1];
        cv::Mat imB = bgr[0];

        cv::Mat vPixels(640*480,2,CV_32F);
        cv::Mat mDepth(640*480,1,CV_32F);

        int n(0);
        for (int j(0); j < 640 * 480; j++)
        {
            int x = (int)vAllPixels.at<float>(j, 0);
            int y = (int)vAllPixels.at<float>(j, 1);
            if ((int)m_imm.at<uchar>(y, x) == 1)
            {
                // cout << "test90" << endl;
                //const float d = imDepth.at<float>(y, x);
                // cout << "testtt     " << y <<"   " << x <<"     " <<  endl;
                //const float ds = imDepth.at<float>(y, x);
                //cout << "t" << endl;
                const float d = m_imd.at<float>(y,x);   //这里不关y,x的事
                //const float d = 1;
                //cout << "t" << endl;
                if (d > 0)
                {
                    vPixels.at<float>(n, 0) = vAllPixels.at<float>(j, 0);
                    vPixels.at<float>(n, 1) = vAllPixels.at<float>(j, 1);
                    mDepth.at<float>(n, 0) = 1. / d;
                    n++;
                }
            }
        }
        //cout << "test1" << endl;
        vPixels = vPixels.rowRange(0,n);
        mDepth = mDepth.rowRange(0,n);
        hconcat(vPixels,cv::Mat::ones(n,1,CV_32F),vPixels);
        cv::Mat vMPRefFrame = K.inv() * vPixels.t();
        vconcat(vMPRefFrame,mDepth.t(),vMPRefFrame);

        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vMPw;

        // Divide by last column
        for (int j(0); j < vMPCurrentFrame.cols; j++)
        {
            vMPCurrentFrame.at<float>(0,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(1,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(2,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(3,j) /= vMPCurrentFrame.at<float>(3,j);
        }

        cv::Mat matProjDepth = vMPCurrentFrame.row(2);
        cv::Mat aux;
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;

        cv::Mat vProjPixels(matCurrentFrame.cols,2,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.cols,1,CV_32F);
        cv::Mat _vPixels(matCurrentFrame.cols,2,CV_32F);

        int p(0);
        for (int j(0); j < matCurrentFrame.cols; j++)
        {
            float x = matCurrentFrame.at<float>(0,j)/matCurrentFrame.at<float>(2,j);
            float y = matCurrentFrame.at<float>(1,j)/matCurrentFrame.at<float>(2,j);
            bool inFrame = (x > 1 && x < (currentFrame.mImDepth.cols - 1) && y > 1 && y < (currentFrame.mImDepth.rows - 1));
            if (inFrame && (mask.at<uchar>(y,x) == 0))
            {
                //cout << "test2" << endl;
                vProjPixels.at<float>(p,0) = x;
                vProjPixels.at<float>(p,1) = y;
                _matProjDepth.at<float>(p,0) = matProjDepth.at<float>(0,j);
                _vPixels.at<float>(p,0) = vPixels.at<float>(j,0);
                _vPixels.at<float>(p,1) = vPixels.at<float>(j,1);
                p++;
            }
        }
        vProjPixels = vProjPixels.rowRange(0,p);
        matProjDepth = _matProjDepth.rowRange(0,p);
        vPixels = _vPixels.rowRange(0,p);

        for (int j(0); j< p; j++)
        {

            //cout << "test3" << endl;
            int _x = (int)vPixels.at<float>(j, 0);
            int _y = (int)vPixels.at<float>(j,1);
            float x = vProjPixels.at<float>(j,0);//x of *
            float y = vProjPixels.at<float>(j,1);//y of *
            /*
                -----------
                | A  | B  |
                ----*------ y
                | C  | D  |
                -----------
                     x
            */
            float x_a = floor(x);
            float y_a = floor(y);
            float x_b = ceil(x);
            float y_b = floor(y);
            float x_c = floor(x);
            float y_c = ceil(y);
            float x_d = ceil(x);
            float y_d = ceil(y);

            float weight = 0;

            if( IsInImage(x_a,y_a,imGrayAccumulator)){

                if(abs(imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                { 
                    //cout << "test211" << endl;
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a)) += weight;
                    //cout << "test21" << endl;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    //cout << "test31" << endl;
                    imRAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                else if ((imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a)) = weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);  //这里出问题
                    imRAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                //cout << "test213" << endl;
                imMinDepth.at<float>(y_a,x_a) = min(imMinDepth.at<float>(y_a,x_a),matProjDepth.at<float>(j,0));
            }
            //cout << "test214" << endl;
            if( IsInImage(x_b,y_b,imGrayAccumulator) && (x_a != x_b))
            {
                if(abs(imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) += weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                else if ((imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) = weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                imMinDepth.at<float>(y_b,x_b) = min(imMinDepth.at<float>(y_b,x_b),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_c,y_c,imGrayAccumulator) && (y_a != y_c) && (x_b != x_c && y_b != y_c))
            {
                if(abs(imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) += weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                else if ((imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) = weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                imMinDepth.at<float>(y_c,x_c) = min(imMinDepth.at<float>(y_c,x_c),matProjDepth.at<float>(j,0));

            }
            if( IsInImage(x_d,y_d,imGrayAccumulator) && (x_a != x_d && y_a != y_d) && (y_b != y_d) && (x_d != x_c))
            {
                if (abs(imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) += weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                else if ((imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) = weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)mDB.mvDataBase[i].mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                imMinDepth.at<float>(y_d,x_d) = min(imMinDepth.at<float>(y_d,x_d),matProjDepth.at<float>(j,0));
            }
        }
    }

    imGrayAccumulator = imGrayAccumulator.mul(1/imCounter);
    imRAccumulator = imRAccumulator.mul(1/imCounter);
    imRAccumulator.convertTo(imRAccumulator,CV_8U);
    cv::Mat imR = cv::Mat::zeros(imRAccumulator.size(),imRAccumulator.type());
    imRAccumulator.copyTo(imR,mask);
    imGAccumulator = imGAccumulator.mul(1/imCounter);
    imGAccumulator.convertTo(imGAccumulator,CV_8U);
    cv::Mat imG = cv::Mat::zeros(imGAccumulator.size(),imGAccumulator.type());
    imGAccumulator.copyTo(imG,mask);
    imBAccumulator = imBAccumulator.mul(1/imCounter);
    imBAccumulator.convertTo(imBAccumulator,CV_8U);
    cv::Mat imB = cv::Mat::zeros(imBAccumulator.size(),imBAccumulator.type());
    imBAccumulator.copyTo(imB,mask);
    imDepthAccumulator = imDepthAccumulator.mul(1/imCounter);

    std::vector<cv::Mat> arrayToMerge;
    arrayToMerge.push_back(imB);
    arrayToMerge.push_back(imG);
    arrayToMerge.push_back(imR);
    cv::merge(arrayToMerge, imRGB);

    imGrayAccumulator.convertTo(imGrayAccumulator,CV_8U);
    imGray = imGray*0;
    imGrayAccumulator.copyTo(imGray,mask);
    imDepth = imDepth*0;
    imDepthAccumulator.copyTo(imDepth,mask);

}



// void Tracking::GeometricModelCorrection(const ORB_SLAM2::Frame &currentFrame,
//                                         cv::Mat &imDepth, cv::Mat &mask){
//     if(currentFrame.mTcw.empty()){
//         std::cout << "Geometry not working." << std::endl;
//     }
//     else if (mDB.mNumElem >= ELEM_INITIAL_MAP){
//         vector<ORB_SLAM2::Frame> vRefFrames = GetRefFrames(currentFrame);
//         vector<DynKeyPoint> vDynPoints = ExtractDynPoints(vRefFrames,currentFrame);
//         mask = DepthRegionGrowing(vDynPoints,imDepth);
//         CombineMasks(currentFrame,mask);
//     }
// }


// vector<ORB_SLAM2::Frame> Tracking::GetRefFrames(const ORB_SLAM2::Frame &currentFrame){

//     cv::Mat rot1 = currentFrame.mTcw.rowRange(0,3).colRange(0,3);
//     cv::Mat eul1 = rotm2euler(rot1);
//     cv::Mat trans1 = currentFrame.mTcw.rowRange(0,3).col(3);
//     cv::Mat vDist;
//     cv::Mat vRot;

//     for (int i(0); i < mDB.mNumElem; i++){
//         cv::Mat rot2 = mDB.mvDataBase[i].mTcw.rowRange(0,3).colRange(0,3);
//         cv::Mat eul2 = rotm2euler(rot2);
//         double distRot = cv::norm(eul2,eul1,cv::NORM_L2);
//         vRot.push_back(distRot);

//         cv::Mat trans2 = mDB.mvDataBase[i].mTcw.rowRange(0,3).col(3);
//         double dist = cv::norm(trans2,trans1,cv::NORM_L2);
//         vDist.push_back(dist);
//     }

//     double minvDist, maxvDist;
//     cv::minMaxLoc(vDist, &minvDist, &maxvDist);
//     vDist /= maxvDist;

//     double minvRot, maxvRot;
//     cv::minMaxLoc(vRot, &minvRot, &maxvRot);
//     vRot /= maxvRot;

//     vDist = 0.7*vDist + 0.3*vRot;
//     cv::Mat vIndex;
//     cv::sortIdx(vDist,vIndex,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

//     mnRefFrames = std::min(MAX_REF_FRAMES,vDist.rows);

//     vector<ORB_SLAM2::Frame> vRefFrames;

//     for (int i(0); i < mnRefFrames; i++)
//     {
//         int ind = vIndex.at<int>(0,i);
//         vRefFrames.push_back(mDB.mvDataBase[ind]);
//     }

//     return(vRefFrames);
// }

// vector<Geometry::DynKeyPoint> Tracking::ExtractDynPoints(const vector<ORB_SLAM2::Frame> &vRefFrames,
//                                                          const ORB_SLAM2::Frame &currentFrame){
//     cv::Mat K = cv::Mat::eye(3,3,CV_32F);
//     K.at<float>(0,0) = currentFrame.fx;
//     K.at<float>(1,1) = currentFrame.fy;
//     K.at<float>(0,2) = currentFrame.cx;
//     K.at<float>(1,2) = currentFrame.cy;

//     cv::Mat vAllMPw;
//     cv::Mat vAllMatRefFrame;
//     cv::Mat vAllLabels;
//     cv::Mat vAllDepthRefFrame;

//     for (int i(0); i < mnRefFrames; i++)
//     {
//         ORB_SLAM2::Frame refFrame = vRefFrames[i];

//         // Fill matrix with points
//         cv::Mat matRefFrame(refFrame.N,3,CV_32F);
//         cv::Mat matDepthRefFrame(refFrame.N,1,CV_32F);
//         cv::Mat matInvDepthRefFrame(refFrame.N,1,CV_32F);
//         cv::Mat vLabels(refFrame.N,1,CV_32F);
//         int k(0);
//         for(int j(0); j < refFrame.N; j++){
//             const cv::KeyPoint &kp = refFrame.mvKeys[j];
//             const float &v = kp.pt.y;
//             const float &u = kp.pt.x;
//             const float d = refFrame.mImDepth.at<float>(v,u);
//             if (d > 0 && d < 6){
//                 matRefFrame.at<float>(k,0) = refFrame.mvKeysUn[j].pt.x;
//                 matRefFrame.at<float>(k,1) = refFrame.mvKeysUn[j].pt.y;
//                 matRefFrame.at<float>(k,2) = 1.;
//                 matInvDepthRefFrame.at<float>(k,0) = 1./d;
//                 matDepthRefFrame.at<float>(k,0) = d;
//                 vLabels.at<float>(k,0) = i;
//                 k++;
//             }
//         }

//         matRefFrame = matRefFrame.rowRange(0,k);
//         matInvDepthRefFrame = matInvDepthRefFrame.rowRange(0,k);
//         matDepthRefFrame = matDepthRefFrame.rowRange(0,k);
//         vLabels = vLabels.rowRange(0,k);
//         cv::Mat vMPRefFrame = K.inv()*matRefFrame.t();
//         cv::vconcat(vMPRefFrame,matInvDepthRefFrame.t(),vMPRefFrame);
//         cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;
//         cv::Mat _vMPw = cv::Mat(4,vMPw.cols,CV_32F);
//         cv::Mat _vLabels = cv::Mat(vLabels.rows,1,CV_32F);
//         cv::Mat _matRefFrame = cv::Mat(matRefFrame.rows,3,CV_32F);
//         cv::Mat _matDepthRefFrame = cv::Mat(matDepthRefFrame.rows,1,CV_32F);

//         int h(0);
//         mParallaxThreshold = 30;
//         for (int j(0); j < k; j++)
//         {
//             cv::Mat mp = cv::Mat(3,1,CV_32F);
//             mp.at<float>(0,0) = vMPw.at<float>(0,j)/matInvDepthRefFrame.at<float>(0,j);
//             mp.at<float>(1,0) = vMPw.at<float>(1,j)/matInvDepthRefFrame.at<float>(0,j);
//             mp.at<float>(2,0) = vMPw.at<float>(2,j)/matInvDepthRefFrame.at<float>(0,j);
//             cv::Mat tRefFrame = refFrame.mTcw.rowRange(0,3).col(3);

//             cv::Mat tCurrentFrame = currentFrame.mTcw.rowRange(0,3).col(3);
//             cv::Mat nMPRefFrame = mp - tRefFrame;
//             cv::Mat nMPCurrentFrame = mp - tCurrentFrame;

//             double dotProduct = nMPRefFrame.dot(nMPCurrentFrame);
//             double normMPRefFrame = cv::norm(nMPRefFrame,cv::NORM_L2);
//             double normMPCurrentFrame = cv::norm(nMPCurrentFrame,cv::NORM_L2);
//             double angle = acos(dotProduct/(normMPRefFrame*normMPCurrentFrame))*180/M_PI;
//             if (angle < mParallaxThreshold)
//             {
//                 _vMPw.at<float>(0,h) = vMPw.at<float>(0,j);
//                 _vMPw.at<float>(1,h) = vMPw.at<float>(1,j);
//                 _vMPw.at<float>(2,h) = vMPw.at<float>(2,j);
//                 _vMPw.at<float>(3,h) = vMPw.at<float>(3,j);
//                 _vLabels.at<float>(h,0) = vLabels.at<float>(j,0);
//                 _matRefFrame.at<float>(h,0) = matRefFrame.at<float>(j,0);
//                 _matRefFrame.at<float>(h,1) = matRefFrame.at<float>(j,1);
//                 _matRefFrame.at<float>(h,2) = matRefFrame.at<float>(j,2);
//                 _matDepthRefFrame.at<float>(h,0) = matDepthRefFrame.at<float>(j,0);
//                 h++;
//             }
//         }

//         vMPw = _vMPw.colRange(0,h);
//         vLabels = _vLabels.rowRange(0,h);
//         matRefFrame = _matRefFrame.rowRange(0,h);
//         matDepthRefFrame = _matDepthRefFrame.rowRange(0,h);

//         if (vAllMPw.empty())
//         {
//             vAllMPw = vMPw;
//             vAllMatRefFrame = matRefFrame;
//             vAllLabels = vLabels;
//             vAllDepthRefFrame = matDepthRefFrame;
//         }
//         else
//         {
//             if (!vMPw.empty())
//             {
//                 hconcat(vAllMPw,vMPw,vAllMPw);
//                 vconcat(vAllMatRefFrame,matRefFrame,vAllMatRefFrame);
//                 vconcat(vAllLabels,vLabels,vAllLabels);
//                 vconcat(vAllDepthRefFrame,matDepthRefFrame,vAllDepthRefFrame);
//             }
//         }
//     }

//     cv::Mat vLabels = vAllLabels;

//     if (!vAllMPw.empty())
//     {
//         cv::Mat vMPCurrentFrame = currentFrame.mTcw * vAllMPw;

//         // Divide by last column
//         for (int i(0); i < vMPCurrentFrame.cols; i++)
//         {
//             vMPCurrentFrame.at<float>(0,i) /= vMPCurrentFrame.at<float>(3,i);
//             vMPCurrentFrame.at<float>(1,i) /= vMPCurrentFrame.at<float>(3,i);
//             vMPCurrentFrame.at<float>(2,i) /= vMPCurrentFrame.at<float>(3,i);
//             vMPCurrentFrame.at<float>(3,i) /= vMPCurrentFrame.at<float>(3,i);
//         }
//         cv::Mat matProjDepth = vMPCurrentFrame.row(2);

//         cv::Mat _vMPCurrentFrame = cv::Mat(vMPCurrentFrame.size(),CV_32F);
//         cv::Mat _vAllMatRefFrame = cv::Mat(vAllMatRefFrame.size(),CV_32F);
//         cv::Mat _vLabels = cv::Mat(vLabels.size(),CV_32F);
//         cv::Mat __vAllDepthRefFrame = cv::Mat(vAllDepthRefFrame.size(),CV_32F);
//         int h(0);
//         cv::Mat __matProjDepth = cv::Mat(matProjDepth.size(),CV_32F);
//         for (int i(0); i < matProjDepth.cols; i++)
//         {
//             if (matProjDepth.at<float>(0,i) < 7)
//             {
//                 __matProjDepth.at<float>(0,h) = matProjDepth.at<float>(0,i);

//                 _vMPCurrentFrame.at<float>(0,h) = vMPCurrentFrame.at<float>(0,i);
//                 _vMPCurrentFrame.at<float>(1,h) = vMPCurrentFrame.at<float>(1,i);
//                 _vMPCurrentFrame.at<float>(2,h) = vMPCurrentFrame.at<float>(2,i);
//                 _vMPCurrentFrame.at<float>(3,h) = vMPCurrentFrame.at<float>(3,i);

//                 _vAllMatRefFrame.at<float>(h,0) = vAllMatRefFrame.at<float>(i,0);
//                 _vAllMatRefFrame.at<float>(h,1) = vAllMatRefFrame.at<float>(i,1);
//                 _vAllMatRefFrame.at<float>(h,2) = vAllMatRefFrame.at<float>(i,2);

//                 _vLabels.at<float>(h,0) = vLabels.at<float>(i,0);

//                 __vAllDepthRefFrame.at<float>(h,0) = vAllDepthRefFrame.at<float>(i,0);

//                 h++;
//             }
//         }

//         matProjDepth = __matProjDepth.colRange(0,h);
//         vMPCurrentFrame = _vMPCurrentFrame.colRange(0,h);
//         vAllMatRefFrame = _vAllMatRefFrame.rowRange(0,h);
//         vLabels = _vLabels.rowRange(0,h);
//         vAllDepthRefFrame = __vAllDepthRefFrame.rowRange(0,h);

//         cv::Mat aux;
//         cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
//         cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;

//         cv::Mat mat2CurrentFrame(matCurrentFrame.cols,2,CV_32F);
//         cv::Mat v2AllMatRefFrame(matCurrentFrame.cols,3,CV_32F);
//         cv::Mat mat2ProjDepth(matCurrentFrame.cols,1,CV_32F);
//         cv::Mat v2Labels(matCurrentFrame.cols,1,CV_32F);
//         cv::Mat _vAllDepthRefFrame(matCurrentFrame.cols,1,CV_32F);

//         int j = 0;
//         for (int i(0); i < matCurrentFrame.cols; i++)
//         {
//             float x = ceil(matCurrentFrame.at<float>(0,i)/matCurrentFrame.at<float>(2,i));
//             float y = ceil(matCurrentFrame.at<float>(1,i)/matCurrentFrame.at<float>(2,i));
//             if (IsInFrame(x,y,currentFrame))
//             {
//                 const float d = currentFrame.mImDepth.at<float>(y,x);
//                 if (d > 0)
//                 {
//                     mat2CurrentFrame.at<float>(j,0) = x;
//                     mat2CurrentFrame.at<float>(j,1) = y;
//                     v2AllMatRefFrame.at<float>(j,0) = vAllMatRefFrame.at<float>(i,0);
//                     v2AllMatRefFrame.at<float>(j,1) = vAllMatRefFrame.at<float>(i,1);
//                     v2AllMatRefFrame.at<float>(j,2) = vAllMatRefFrame.at<float>(i,2);
//                     _vAllDepthRefFrame.at<float>(j,0) = vAllDepthRefFrame.at<float>(i,0);
//                     float d1 = matProjDepth.at<float>(0,i);
//                     mat2ProjDepth.at<float>(j,0) = d1;
//                     v2Labels.at<float>(j,0) = vLabels.at<float>(i,0);
//                     j++;
//                 }
//             }
//         }
//         vAllDepthRefFrame = _vAllDepthRefFrame.rowRange(0,j);
//         vAllMatRefFrame = v2AllMatRefFrame.rowRange(0,j);
//         matProjDepth = mat2ProjDepth.rowRange(0,j);
//         matCurrentFrame = mat2CurrentFrame.rowRange(0,j);
//         vLabels = v2Labels.rowRange(0,j);

//         cv::Mat u1((2*mDmax+1)*(2*mDmax+1),2,CV_32F);
//         int m(0);
//         for (int i(-mDmax); i <= mDmax; i++){
//             for (int j(-mDmax); j <= mDmax; j++){
//                 u1.at<float>(m,0) = i;
//                 u1.at<float>(m,1) = j;
//                 m++;
//             }
//         }

//         cv::Mat matDepthCurrentFrame(matCurrentFrame.rows,1,CV_32F);
//         cv::Mat _matProjDepth(matCurrentFrame.rows,1,CV_32F);
//         cv::Mat _matCurrentFrame(matCurrentFrame.rows,2,CV_32F);

//         int _s(0);
//         for (int i(0); i < matCurrentFrame.rows; i++)
//         {
//             int s(0);
//             cv::Mat _matDiffDepth(u1.rows,1,CV_32F);
//             cv::Mat _matDepth(u1.rows,1,CV_32F);
//             for (int j(0); j < u1.rows; j++)
//             {
//                 int x = (int)matCurrentFrame.at<float>(i,0) + (int)u1.at<float>(j,0);
//                 int y = (int)matCurrentFrame.at<float>(i,1) + (int)u1.at<float>(j,1);
//                 float _d = currentFrame.mImDepth.at<float>(y,x);
//                 if ((_d > 0) && (_d < matProjDepth.at<float>(i,0)))
//                 {
//                     _matDepth.at<float>(s,0) = _d;
//                     _matDiffDepth.at<float>(s,0) = matProjDepth.at<float>(i,0) - _d;
//                     s++;
//                 }
//             }
//             if (s > 0)
//             {
//                 _matDepth = _matDepth.rowRange(0,s);
//                 _matDiffDepth = _matDiffDepth.rowRange(0,s);
//                 double minVal, maxVal;
//                 cv::Point minIdx, maxIdx;
//                 cv::minMaxLoc(_matDiffDepth,&minVal,&maxVal,&minIdx,&maxIdx);
//                 int xIndex = minIdx.x;
//                 int yIndex = minIdx.y;
//                 matDepthCurrentFrame.at<float>(_s,0) = _matDepth.at<float>(yIndex,0);
//                 _matProjDepth.at<float>(_s,0) = matProjDepth.at<float>(i,0);
//                 _matCurrentFrame.at<float>(_s,0) = matCurrentFrame.at<float>(i,0);
//                 _matCurrentFrame.at<float>(_s,1) = matCurrentFrame.at<float>(i,1);
//                 _s++;
//             }
//         }

//         matDepthCurrentFrame = matDepthCurrentFrame.rowRange(0,_s);
//         matProjDepth = _matProjDepth.rowRange(0,_s);
//         matCurrentFrame = _matCurrentFrame.rowRange(0,_s);

//         mDepthThreshold = 0.6;

//         cv::Mat matDepthDifference = matProjDepth - matDepthCurrentFrame;

//         mVarThreshold = 0.001; //0.040;

//         vector<Geometry::DynKeyPoint> vDynPoints;

//         for (int i(0); i < matCurrentFrame.rows; i++)
//         {
//             if (matDepthDifference.at<float>(i,0) > mDepthThreshold)
//             {
//                 int xIni = (int)matCurrentFrame.at<float>(i,0) - mDmax;
//                 int yIni = (int)matCurrentFrame.at<float>(i,1) - mDmax;
//                 int xEnd = (int)matCurrentFrame.at<float>(i,0) + mDmax + 1;
//                 int yEnd = (int)matCurrentFrame.at<float>(i,1) + mDmax + 1;
//                 cv::Mat patch = currentFrame.mImDepth.rowRange(yIni,yEnd).colRange(xIni,xEnd);
//                 cv::Mat mean, stddev;
//                 cv::meanStdDev(patch,mean,stddev);
//                 double _stddev = stddev.at<double>(0,0);
//                 double var = _stddev*_stddev;
//                 if (var < mVarThreshold)
//                 {
//                     DynKeyPoint dynPoint;
//                     dynPoint.mPoint.x = matCurrentFrame.at<float>(i,0);
//                     dynPoint.mPoint.y = matCurrentFrame.at<float>(i,1);
//                     dynPoint.mRefFrameLabel = vLabels.at<float>(i,0);
//                     vDynPoints.push_back(dynPoint);
//                 }
//             }
//         }

//         return vDynPoints;
//     }
//     else
//     {
//         vector<Geometry::DynKeyPoint> vDynPoints;
//         return vDynPoints;
//     }
// }


// cv::Mat Tracking::DepthRegionGrowing(const vector<DynKeyPoint> &vDynPoints,const cv::Mat &imDepth){

//     cv::Mat maskG = cv::Mat::zeros(480,640,CV_32F);

//     if (!vDynPoints.empty())
//     {
//         mSegThreshold = 0.20;

//         for (size_t i(0); i < vDynPoints.size(); i++){
//             int xSeed = vDynPoints[i].mPoint.x;
//             int ySeed = vDynPoints[i].mPoint.y;
//             const float d = imDepth.at<float>(ySeed,xSeed);
//             if (maskG.at<float>(ySeed,xSeed)!=1. && d > 0)
//             {
//                 cv::Mat J = RegionGrowing(imDepth,xSeed,ySeed,mSegThreshold);
//                 maskG = maskG | J;
//             }
//         }

//         int dilation_size = 15;
//         cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
//                                                cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
//                                                cv::Point( dilation_size, dilation_size ) );
//         maskG.cv::Mat::convertTo(maskG,CV_8U);
//         cv::dilate(maskG, maskG, kernel);
//     }
//     else
//     {
//         maskG.cv::Mat::convertTo(maskG,CV_8U);
//     }

//     cv::Mat _maskG = cv::Mat::ones(480,640,CV_8U);
//     maskG = _maskG - maskG;

//     return maskG;
// }


// void Tracking::CombineMasks(const ORB_SLAM2::Frame &currentFrame, cv::Mat &mask)
// {
//     cv::Mat _maskL = cv::Mat::ones(currentFrame.mImMask.size(),currentFrame.mImMask.type());
//     _maskL = _maskL - currentFrame.mImMask;

//     cv::Mat _maskG = cv::Mat::ones(mask.size(),mask.type());
//     _maskG = _maskG - mask;

//     cv::Mat _mask = _maskL | _maskG;

//     cv::Mat __mask = cv::Mat::ones(_mask.size(),_mask.type());
//     __mask = __mask - _mask;
//     mask = __mask;

// }

bool Tracking::IsInImage(const float &x, const float &y, const cv::Mat image)
{
    return (x >= 0 && x < (image.cols) && y >= 0 && y < image.rows);
}

float Tracking::Area(float x1, float x2, float y1, float y2){
    float xc1 = max(x1-0.5,x2-0.5);
    float xc2 = min(x1+0.5,x2+0.5);
    float yc1 = max(y1-0.5,y2-0.5);
    float yc2 = min(y1+0.5,y2+0.5);
    return (xc2-xc1)*(yc2-yc1);
}

bool Tracking::DataBase::IsFull(){
    return (mIni == (mFin+1) % MAX_DB_SIZE);
}

void Tracking::DataBase::InsertFrame2DB(const ORB_SLAM2::Frame &currentFrame){

    if (!IsFull()){
        mvDataBase[mFin] = currentFrame;
        mFin = (mFin + 1) % MAX_DB_SIZE;
        mNumElem += 1;
    }
    else {
        mvDataBase[mIni] = currentFrame;
        mFin = mIni;
        mIni = (mIni + 1) % MAX_DB_SIZE;
    }
}

} //namespace ORB_SLAM
