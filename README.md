# COEB-SLAM
COEB-SLAM is a robust visual SLAM system designed for dynamic environments, primarily addressing issues such as motion blur and frame loss.

# 1.  Getting Started
- Install ORB-SLAM2 prerequisites: C++11 or C++0x Compiler, Pangolin, OpenCV and Eigen3 (https://github.com/raulmur/ORB_SLAM2).
- Please refer to the official tutorial for the installation environment required to deploy Yolov5.(https://github.com/ultralytics/yolov5)

- Clone this repo:
```bash
git clone https://github.com/biscuitzb/COEB-SLAM.git
cd COEB-SLAM
```
```
cd COEB-SLAM
chmod +x build.sh
./build.sh
```

## RGB-D Example on TUM Dataset
- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it. Please download the bag file.

- Associate RGB images and depth images executing the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools):

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```
These associations files are given in the folder `./Examples/RGB-D/associations/` for the TUM dynamic sequences.

  ```
  roslaunch yolov5.launch
  rosrun ORB_SLAM2 RGBD Vocabulary/ORBvoc.txt Examples/RGB-D/tum_bag.yaml
  rosbag play TUM_DATASET.bag
  ```

  
## Acknowledgements
Our code builds on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).


