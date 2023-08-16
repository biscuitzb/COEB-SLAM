#!/usr/bin/env python
# -*- coding: utf-8 -*-


#这个版本已实现窗口筛除模糊帧


import time

from tkinter import image_names
from tracemalloc import start
from turtle import end_fill
import cv2
from scipy import rand
import torch
import rospy
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
import message_filters
from std_msgs.msg import Float64MultiArray


class Yolo_Dect:
    def __init__(self):
        # start = 0
        # end = 0
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        rgb_image_topic = rospy.get_param(
            '~rgb_image_topic', '')
        depth_image_topic = rospy.get_param(
            '~depth_image_topic', '')
        pub_topic = rospy.get_param('~pub_topic', '')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False
        self.color_list = []
        self.depth_list = []
        self.box_list = []
        self.time_list = []
        self.last_frame = Image()
        self.last_frame = cv2.imread("none")
        self.value_list = []
        self.mark = 0
        self.feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Load class color
        self.classes_colors = {}

        # image subscribe
        #print("test1\n")
        self.rgb_sub = message_filters.Subscriber(rgb_image_topic,Image)
        self.depth_sub = message_filters.Subscriber(depth_image_topic,Image)
        self.image = message_filters.ApproximateTimeSynchronizer([self.rgb_sub,self.depth_sub],10,1)
        self.image.registerCallback(self.image_callback)
        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)
        #self.detect_pub = rospy.Publisher(
            # '/yolov5/detection_image',  Image, queue_size=1)
        self.rgb_pub = rospy.Publisher(
            '/yolov5/rgb_image',  Image, queue_size=1)           
        self.depth_pub = rospy.Publisher(
            '/yolov5/depth_image',  Image, queue_size=1)
        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image,image2):
        global start
        start = time.time()
        #print(num)
        header = image.header.stamp.to_sec()
        #print(type(self.bridge.imgmsg_to_cv2(image,"bgr8")))
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        depth_image = np.frombuffer(image2.data, dtype=np.uint8).reshape(
            image2.height, image2.width, -1)
        #depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        depth_image = image2
        value = self.LK(self.last_frame,self.color_image)
        if ( value > 500):
            self.mark = 100 
        self.last_frame = self.color_image
        results = self.model(self.color_image)
        # xmin    ymin    xmtax   ymax  confidence  class    name
        boxs = results.pandas().xyxy[0].values
#添加修改
        if self.mark < 50:
            self.dectshow(self.color_image, boxs, image.height, image.width,depth_image,header)
        if self.mark >50:
            self.value_list.append(value)
            self.color_list.append(self.color_image)
            self.depth_list.append(depth_image)
            self.box_list.append(boxs)
            self.time_list.append(header)
        if (len(self.color_list) >3) & (self.mark > 50):
            self.mark = 0
            number = self.value_list.index(max(self.value_list))
            self.color_list.pop(number)
            self.depth_list.pop(number)
            self.time_list.pop(number)
            self.box_list.pop(number)
            #self.value_list.pop(number)
            for i,(color,boxes,depth,time_header) in enumerate(zip(self.color_list,self.box_list,self.depth_list,self.time_list)):
                self.dectshow(color, boxes, image.height, image.width,depth,time_header)
            self.color_list.clear()
            self.depth_list.clear()
            self.time_list.clear()
            self.box_list.clear()
            self.value_list.clear()
        #cv2.waitKey(3)
    def dectshow(self, org_img, boxs, height, width,depth_image,header):
        #img = org_img.copy()
        rgb_image = org_img.copy()
        count = 0
        for i in boxs:
            count += 1
        #img = np.zeros(img.shape,np.uint8)
        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            boundingBox.Class = box[-1]

            if(boundingBox.Class == "person"):
                self.boundingBoxes.bounding_boxes.append(boundingBox)

        self.publish_image( height, width,rgb_image,depth_image,header)

    def publish_image(self,  height, width,rgb_image,depth_image,Header_stamp):
 
        rgb_image_temp = Image()

        header = Header(stamp=rospy.Time(Header_stamp))
        header.frame_id = self.camera_frame


        rgb_image_temp.height = height
        rgb_image_temp.width = width
        rgb_image_temp.encoding = 'rgb8'
        rgb_image_temp.data = np.array(rgb_image).tobytes()
        rgb_image_temp.header = header
        rgb_image_temp.step = width * 3

        #self.detect_pub.publish(image_temp)
        
        # self.position_pub.publish(self.boundingBoxes)
        self.position_pub.publish(self.boundingBoxes)
        self.rgb_pub.publish(rgb_image_temp)        
        self.depth_pub.publish(depth_image)
        global end
        end = time.time()
        print("窗口开 检测时间：  ",end - start)
        #cv2.imwrite("/home/jnu/wzb/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/1.jpg",self.bridge.imgmsg_to_cv2(depth_image,"mono16"))

    def LK(self,image_pre,image_new):
        if image_pre is None :
            return 0
# ShiTomasi 角点检测参数


# lucas kanade光流法参数


# 创建随机颜色
        #color = np.random.randint(0, 255, (100, 3))

# 获取第一帧，找到角点
        #old_frame = cv2.imread(name1)
# 找到原始灰度图
        old_gray = cv2.cvtColor(image_pre, cv2.COLOR_BGR2GRAY)

# 获取图像中的角点，返回到p0中
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

# 创建一个蒙版用来画轨迹
        #mask = np.zeros_like(image_pre)

        #frame = cv2.imread(name2)
        frame_gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)

# 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
# 选取好的跟踪点
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        sum = 0.00
        count = 0.00

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a = int(a)
            b = int(b)
            if((((a - c) * (a - c)) < 1000) & (((d - b) * (d - b))<1000)):
                sum = (a - c) * (a - c) + (d - b) * (d - b)+sum
                count += 1
            #image_new = cv2.circle(image_new, (a, b), 5, (255,0,0), -1)
        #print(sum/(count+1))
        return sum/(count+1)


def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":
    num = 0 

    main()