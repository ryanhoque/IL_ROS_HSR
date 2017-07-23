#!/usr/bin/python
# -*- coding: utf-8 -*-

import hsrb_interface
import rospy
import sys
import math
import tf
import tf2_ros
import tf2_geometry_msgs
import IPython
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped

from tmc_suction.msg import (
    SuctionControlAction,
    SuctionControlGoal
)

import actionlib

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM


__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0

class VGripper(object):

    def __init__(self,graspPlanner,cam):
        #topic_name = '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        suction_action = '/hsrb/suction_control'

        self.suction_control_client = actionlib.SimpleActionClient(
            suction_action, SuctionControlAction)
     
       
        try:
            if not self.suction_control_client.wait_for_server(
                rospy.Duration(_CONNECTION_TIMEOUT)):
                raise Exception(_suction_action + 'does not exist')
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)



        not_read = True
        while not_read:

            try:
                cam_info = cam.read_info_data()
                if(not cam_info == None):
                    not_read = False
            except:
                rospy.logerr('info not recieved')
       

        self.pcm = PCM()
        self.pcm.fromCameraInfo(cam_info)
        self.br = tf.TransformBroadcaster()
        self.gp = graspPlanner


    def broadcast_poses(self,poses):
        #while True: 
        
        count = 0

        
        for pose in poses:
            
            num_pose = pose[1]
            label = pose[0]

            

            td_points = self.pcm.projectPixelTo3dRay((num_pose[0],num_pose[1]))
            pose = np.array([td_points[0],td_points[1],0.001*num_pose[2]])
            

            self.br.sendTransform((td_points[0], td_points[1], pose[2]-0.013),
                    (0.0, 0.0, 0.0, 1.0),
                    rospy.Time.now(),
                    'card',
                    'head_rgbd_sensor_rgb_frame')
            count += 1


    def find_to_pick_region(self,results,c_img,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        poses = []
        #IPython.embed()
        
        p_list = []
        for result in results['objects']:
            print result

            x_min = float(result['box_index'][0])
            y_min = float(result['box_index'][1])
            x_max = float(result['box_index'][2])
            y_max = float(result['box_index'][3])

            x = (x_max-x_min)/2.0 + x_min
            y = (y_max - y_min)/2.0 + y_min

            cv2.imshow('debug',c_img[int(y_min):int(y_max),int(x_min):int(x_max)])

            cv2.waitKey(30)
            
            #Crop D+img
            d_img_c = d_img[int(y_min):int(y_max),int(x_min):int(x_max)]

            depth = self.gp.find_max_depth(d_img_c)
            poses.append([result['num_class_label'],[x,y,depth]])

        self.broadcast_poses(poses)








if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()