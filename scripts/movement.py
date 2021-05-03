#!/usr/bin/env python3
import rospy, cv2, cv_bridge
import keras_ocr 
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import moveit_commander

#pipeline = keras_ocr.pipeline.Pipeline()
#need to add images for recognition 
#images = [img1, img2, ...]
#prediction_groups = pipline.recognize(images)

class Perception(object):
    def __init__(self):
        rospy.init_node("movement")

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # initalize the debugging window
        # cv2.namedWindow("window", 1)

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
                Image, self.image_callback)
        
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidar_callback)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                Twist, queue_size=1)

        self.twist = Twist()
        self.target = "red"
        self.target_on_screen = False
        self.target_angle = 0
       
        self.target_distance = 2
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.move_group_arm.go([0,0.7,0,-0.7], wait=True)
        self.move_group_gripper.go([0.01,0.01], wait=True)
        self.run()

    def run(self):
        r = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.target_on_screen:
                vel_msg = Twist()
                vel_msg.angular.z = min(0.2, self.target_angle * 0.003)
                  
                if abs(self.target_angle) < 5:
                    print("distance", self.target_distance)
                    vel_msg.linear.x = 0.3 * (self.target_distance - 0.22) 
                else:
                    print("not moving linearly", self.target_angle)
  
                self.cmd_vel_pub.publish(vel_msg) 
            else:
                vel_msg = Twist()
                vel_msg.angular.z = 0.4
                self.cmd_vel_pub.publish(vel_msg)
            r.sleep()
                

    def lidar_callback(self, data):
        front_angles = [355 + x for x in range(5)] + [x for x in range(5)]
        distances = []
        for angle in front_angles:
            if data.ranges[angle] < data.range_max:
                 distances.append(data.ranges[angle])
        if len(distances) == 0:
            self.target_distance = data.range_max
        else:
            self.target_distance = sum(distances)/len(distances)
        

    def image_callback(self, msg):
        # converts the incoming ROS message to cv2 format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #need to define green, blue, red 

        if self.target == "blue":
            lower = np.array([ 110, 50, 50])
            upper = np.array([130, 255, 255])
            #maskblue = cv2.inRange(hsv, lower_blue, upper_blue)
        elif self.target == "green":
            lower = np.array([40,40,40])
            upper = np.array([70,255,255])
        elif self.target == "red":
            lower = np.array([0, 50, 50])
            upper = np.array([10, 255, 255])

        #gotta do something with this lmao

        blue_mask = cv2.inRange(hsv, lower, upper)
        h, w, d = image.shape
        search_top = int(h)
        search_bot = int(0)
        #blue_mask[0:search_top, 0:w] = 0
        M = cv2.moments(blue_mask)
        #blue_mask[search_bot:h, 0:w] = 0        
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # the center point of the yellow pixels
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)
            self.target_angle = w/2 - cx
            self.target_on_screen = True
            #vel_msg = Twist()
            #vel_msg.angular.z = error_term * 0.001
            #self.cmd_vel_pub.publish(vel_msg)
            #if abs(error_term) < 10:
            # 
        else:
            self.pointed_at_target = False
            self.target_on_screen = False
        
if __name__ == '__main__':
    p = Perception()
    rospy.spin()

