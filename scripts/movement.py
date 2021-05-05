#!/usr/bin/env python3
import rospy, cv2, cv_bridge
import keras_ocr 
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import moveit_commander


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
        self.initialize = False 
        self.twist = Twist()
        self.block_target = "2" 
        self.look_for_block = False 
        self.move_toward_block = False 
        self.prediction_groups = []
        self.target = "blue"
        self.target_on_screen = False
        self.target_angle = 0
        self.state = "dumbell"
        self.width = 0 
        self.target_distance = 2
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.move_group_arm.go([0,0.7,0,-0.7], wait=True)
        self.move_group_gripper.go([0.01,0.01], wait=True)
        self.pipeline = keras_ocr.pipeline.Pipeline()
        self.initialize = True 
        self.run()
        

    def run(self):
        if not self.initialize:
            return 
        r = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.state == "dumbell":
                if self.target_on_screen:
                    vel_msg = Twist()
                    vel_msg.angular.z = min(0.2, self.target_angle * 0.003)
                    
                    if abs(self.target_angle) < 5:
                        print("distance", self.target_distance)
                        vel_msg.linear.x = 0.3 * (self.target_distance - 0.22) 
                    #else:
                        #print("not moving linearly", self.target_angle)
                    if(self.target_distance <= 0.23):
                        vel_msg.linear.x = 0 
                        vel_msg.angular.z = 0
                        self.cmd_vel_pub.publish(vel_msg) 
                        self.move_group_gripper.go([-0.00,-0.00], wait=True)
                        rospy.sleep(0.5)
                        self.move_group_arm.go([0,-0.7,0,-0.7], wait=True)
                        self.state = "block"
                        #pick up , change states 
                    else:  
                        self.cmd_vel_pub.publish(vel_msg) 
                else:
                    vel_msg = Twist()
                    vel_msg.angular.z = 0.4
                    self.cmd_vel_pub.publish(vel_msg)
                r.sleep()
            elif self.state == "block":
                self.look_for_block = True 
                x_avr = 50 
                while self.look_for_block:
                    r.sleep()
                #now decide what to do w self.pred groups 
                for el in self.prediction_groups[0]:
                    if el[0] == self.block_target:
                        x_avr = sum(el[1][0:5,0])/4 
                err = self.width/2 - x_avr 
                print("ERROR:")
                print(err)
                if (abs(err) < 50):
                    self.state = "move_toward_block"
                    continue 
                k = 0.003
                vel_msg = Twist()
                vel_msg.angular.z = err * k
                self.cmd_vel_pub.publish(vel_msg)
                rospy.sleep(0.5)
                vel_msg.angular.z = 0
                self.cmd_vel_pub.publish(vel_msg)
                rospy.sleep(0.5)
            elif self.state == "move_toward_block":
                if self.target_distance <= 0.95:
                    vel_msg = Twist()
                    vel_msg.linear.x = 0
                    self.cmd_vel_pub.publish(vel_msg)
                    self.state = "drop_db"
                    continue  
                err = self.target_distance - 0.9
                print("TARG DIST")
                print(self.target_distance)
                k = 0.1
                vel_msg = Twist()
                vel_msg.linear.x = err * k
                print("VEL:")
                print(vel_msg.linear.x)
                self.cmd_vel_pub.publish(vel_msg)
                r.sleep()
            elif self.state == "drop_db":
                self.move_group_arm.go([0,0.7,0,-0.7], wait=True)
                self.move_group_gripper.go([0.01,0.01], wait=True)
                vel_msg = Twist()
                vel_msg.linear.x = -0.5
                self.cmd_vel_pub.publish(vel_msg)
                rospy.sleep(0.5)
                vel_msg.linear.x = 0
                self.cmd_vel_pub.publish(vel_msg)
                self.state = "dumbell"
                self.block_target = "3"
                self.target = "red"



    def lidar_callback(self, data):
        if not self.initialize:
            return 
        front_angles = [355 + x for x in range(5)] + [x for x in range(5)]
        distances = []
        for angle in front_angles:
            if data.ranges[angle] < data.range_max:
                 distances.append(data.ranges[angle])
        if len(distances) == 0:
            self.target_distance = data.range_max
        else:
            self.target_distance = sum(distances)/len(distances)

    #code for blocks 
    # #pipeline = keras_ocr.pipeline.Pipeline()
    #need to add images for recognition 
    #images = [img1, img2, ...]
    #prediction_groups = pipline.recognize(images) 
     #could we define black?? 

    def image_callback(self, msg):
        if not self.initialize:
            return 
        # converts the incoming ROS message to cv2 format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, w, d = image.shape
        self.width = w 


        if self.state == "dumbell":

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

        elif self.state == "block":
            if not self.look_for_block:
                return 
            #stuff 
            #we could make more targets like 1,2,3
            self.prediction_groups = self.pipeline.recognize([image]) 
            print(self.prediction_groups)
       
            self.look_for_block = False 
                    #then stop looking and start moving 
            
            #otherwise, we need to turn 
        
        
if __name__ == '__main__':
    p = Perception()
    rospy.spin()

