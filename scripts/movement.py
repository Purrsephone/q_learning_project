#!/usr/bin/env python3
import rospy, cv2, cv_bridge
import keras_ocr 
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import moveit_commander
import os
import sys

# TODO
# What if we overshoot?
# Refactoring
# Drop dumbbells clower to blocks
# Maybe extend arm forward further
# Writeup
# Commenting

path_prefix = os.path.dirname(__file__) + "/action_states/"

class Perception(object):
    def __init__(self):
        rospy.init_node("movement")

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
                Image, self.image_callback)
        
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidar_callback)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                Twist, queue_size=1)
        self.initialize = False 
        self.look_for_block = False 
        self.move_toward_block = False 
        self.prediction_groups = []
        self.target_on_screen = False
        self.target_angle = 0
        self.state = "dumbell"
        self.width = 0 
        self.target_distance = 2
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.pipeline = keras_ocr.pipeline.Pipeline()

        self.move_group_arm.go([0,0.7,0,-0.7], wait=True)
        self.move_group_gripper.go([0.01,0.01], wait=True)
        qm_csv = open("qmatrix.csv")
        qmatrix_arr = np.loadtxt(qm_csv, delimiter=",")
        qm_csv.close()
        
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt")
        colors = ["red", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"dumbbell": colors[int(x[0])], "block": int(x[1])},
            self.actions
        ))
        
        # Create a matrix that takes a state, and the current action,
        # and computes the next state
        self.state_action = np.zeros([len(self.action_matrix), len(self.actions)])
        for i in range(len(self.action_matrix)):
            for j in range(len(self.action_matrix[i])):
                  action = int(self.action_matrix[i][j])
                  if action == -1:
                      continue
                  self.state_action[i][action] = j

        self.action_queue = []
        current_state = 0
        for i in range(3):
            options = qmatrix_arr[current_state]
            max_q = 0
            next_action = 0
            for i in range(len(options)):
                if options[i] > max_q:
                    max_q = options[i]
                    next_action = i
            current_state = int(self.state_action[current_state][next_action])
            self.action_queue.append(self.actions[next_action])

        self.start_next_action()

        self.initialize = True 
        self.run()

    def start_next_action(self):
        action = self.action_queue.pop(0)
        self.target = action["dumbbell"]
        self.block_target = str(action["block"])
        
    def set_velocity(self, linear, angular):
        vel_msg = Twist()
        vel_msg.linear.x = linear
        vel_msg.angular.z = angular
        self.cmd_vel_pub.publish(vel_msg)

    def run(self):
        if not self.initialize:
            return 
        r = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.state == "dumbell":
                if self.target_on_screen:
                    if(self.target_distance <= 0.23):
                        # pick up dumbbell
                        self.set_velocity(0, 0)
                        self.raise_arm()
                        self.state = "block"
                        #pick up , change states 
                    else:  
                        # Proportional control to point at dumbbell
                        linear = 0
                        angular = min(0.2, self.target_angle * 0.003)
                        # Move towards dumbbell
                        if abs(self.target_angle) < 5 or (self.target_distance > 1 and abs(self.target_angle) < 15):
                            print("distance", self.target_distance)
                            linear = 0.3 * (self.target_distance - 0.22) 
                        self.set_velocity(linear, angular) 
                else:
                    self.set_velocity(0, 0.4)
                r.sleep()
            elif self.state == "block":
                self.wait_for_image_recognition()
                #now decide what to do w self.pred groups 
                x_avr = 50 
                for el in self.prediction_groups[0]:
                    if el[0] == self.block_target:
                        x_avr = sum(el[1][0:5,0])/4 
                err = self.width/2 - x_avr 
                if (abs(err) < 50):
                    self.state = "move_toward_block"
                    continue 
                k = 0.003
                self.set_velocity(0, err * k)
                rospy.sleep(0.5)
                self.set_velocity(0, 0)
                rospy.sleep(0.5)
            elif self.state == "move_toward_block":
                if self.target_distance <= 0.95:
                    self.set_velocity(0, 0)
                    self.state = "drop_db"
                    continue  
                err = self.target_distance - 0.9
                k = 0.1
                self.set_velocity(err * k, 0)
                r.sleep()
            elif self.state == "drop_db":
                self.lower_arm()
                self.set_velocity(-0.5, 0)
                rospy.sleep(0.5)
                self.set_velocity(0, 0)
                self.state = "dumbell"
                self.start_next_action()
                rospy.sleep(0.5)

    def wait_for_image_recognition(self):
        self.look_for_block = True
        while self.look_for_block:
            rospy.sleep(0.2)

    def raise_arm(self):
        self.move_group_gripper.go([-0.00,-0.00], wait=True)
        rospy.sleep(0.5)
        self.move_group_arm.go([0,-0.7,0,0], wait=True)

    def lower_arm(self):
        self.move_group_arm.go([0,0.7,0,-0.7], wait=True)
        self.move_group_gripper.go([0.01,0.01], wait=True)

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
            elif self.target == "green":
                lower = np.array([40,40,40])
                upper = np.array([70,255,255])
            elif self.target == "red":
                lower = np.array([0, 50, 50])
                upper = np.array([10, 255, 255])

            blue_mask = cv2.inRange(hsv, lower, upper)
            h, w, d = image.shape
            blue_mask[0:h//10, 0:w] = 0
            M = cv2.moments(blue_mask)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                # the center point of the yellow pixels
                cv2.circle(image, (cx, cy), 20, (0,0,255), -1)
                self.target_angle = w/2 - cx
                self.target_on_screen = True
            else:
                self.target_on_screen = False

        elif self.state == "block":
            # Set self.look_for_block to wait for
            # image recognition to run
            if not self.look_for_block:
                return 
            self.prediction_groups = self.pipeline.recognize([image]) 
            print(self.prediction_groups)
       
            self.look_for_block = False 
        
        
if __name__ == '__main__':
    p = Perception()
    rospy.spin()

