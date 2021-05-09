#!/usr/bin/env python3
import rospy, cv2, cv_bridge
import keras_ocr 
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import moveit_commander
import os
import sys

path_prefix = os.path.dirname(__file__) + "/action_states/"

class WeightLifter(object):
    def __init__(self):
        rospy.init_node("movement")

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()
           
        # wait for set up 
        rospy.sleep(1)
        
        # set to false before set up is complete 
        self.initialized = False 

        # Set up subscribers and publishers
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
                Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidar_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                Twist, queue_size=1)

        # Current state
        self.state = "dumbbell"

        # Locating dumbbell
        self.target_db_visible = False
        self.target_db_angle = 0


        # Moving towards a block
        self.look_for_block = False 
        self.move_toward_block = False 
        self.prediction_groups = [] # Prediction groups for OCR
        self.move_to_block_counter = 0

        # OCR
        self.pipeline = keras_ocr.pipeline.Pipeline()

        # Width of the camera
        self.camera_width = 0

        # Average distance to object in front of 
        # robot (via LIDAR) 
        self.front_distance = 2

        # Arm movement
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.lower_arm()

        # Load the QMatrix from training  
        with open("qmatrix.csv") as qm_csv:
            qmatrix_arr = np.loadtxt(qm_csv, delimiter=",")
        
        # Load the action matrix
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
        # iterates through 2D array 
        for i in range(len(self.action_matrix)):
            for j in range(len(self.action_matrix[i])):
                  action = int(self.action_matrix[i][j])
                  # skip if action not possible from state 
                  if action == -1:
                      continue
                  self.state_action[i][action] = j

        # Using the Q Matrix, fill up a queue of actions
        # to perform
        self.action_queue = []
        current_state = 0
        # iterate through 3 step action sequence 
        for i in range(3):
            options = qmatrix_arr[current_state]
            max_q = 0
            next_action = 0
            # iterate through action options based on current state
            for i in range(len(options)):
                # choose action with max q value 
                if options[i] > max_q:
                    max_q = options[i]
                    next_action = i
            # update current state and actions list 
            current_state = int(self.state_action[current_state][next_action])
            self.action_queue.append(self.actions[next_action])
        
        # Load the first action from the queue
        self.start_next_action()

        # Start moving
        self.initialized = True 
        self.run()

    def start_next_action(self):
        """
        Load the next action from the queue
        """
        # checks if action is complete 
        if len(self.action_queue) == 0:
            rospy.loginfo("I did it !")
            sys.exit(0)
        action = self.action_queue.pop(0)
        self.db_target = action["dumbbell"]
        self.block_target = str(action["block"])
        
    def set_velocity(self, linear, angular):
        """
        Set the velocity of the robot
        """
        vel_msg = Twist()
        vel_msg.linear.x = linear
        vel_msg.angular.z = angular
        self.cmd_vel_pub.publish(vel_msg)

    def run(self):
        """
        Execute the logic for the current state
        """
        r = rospy.Rate(2)
        # run continuously 
        while not rospy.is_shutdown():
            if self.state == "dumbbell":
                # In the dumbbell state, the robot will look for
                # and move to the target dumbbell. Once close enough,
                # it will pick it up and transition to the `point_toward_block` state
                # checks if dumbbell is in sight  
                if self.target_db_visible:
                    # checks if close enough to dumbbell 
                    # we found 0.23 to be a good distance 
                    if(self.front_distance <= 0.23):
                        # pick up dumbbell
                        self.set_velocity(0, 0)
                        self.raise_arm()
                        self.state = "point_toward_block"
                    # else, not close enough, keep moving toward dumbbell 
                    else:  
                        # Proportional control to point at dumbbell
                        linear = 0
                        angular = min(0.2, self.target_db_angle * 0.003)
                        # Move towards dumbbell. When the robot is further away,
                        # it can be more lenient with how close it is pointed, so 
                        # we allow the angle to be 15 pixels. When it is closer,
                        # it needs to be more accurate in order to pick up the
                        # dumbbell, so we lower it to 5 pixels.
                        if abs(self.target_db_angle) < 5 or (self.front_distance > 1 and abs(self.target_db_angle) < 15):
                            linear = 0.3 * (self.front_distance - 0.22) 
                        self.set_velocity(linear, angular) 
                else:
                    # If the target dumbbell is not visible, just turn until the robot finds it
                    self.set_velocity(0, 0.4)
                r.sleep()
            elif self.state == "point_toward_block":
                # In the point_toward_block state, the robot uses image recognition
                # and proportional control to point itself towards the target block.
                self.wait_for_image_recognition()

                # Find the x positions of each number that 
                # matches our target, and average all of them.
                x_positions = []
                # iterate through labeled predictions 
                for el in self.prediction_groups[0]:
                    if el[0] == self.block_target:
                        x_positions.append(sum(el[1][0:5,0])/4)
                if len(x_positions) == 0:
                    # If no target was found, assume it is on the
                    # left side of the screen so the robot will
                    # just turn left until it finds the number.
                    x_avr = 50
                else:
                    # average the x positions if a target was found 
                    x_avr = sum(x_positions)/len(x_positions)
                err = self.camera_width/2 - x_avr 
                if (abs(err) < 50):
                    # If pointing close enough to the block,
                    # start moving forwards
                    self.state = "move_toward_block"
                    continue 
                # Turn a bit, then stop and rerun image recognition
                self.set_velocity(0, err * 0.003)
                rospy.sleep(0.5)
                self.set_velocity(0, 0)
                rospy.sleep(0.5)
            elif self.state == "move_toward_block":
                # In the move_toward_block state, the robot will
                # use LIDAR to drive up to the block.

                # Every 8 iterations of this state, switch back to 
                # point_toward_block state to make sure the robot
                # is still headed in the right direction.
                self.move_to_block_counter += 1
                if self.move_to_block_counter >= 8 and self.front_distance > 2:
                    self.move_to_block_counter = 0
                    self.set_velocity(0, 0)
                    self.state = "point_toward_block"
                    continue

                # If close enough to the block, transition 
                # to the drop_db state
                if self.front_distance <= 0.7:
                    self.set_velocity(0, 0)
                    self.state = "drop_db"
                    continue  

                # Proportional control to move towards the block
                err = self.front_distance - 0.6
                self.set_velocity(err * 0.1, 0)
                r.sleep()
            elif self.state == "drop_db":
                # In the drop_db state, the robot will lower its arm,
                # release the dumbbell, and then move on to the next action.
                self.lower_arm()
                self.set_velocity(-0.5, 0)
                rospy.sleep(0.5)
                self.set_velocity(0, 0)
                self.state = "dumbbell"
                self.start_next_action()
                rospy.sleep(0.5)

    def wait_for_image_recognition(self):
        """
        This will loop until the image recognition has run
        """
        self.look_for_block = True
        #repeat until we have found block 
        while self.look_for_block:
            rospy.sleep(0.2)

    def raise_arm(self):
        """
        Close the gripper and raise the arm
        """
        self.move_group_gripper.go([-0.00,-0.00], wait=True)
        rospy.sleep(0.5)
        self.move_group_arm.go([0,-0.7,0,0], wait=True)

    def lower_arm(self):
        """
        Lower the arm and release the gripper
        """
        self.move_group_arm.go([0,0.7,0,-0.7], wait=True)
        self.move_group_gripper.go([0.01,0.01], wait=True)

    def lidar_callback(self, data):
        """
        Average the LIDAR distances for a 10 angle
        spead in front of the robot.
        """
        # ensure that environment is ready before calling this func 
        if not self.initialized:
            return 
        front_angles = [355 + x for x in range(5)] + [x for x in range(5)]
        distances = []
        # select small subset of angles to be the front 
        for angle in front_angles:
            if data.ranges[angle] < data.range_max:
                 distances.append(data.ranges[angle])
        # nothing is in sight, set to front distance max range 
        if len(distances) == 0:
            self.front_distance = data.range_max
        # else, average readings from subset of front angles 
        else:
            self.front_distance = sum(distances)/len(distances)

    def image_callback(self, msg):
        """
        Upon receiving a camera image from the robot,
        either look for a dumbbell, or run OCR, depending
        on the state.
        """
        # ensure that environment is ready before calling this func 
        if not self.initialized:
            return 
        # converts the incoming ROS message to cv2 format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, w, d = image.shape
        self.camera_width = w 

        if self.state == "dumbbell":
            # In the dumbbell state, look for the current target
            # dumbbell by color
            # looking for blue db 
            if self.db_target == "blue":
                lower = np.array([ 110, 50, 50])
                upper = np.array([130, 255, 255])
            # looking for green db 
            elif self.db_target == "green":
                lower = np.array([40,40,40])
                upper = np.array([70,255,255])
            # looking for red db 
            elif self.db_target == "red":
                lower = np.array([0, 50, 50])
                upper = np.array([10, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            h, w, d = image.shape

            # Cut off the top 10th of the camera, because the arm
            # is there, and the red dot creates a false positive.
            mask[0:h//10, 0:w] = 0
            M = cv2.moments(mask)
            # check if target color in sight 
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                self.target_db_angle = w/2 - cx
                self.target_db_visible = True
            # target color is not in sight 
            else:
                self.target_db_visible = False

        elif self.state == "point_toward_block":
            # In the point_toward_block state, run OCR
            # when requested to find the correct block.

            # If OCR has not been requested, return now
            if not self.look_for_block:
                return 
            self.prediction_groups = self.pipeline.recognize([image]) 
            self.look_for_block = False 
        
if __name__ == '__main__':
    p = WeightLifter()
    rospy.spin()

