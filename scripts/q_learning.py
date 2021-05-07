#!/usr/bin/env python3

import rospy
import numpy as np
import random
import os
import copy
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from q_learning_project.msg import RobotMoveDBToBlock, QLearningReward, QMatrix

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

class QLearning(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("q_learning")

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-9 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][12] = 5
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt")

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { dumbbell: "red", block: 1}
        colors = ["red", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"dumbbell": colors[int(x[0])], "block": int(x[1])},
            self.actions
        ))


        # Fetch states. There are 64 states. Each row index corresponds to the
        # state number, and the value is a list of 3 items indicating the positions
        # of the red, green, blue dumbbells respectively.
        # e.g. [[0, 0, 0], [1, 0 , 0], [2, 0, 0], ..., [3, 3, 3]]
        # e.g. [0, 1, 2] indicates that the green dumbbell is at block 1, and blue at block 2.
        # A value of 0 corresponds to the origin. 1/2/3 corresponds to the block number.
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.txt")
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))

        # The Q Matrix, which can be indexed with a state and an action 
        # to get a Q value
        self.q_matrix = np.zeros([len(self.states), len(self.actions)])

        # All of the publishers/subscribers for handling training
        self.action_publisher = rospy.Publisher('/q_learning/robot_action', RobotMoveDBToBlock, queue_size=10)
        self.matrix_publisher = rospy.Publisher('/q_learning/q_matrix', QMatrix, queue_size=10)
        self.subscriber = rospy.Subscriber('/q_learning/reward', QLearningReward, self.get_reward)

        # Keep track of which state the robot is in,
        # and where the robot will go
        self.current_state = 0
        self.next_state = 0
        self.current_action = 0

        # Training parameters for reinforcement learning
        self.alpha = 1
        self.gamma = 0.8
       
        # Information for determing when the 
        # training has converged 
        self.previous_matrices = []
        self.convergence_threshold = 0.01
        self.min_iterations = 1000

        # Wait for everything to start up
        rospy.sleep(1)

        # Start training
        self.train_one_iteration()

    def get_reward(self, reward):
        """
        When a reward is recieved, update the self.q_matrix,
        and then check for convergence. If training has not
        converged, run another iteration.
        """
        
        # The actual Q Learning algorithm
        self.q_matrix[self.current_state][self.current_action] = self.q_matrix[self.current_state][self.current_action] + self.alpha * (reward.reward + self.gamma * max(self.q_matrix[self.next_state]) - self.q_matrix[self.current_state][self.current_action])
        self.previous_matrices.append(copy.deepcopy(self.q_matrix))
        self.current_state = self.next_state
        
        # Publish the matrix for grading
        matrix_msg = QMatrix()
        matrix_msg.q_matrix = self.q_matrix.tolist()
        self.matrix_publisher.publish(matrix_msg)

        if not self.is_converged(reward.iteration_num):
            self.train_one_iteration()
        else:
            rospy.loginfo("converged")
            rospy.loginfo(self.q_matrix)
            self.save_q_matrix()

    def is_converged(self, iteration):
        """
        Check if the training has converged
        """
        # Set the minimum number of iterations
        if iteration < self.min_iterations:
            return False
        # Check the last ten matrices. If the difference 
        # for any term with the current matrix is above
        # self.convergence_threshold, then it has not
        # converged.
        most_recent = self.previous_matrices[-1]
        previous = self.previous_matrices[-10:-1]
        for m in previous:
            difference = abs(m - most_recent).max()
            if difference > self.convergence_threshold:
                return False
        return True

    def train_one_iteration(self):
        """
        Pick the  next action to take and publish it
        """        
        possible_actions = [(i, int(x)) for i, x in enumerate(self.action_matrix[self.current_state]) if x != -1]
        # If there are no possible actions, then restart
        # from the beginning. Otherwise, pick an action
        # and publish it. Once the reward is received,
        # the next iteration will be started by the 
        # get_reward function
        if len(possible_actions) == 0:
            self.next_state = 0
            self.current_state = 0
            self.current_action = 0
            self.train_one_iteration()
        else:
            self.next_state, self.current_action = random.choice(possible_actions)
            action_msg = RobotMoveDBToBlock()
            action_msg.robot_db = self.actions[self.current_action]["dumbbell"]
            action_msg.block_id = self.actions[self.current_action]["block"]
            self.action_publisher.publish(action_msg)

    def save_q_matrix(self):
        """
        Save the QMatrix to a csv file
        """
        np.savetxt("qmatrix.csv", self.q_matrix, delimiter=",")

if __name__ == "__main__":
    node = QLearning()
    rospy.spin()
