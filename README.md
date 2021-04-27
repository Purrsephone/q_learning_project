# q_learning_project

# Implementation Plan

Team Members: Sophie Veys, Alex Oro

## Q-Learning

TODO write this part

## Robot Perception

We will use the RGB camera to locate the dumbbells by searching for pixels with their colors. 
We will use keras_ocr with the robots camera to recognize where the numbered blocks are, and point towards them. 
For both dumbbells and blocks, we will use the LIDAR to tell how far away we are from it.

## Robot manipulation & movement

We will have two states for the arm, lowered and raised. 
We will use the LIDAR sensor to align the arm with the objects
before gripping the dumbbell and raising the arm. 
For navigation, once we have found the object the robot needs to 
go to, we will point the robot towards it using the camera, and drive
forward until the LIDAR sensor indicates the robot is close enough.

## Testing

We will test the movement/navigation by visual inspection in Gazebo.
For Q-Learning, TODO

## Timeline

We will finish half this week, and half next week. (TODO figure out which half)
