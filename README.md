# q_learning_project

# Implementation Plan

Team Members: Sophie Veys, Alex Oro

## Q-Learning

We will use the provided action matrix to randomly select actions and 
execute them using the provided phantom movement and virtual world scripts.
We will use the provided equation for computing Q, and will experiment 
with the parameters. We will keep track of the previous 5 to 10 Q matrices,
and if the change between those matrices is below a threshold, we will stop training.
Once training is complete, we will use the Q-matrix by having the robot loop through 
all possible actions, and perform the one that would lead to the highest Q value.

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
For Q-Learning, we will use the phantom movement script to visually 
inspect training. We may also compute some terms of the matrix
by hand to see if the robot is training as expected. We will also
try training the robot multiple times, and see if it converges to similar
answers.

## Timeline

We plan to finish the Q-learning portion by May 1st, and then complete the 
movement and navigation by May 7th.
