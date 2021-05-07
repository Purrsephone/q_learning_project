# q_learning_project

# Implementation Plan

Team Members: Sophie Veys, Alex Oro

## Q-Learning

We will use the provided action matrix to randomly select actions and 
execute them using the provided phantom movement and virtual world scripts.
We will use the provided equation for computing Q, and will experiment 
with the parameters. We will keep track of the previous 10 Q matrices,
and if the change between those matrices is below a threshold, we will stop training. 
Additionally, we will set a minimum number of runs to ensure the algo doesn't converge too soon. 
Once training is complete, we will use the Q-matrix by having the robot loop through 
all possible actions, and perform the one that would lead to the highest Q value.

## Robot Perception

We will use the RGB camera to locate the dumbbells by searching for pixels with their colors, 
defining both min and max values for the colors as we did for the line follower exercise. 
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

We will test the movement/navigation by visual inspection in Gazebo 
and printing the color/number values that the robot is reporting. 
For Q-Learning, we will use the phantom movement script to visually 
inspect training. We may also compute some terms of the matrix
by hand to see if the robot is training as expected. We will also
try training the robot multiple times, and see if it converges to similar
answers.

## Timeline

We plan to finish the Q-learning portion by May 1st, and then complete the 
movement and navigation by May 7th.


# Write-Up 

## Objectives description (2-3 sentences): Describe the goal of this project.
The first goal of this project was to use Q-learning to identify the best actions for the robot to take from each possible state to maximize its reward. The second goal of this project was to program the robot to actually take these actions.   

## High-level description (1 paragraph): At a high-level, describe how you used reinforcement learning to solve the task of determining which dumbbells belong in front of each numbered block.

Until the q-matrix converged, we continued to pick random actions and used the q-learning algorithm to determine the best action for each state based on the associated reward values. We checked for convergence between iterations. We also considered which states were reachable from the current state. This yielded a q-matrix with q-values for each action in each state (when possible). We then used this q matrix to select the 3 action sequence that would yield the highest reward.    
	

## Q-learning algorithm description: Describe how you accomplished each of the following components of the Q-learning algorithm in 1-3 sentences, and also describe what functions / sections of the code executed each of these components (1-3 sentences per function / portion of code):

### Selecting and executing actions for the robot (or phantom robot) to take 
		
While the q matrix had not converged, we chose actions at random, performed them, computed the resulting state, and logged the q values. We repeated this until there were no possible actions from our current state, in which case we set the state back to 0 and repeated the process. This can be found in the train_one_iteration and get_reward functions.  

### Updating the Q-matrix
We saved the computed q value in the q matrix at the index representing the current state and current action. This can be found in the get_reward function of q_learning.py.   

### Determining when to stop iterating through the Q-learning algorithm
We kept track of the previous 10 q matrices. We compared the latest matrix to the previous 10, and decided that it had converged if there was minimal difference between the current and previous attempts. We also set a minimum number of iterations (10000) because otherwise the matrix would converge after 10 iterations simply because it had not found anything yet. This can be found in the is_converged function of q_learning.py).   


### Executing the path most likely to lead to receiving a reward after the Q-matrix has converged on the simulated Turtlebot3 robot 
We chose to perform the action with the highest q value in state 0. This action moved us to another state, so we again selected the action with the highest q value. After 3 repetitions, we ended up a mapping of dumbbells to blocks that maximized the reward. This can be found in the init function of the Weightlifter class (lines 62—83 of movement.py).   

### Robot perception description: Describe how you accomplished each of the following components of the perception elements of this project in 1-3 sentences, any online sources of information/code that helped you to recognize the objects, and also describe what functions / sections of the code executed each of these components (1-3 sentences per function / portion of code):

### Identifying the locations and identities of each of the colored dumbbells 
We defined upper and lower bounds for colors and used OpenCV to narrow in on pixels of a specified color. We then used proportional control and the lidar sensor to navigate toward the dumbbell of that color.The identification code can be found in the image_callback function in movement.py and the movement code can be found in run.py (and associated helper functions).  

### Identifying the locations and identities of each of the numbered blocks**  
We used KerasOCR pre-trained models to identify the numbers on the blocks. We then used proportional control and the lidar sensor to navigate toward the block with the desired number. The identification code can be found in the image_callback function in movement.py and the movement code can be found in run.py (and associated helper functions).  

### Robot manipulation and movement: Describe how you accomplished each of the following components of the robot manipulation and movement elements of this project in 1-3 sentences, and also describe what functions / sections of the code executed each of these components (1-3 sentences per function / portion of code): 

### Moving to the right spot in order to pick up a dumbbell 
We followed a similar approach to the line follower example from class, where we searched for the color of the dumbbell, and then used OpenCV and proportional control to point the robot towards the dumbbell. Then, we used LIDAR to bring the robot close to the dumbbell. This is handled by the `dumbbell` state in lidar_callback and run in the WeightLifter class.  

### Picking up the dumbbell
When moving towards the dumbbell, we kept the arm lowered, and moved until the robot was close enough that the dumbbell was in the claw. Then, we had the robot close its claw and pull the dumbbell high above its head. This is handled by the WeightLifter.raise_arm function, which is called in the `dumbbell` state in WeightLifter.run  

### Moving to the desired destination (numbered block) with the dumbbell  
We had the robot rotate a bit to the left, and then wait and run image recognition. Once it found the desired number, it would use proportional control to point towards it, and then use LIDAR to move to the target. We periodically have the robot switch back to image recognition to make sure it is pointing in the correct direction. This is handled by the `point_toward_block` and `move_toward_block` states in WeightLifter.run, WeightLifter.lidar_callback, and WeightLifter.image_callback.  

### Putting the dumbbell back down at the desired destination
Once the robot was close enough to the block according to LIDAR, we instructed it to lower its arm, release the grip on the dumbbell, and then pull back a bit. This is handled by the `drop_db` state in WeightLifter.run, and the WeightLifter.lower_arm function.  

## Challenges (1 paragraph): Describe the challenges you faced and how you overcame them.  
We had some difficulties with Gazebo, and would occasionally get inconsistent results or see programs crash. We resolved this by frequently restarting everything. Working with the arm was a bit difficult, as finding the right angles for both reaching far enough forward, but also keeping the robot balanced, was difficult. But we were able to work through that through trial and error though. In general, tuning any magic numbers proved difficult, as it was hard to tell if the robot was, for example, moving a bit too fast, or if something was seriously wrong with our code. In the end, we were able to debug and tune all of these issues.  

## Future work (1 paragraph): If you had more time, how would you improve your implementation?   
One area for improvement is in how the robot picks up the dumbbell. Currently, it gets very close, and then nudges it a bit until it’s at the right angle to pick it up. Given more time, we could modify this so the robot can use it’s LIDAR to more effectively locate the dumbbell, and pick it up more smoothly. Another way we could improve the robot is speeding up the phase where it locates the numbered blocks. We could probably do this by finding a faster algorithm that simply recognizes if there are numbers present in the image, and use this to turn more quickly until the robot sees numbers. Once it sees some numbers, then it could run the more accurate OCR and slowly point towards the block.   

## Takeaways (at least 2 bullet points with 2-3 sentences per bullet point): What are your key takeaways from this project that would help you/others in future robot programming assignments working in pairs? For each takeaway, provide a few sentences of elaboration.



