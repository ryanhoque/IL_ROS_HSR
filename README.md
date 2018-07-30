# IL_ROS_HSR

Code for finishing up the bed-making project. It's based on Michael Laskey's old code.

# Bed-Making Instructions

Here are full instructions for the bed-making project with the HSR.


- [Installation](#installation)
- [Setup for Data Collection](#setup-for-data-collection)
    - [The Bed Frame](#the-bed-frame)
    - [The Bed Sheet](#the-bed-sheet)
- [Slow Data Collection](#slow-data-collection)
- [Fast Data Collection](#fast-data-collection)
- [Neural Network Training](#neural-network-training)
- [Evaluation](#evaluation)


## Installation

Install this and the [fast_grasp_detect repository][2]:

- I use the [requirements.txt file shown here][6] in my Python 2.7 virtualenv, but it's probably
  easy to just `pip install` things as you go. *From now on, I assume you are in the Python
  virtualenv*.
- Use `python setup.py develop` in the two repositories.
- Install TensorFlow 1.8.
- Install the TMC code library, which includes `hsrb_interface`.
- Adjust and double-check the [configuration file][1] and other paths, to ensure that you're
  referring to correct workspaces.
- Also double check that the overall data directory you're writing to is mounted and accessible.
- Make sure the HSR is charged, but that the charging tube is disconnected.

## Setup for Data Collection

First, set up the bed frame (and fix one side of the sheet). Second, during data collection, we'll
want to re-arrange the sheet on the bed in various configurations.

You should try and get the bed setup to look [like what we have in this GIF][5], with the exception
of perhaps the precise sheet (and with different robots if necessary). Notice the AR marker in the
GIF.


### The Bed Frame

To set up a bed, get the initial frame with a dark blue sheet fixed on it so it stays still, and
find a clear, open space. Having a fixed background (e.g., blue in this case) is useful for quickly
evaluating performance since we can measure the colors there using OpenCV. For space, the robot just
needs to go around one side of the bed, as shown in the images below. In addition, there also needs
to be space for an [AR 11 Marker][3], which is specific to the HSR (see below for instructions on
how to arrange the bed relative to the HSR). The AR marker must also be oriented correctly; rotating
it by 90 degrees, for instance, will change the other coordinate frames that we rely on.

![](imgs/init_setup_01.JPG)

![](imgs/init_setup_02.JPG)

In terms of dimensions:

- The **bed frame** should be 26 x 36 inches in dimension.
- The **bed sheet** should be (about) 36 x 42 inches. I've found that this makes it reliably avoid
  issues with the corners lying over the edge of the bed, which should make it fine for collecting
  training data. The Cal and Teal blankets are 40 x 42 inches, so when applying transfer learning
  we'll probably want to increase the offset goal towards where the robot should pull the sheet and
  ensure that the setup avoids "extreme" cases with corners lying outside the top of the bed frame.

Align the 42 inch side of the bed with the 36 inch side of the bed frame, as expected.

For a fully flattened sheet, have 2-3 inches of extra space at the shorter end of the bed:

![](imgs/init_setup_03.JPG)

In addition, for the long side, have 5 inches or less extra space. If it's 6-7 inches or longer,
then this risks having more corners that are not on top of the bed frame, making it hard (if not
impossible) for the HSR to grasp at those points.

![](imgs/init_setup_04.JPG)

Apply pins in the back to make it sturdy. Double check that the previous measurements are still
roughly approximate.

![](imgs/init_setup_05.JPG)

**(TODO need to describe collecting data on opposite side, can we detach the fabric?)**

Now we have the sheet on the bed, for initial data collection.  **How do we know where to precisely
put the bed?** Previously, we taped AR marker 11 on the ground, so move the HSR by using our
built-in joystick, so that it can see the AR marker in its cameras.  **For these steps, be sure you
are in HSRB mode (`export ROS_MASTER_URI ...`) and in the correct python virtual environment as
discussed earlier!** 

- Run `python scripts/joystick_X.py` first and then move the robot to the designated starting
  position. (It should be marked with tape ... put tape on if it isn't!)
- Kill the joystick script.
- In another tab, run `rosrun rviz rviz`.
- Get the bed setup by putting the bed towards where the rviz markers are located. Just a coarse
  match is expected and sufficient.  To be clear:
    - Match the following frames: `head up`, `head down`, `bottom up`, and `bottom down`.
    - The centers should be fixed, but the orientation can be a bit confusing ... for now I'll keep
      it the way it is. At least the blue axis (z axis I think) should point *downwards*. 
    - In order to do this step, you need to run `python main/collect_data_bed.py` first to get the
      frames to appear, but hopefully after this, no changes to the bed positioning are needed.
    - If you run this the first time, it is likely you will need to reposition the robot so that the
      AR marker is visible. Use rviz for visualizing, but again, this requires the data collection
      script to be run for the AR marker/11 frame to appear.
- The easiest way to do the last step above is by running `python main/collect_data_bed.py` ---
  which we have to do anyway for the data collection --- and adjusting at the beginning.

Here's what my rviz setup looks like:

![](imgs/rviz_1.png)

![](imgs/rviz_2.png)

Note I: the bed is as close to the AR marker as possible.

Note II: in the older project where grasp points =/= corner points, the `head_down` and `head_up`
frames were where we actually told the HSR gripper to go to after it gripped a sheet. Since we now
have corners, the corners need to be dragged slightly further away from the actual corner of the bed
frame. However, we'll leave `head_down` and `head_up` as frames that represent *bed frame
locations*.


### The Bed Sheet

**TODO need init collection** 

Reminders:

- Don't put the red marker on the opposite half of the bed from the HSR. It won't be able to reach
  the other end.
- The corner must be visible.


## Fast Data Collection

For faster data collection, use `python main/collect_data_bed_fast.py`.

- This involves us manually simulating what the sheet would look like. This way, 50 "rollouts"
  can be collected in two hours.
- However, this involves some care to ensure that the human acts like the robot would ... and we
  definitely need to double check the data by visualization, etc.
- It is saved in a similar format so that's good, except the ordering of the grasp or successes
  might not be the same, but my code can handle both. Also, the rollouts may not necessarily
  have logical connections to each other, but it's best to simulate them so that they are
  temporally connected.





## Slow Data Collection

1. As mentioned above, run `python main/collect_data_bed.py`. Make sure there
are no error messages. 

    - If there are no topics found initially, that likely means the AR marker is
      not visible. Please check rviz.
    - For each setup, we sample a starting bed configuration with a red line and
      then have to *physically adjust the bed sheet to match that image*. This
      provides the variation in starting states. Again, only coarse matches are
      needed. However, please keep the bed adjusted so that it is close to the
      z-axis (colored blue) of the two bottom poses.

2. After the sheet has been adjusted, the robot can move.

    - Press B on the joystick. Then the robot should move up to the bed, and
      pause.
    - An image loader/viewer pops up. Click "Load" (upper right corner) to load
      the current camera image of the robot.
    - Drag the bounding box where the robot should grasp. Click "send command"
      and then close the window. *The grasp will not execute until the window is
      closed*.

3. After the grasp, we need to check for transitioning vs re-grasping.

    - Load the image as usual by clicking "Load". 
    - Below the "Load" button, drag and drop either "Success" or "Failure"
      depending on what we think happened.
    - Click "Confirm Class". This is especially important! What matters is the
      class that you see in the list that appears.
    - Draw a bounding box. I don't think it matters if we know the robot
      transitions, but if the robot has to re-grasp, then definitely put the
      correct pose.
    - Send the command, close the window.

4. Data Storage

    - After the HSR has transitioned to the start, it will save the data under
      the specified directory, as `rollout_X.p` where `X` is the index. Check
      the print logs for the location.
    - The `rollout_X.p` is a list of length K, where K>=5.  Use
      `scripts/quick_rollout_check.py` to investigate quickly.  It contains:
        - A list of two 3-D points, representing the "down corner" and "up
          corner" respectively, which are used for the initial state sampling.
        - And then a bunch of dictionaries, all with these keys:
            - `c_img`: camera image. Note that if a grasp just failed, then
              the subsequent image that the success network would see is the
              same exact image as the grasping network would see. This makes
              sense: at attempt `t` just after we think a grasp failure
              happened, the image `I` is what the success net would see, so it
              must classify it as a failure. Then in the next dictionary, `I`
              stays the same since we have figure out where to grasp next.
            - `d_img`: depth image. Don't worry about this too much.
            - `class`: either 0 (success/good) or 1 (failure/bad), use these for 
              the 'success' type.
            - `pose`: a 2D point from where we marked it in the interface.
              You'll see it in the Tkinter pop-up menu. Use these for the
              'grasp' types.
            - `type`: 'grasp' or 'success'
            - `side`: 'BOTTOM' or 'TOP' (the HSR starts in the bottom position)
        - These repeat for each grasp and success check, and for both sides.
          The first dictionary is of type 'grasp' and represents the data that
          the grasping network would see, and has 'pose' as the center of the
          bounding box we drew. The second dictionary is of type 'success' for a
          success check, and which also lets us draw a bounding box. Two cases:
            - *First grasp succeeded?* The next dictionary has `c_img`
              corresponding to the top of the bed, with type 'grasp', and has a
              different `pose` corresponding to the next bounding box we drew.
              So the bounding box we draw for the success network, assuming
              the previous grasp succeeded, is ignored.
            - *First grasp failed?* The next dictionary has the same `c_img` as
              discussed above, with type 'grasp'. It also has the same `pose`
              since we should have drawn it just now. (The pose is also
              effectively ignored, except during the interface, we need to be
              careful about where we draw the pose in this case because it
              immediately impacts the next grasp attempt.)
          The cycle repeats. So either way the two types alternate.
      Hence, the shortest length of `rollout_X.p` is 5, because the bottom and
      top each require two dictionaries (one each for grasping and then the
      success). Of course, it will be longer whenever we have to re-grasp.

Here's an example of the pop-up menu. In the "Bounding Boxes" the class that
gets recorded is shown there. Once you see something like this, you can "Send
Command" and close it.

![](imgs/failure_1.png)





## Neural Network Training

0. Data dimension: by default we do NOT use the raw (480,640,3)-sized images, but we pass them
through a pre-trained YOLO network to get (14,14,1024)-dimensional features, and THEN we do the rest
of the stuff from there. In other words, when we call a training minibatch, we will get a batch of
size (B,14,14,1024).

1. Collect and understand the data. 

    - The easiest way to understand the data is by running: `python scripts/check_raw_data.py` as
      that will give us statistics as well as save images that we can use for later.
    - Also do `python scripts/data_augmentation_example.py` to check data augmentation, for both 
      the depth and the RGB images (check the code to change settings).

2. Investigate what kind of training works best for the grasp data. For this, perform cross
validation on the grasping data. (And maybe the success data, but for now just do grasping.)

    - Check the configuration file for grasping. Make sure:
        - `self.PERFORM_CV = True`
        - `self.CV_HELD_OUT_INDEX` is set to a number between 0 and 9, inclusive.
        - `self.ROLLOUT_PATH` refers to where all the 50 (or so) rollouts are stored.
        - `self.CV_GROUPS` splits the rollouts randomly and evenly into groups.
    - Run `python train_bed_grasp.py`. It should load in the network and the data, and will save in
      `grasp_output/`, with the following information by default:
       ```
       seita@autolab-titan-box:/nfs/diskstation/seita/bed-make$ ls -lh grasp_output/*
       -rw-rw-r-- 1 nobody nogroup 336M Jul  5 17:25 grasp_output/07_05_17_24_56_CS_0_save.ckpt-500.data-00000-of-00001
       -rw-rw-r-- 1 nobody nogroup 2.3K Jul  5 17:25 grasp_output/07_05_17_24_56_CS_0_save.ckpt-500.index
       -rw-rw-r-- 1 nobody nogroup 692K Jul  5 17:25 grasp_output/07_05_17_24_56_CS_0_save.ckpt-500.meta
       -rw-rw-r-- 1 nobody nogroup 336M Jul  5 17:26 grasp_output/07_05_17_26_09_CS_0_save.ckpt-1000.data-00000-of-00001
       -rw-rw-r-- 1 nobody nogroup 2.3K Jul  5 17:26 grasp_output/07_05_17_26_09_CS_0_save.ckpt-1000.index
       -rw-rw-r-- 1 nobody nogroup 692K Jul  5 17:26 grasp_output/07_05_17_26_09_CS_0_save.ckpt-1000.meta
       -rw-rw-r-- 1 nobody nogroup  324 Jul  5 17:26 grasp_output/checkpoint
       
       grasp_output/2018_07_05_17_23:
       total 4.0K
       -rw-rw-r-- 1 nobody nogroup 2.0K Jul  5 17:23 config.txt
       
       grasp_output/stats:
       total 4.0K
       -rw-rw-r-- 1 nobody nogroup 720 Jul  5 17:26 grasp_net.p
       ```
        - `config.txt` file is saved in a file reflecting the time the code was run, and has all
          the configurations, so we always know what we ran. :)
        - `stats/grasp_net.p` is a dict where 'test' and 'train' are the test and train losses,
          respectively, saved at some fixed epochs.
        - The other stuff, of course, is from `tf.Saver`.
    - Do something similar to the above for the "success" data.

3. Now train for real. As before, we run `python train_bed_grasp.py` and `python
train_bed_success.py` in the `fast_grasp_detect` repository. But this time make sure
`self.PERFORM_CV = False` so that all the CV stuff is ignored.


## Evaluation

1. Run `python deploy_network.py` for testing the method we propose with deep
imitation learning.

2. Run `python deploy_analytic.py` for testing the baseline method.


[1]:https://github.com/DanielTakeshi/IL_ROS_HSR/blob/master/src/il_ros_hsr/p_pi/bed_making/config_bed.py
[2]:https://github.com/DanielTakeshi/fast_grasp_detect
[3]:https://docs.hsr.io/manual_en/development/ar_marker.html
[4]:https://github.com/DanielTakeshi/fast_grasp_detect/commit/424463e12996b037c3f3539e58d1b5572f4ca835
[5]:http://bair.berkeley.edu/static/blog/dart/bed_making_gif.gif
[6]:https://github.com/DanielTakeshi/IL_ROS_HSR/blob/master/requirements.txt
