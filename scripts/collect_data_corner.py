import numpy as np
import os, cv2, hsrb_interface # (make sure we are in hsrb_mode)
from il_ros_hsr.core.sensors import RGBD
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

SAVE_PATH = 'corner_data/'

# make save dir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Orient the robot appropriately. make sure it is starting about a foot away from the long side of the bed, facing parallel to the bed
robot = hsrb_interface.Robot()
whole_body = robot.get('whole_body')
whole_body.move_to_go()
whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
whole_body.move_to_joint_positions({'head_pan_joint': np.pi/2.0})
whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})
whole_body.move_to_joint_positions({'arm_lift_joint': 0.120})

# start collecting data
rollout_num = 1 # do about 120 rollouts for decent dataset
camera = RGBD()
labels = list()
while True:
	print("rollout", rollout_num)
	# generate random numbers for top/bottom, success/failure of rollout
	top = np.random.rand(2)
	success = np.random.rand(2)
	labels.append(str(success)) # classification labels
	print("Top (1)/Bottom (0): ", top)
	print("Success (1)/Failure (0): ", success)
	raw_input("press enter to take the picture")
	c_img = camera.read_color_data()
    d_img = camera.read_depth_data()
    # preprocess depth image
    if np.isnan(np.sum(d_img)):
        cv2.patchNaNs(d_img, 0.0)
    d_img = depth_to_net_dim(d_img, cutoff=1400)
    # save images
    cv2.imwrite(os.path.join(SAVE_PATH, "rgb_" + str(rollout_num) + ".png"), c_img)
    cv2.imwrite(os.path.join(SAVE_PATH, "depth_" + str(rollout_num) + ".png"), d_img)
	rollout_num += 1

# write labels to output
with open(os.path.join(SAVE_PATH, "labels.txt")) as fh:
	fh.write("\n".join(labels))

