"""Use this script to _format_ Ron's raw data into trainable data.

Ideally we will have run `check_ron_data.py` beforehand.
"""
import cv2, pickle, sys, os
import numpy as np
np.set_printoptions(suppress=True, linewidth=200, precision=5)


# --- ADJUST PATHS ---
RAW_PATH       = '/nfs/diskstation/seita/bed-make/corner_ron_data_v02/'
TARG_PATH_BOTH = '/nfs/diskstation/seita/bed-make/rollouts_ron_v02/' # combine angles
TARG_PATH_C0   = '/nfs/diskstation/seita/bed-make/rollouts_ron_v02_c0/' # angle 0 only
TARG_PATH_C1   = '/nfs/diskstation/seita/bed-make/rollouts_ron_v02_c1/' # angle 1 only
kfold = 10

# I know the bins of armState[0] and headState[0] by inspecting the data beforehand.
arm_bins = np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
cs0_bins = np.array([-1.06465, -0.7854])

def find_nearest(array, value):
    """For finding nearest bin value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# For each data point, we'll add to this list.
c0_rollouts = []
c1_rollouts = []


files = sorted([x for x in os.listdir(RAW_PATH) if x[-4:] == '.pkl'])

for pkl_f in files:
    number = str(pkl_f.split('.')[0])
    file_path = os.path.join(RAW_PATH,pkl_f)
    data = pickle.load(open(file_path,'rb'))
    cs0_bin_idx = find_nearest(cs0_bins, data['headState'][0])
    print("\nOn data: {}, number {}".format(file_path, number))
    print("    data['armState']:  {}".format(np.array(data['armState'])))
    print("    data['headState']: {}".format(np.array(data['headState'])))
    print("    camera index: {}".format(cs0_bin_idx))
    assert data['RGBImage'].shape == (480, 640, 3)
    assert data['depthImage'].shape == (480, 640)

    # Format these into the dictionaries we actually need.
    # The 'side' and 'class' keys should be ignored by training code.
    # Note: maybe use 'd_img': (data['depthImage'].copy()).astype('uint16')
    # in our code we have uint16 for depth but Ron's uses float32.
    # These depth values don't span in (0,255) so be careful, that's part of processing.
    # Pose must be float as we later divide to be in [-1,1] for optimization.

    x,y = data['markerPos']
    info = {
        'c_img': data['RGBImage'].copy(),
        'd_img': data['depthImage'].copy(),
        'pose':  [float(x), float(y)] ,
        'type': 'grasp',
        'side': 'BOTTOM',
        'class': 0,
    }

    if cs0_bin_idx == 0:
        c0_rollouts.append(info)
    elif cs0_bin_idx == 1:
        c1_rollouts.append(info)
    else:
        raise ValueError(cs0_bin_idx)

L0 = len(c0_rollouts)
L1 = len(c1_rollouts)
print("len(c0_list): {}".format(L0))
print("len(c1_list): {}".format(L1))


def save_rollouts(pp, rollout_list, kfold, head):
    """Use to save rollouts from permutation indices pp and list of rollouts.
    `array_split` results in list of: `[arr_1, arr_2, ..., arr_kfold]`, where
    each `arr_X` is a numpy array of indices for this cross validation group.
    """
    groups = np.array_split(pp, kfold)

    for (cv_idx, array) in enumerate(groups): rollout = [] for idx in array:
            rollout.append( rollout_list[idx] )

        head2 = os.path.join(head,'rollout_'+str(cv_idx))
        if not os.path.exists(head2):
            os.makedirs(head2)

        with open( os.path.join(head2,'rollout.p'), 'w' ) as f:
            pickle.dump(rollout, f)


# Randomly allocate different indices for cross validation purposes.
pp0 = np.random.permutation(L0)
pp1 = np.random.permutation(L1)
pp  = np.random.permutation(L0+L1)
save_rollouts(pp0, c0_rollouts,             kfold, TARG_PATH_C0)
save_rollouts(pp1, c1_rollouts,             kfold, TARG_PATH_C1)
save_rollouts(pp,  c0_rollouts+c1_rollouts, kfold, TARG_PATH_BOTH)
