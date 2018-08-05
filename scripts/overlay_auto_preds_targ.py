"""Use this script for making visuals with overlaid predictions and targets.

This automates the task, given some scripts that I ran, e.g. after:
    ./main/grasp.sh | tee logs/grasp.log
Better than copying/pasting, after all...

So, we need a bunch of those pickle files that we stored out of training.
They should have the best set of predictions, along with the actual labels.
From there, we have the information we need to do analysis and figures.

CAUTION! This assumes I did it using my fast data collection where there's
exactly two grasps (and two successes) per rollouts file. This will need to
be adjusted if that is not the case ...

Update: actually, in addition to this, let's create a heat map. We can also
plot where our data is collected, to see where the bias is located.
"""
import cv2, os, pickle, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim

# ADJUST
ROLLOUT_HEAD = '/nfs/diskstation/seita/bed-make/rollouts_white_v01'
rgb_baseline = True
if rgb_baseline:
    RESULTS_PATH = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/white_fix26_80r_rgb'
    OUTPUT_PATH  = '/nfs/diskstation/seita/bed-make/figures/white_v01_rgb'
else:
    RESULTS_PATH = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/white_v01_fix26_80r'
    OUTPUT_PATH  = '/nfs/diskstation/seita/bed-make/figures/white_v01'
pfiles = sorted([x for x in os.listdir(RESULTS_PATH) if '_raw_imgs.p' in x])

INNER = 3
OUTER = 4
all_targs = []
all_preds = []
all_L2s   = []
all_x     = []
all_y     = []


for pf in pfiles:
    other_pf = pf.replace('_raw_imgs.p','.p')
    print("\n\n\n ----- Now on: {}\n ----- with:   {}".format(pf, other_pf))
    data_imgs  = pickle.load( open(os.path.join(RESULTS_PATH,pf),'rb') )
    data_other = pickle.load( open(os.path.join(RESULTS_PATH,other_pf),'rb') )

    # Load predictions, targets, and images.
    y_pred = data_other['preds']
    y_targ = data_other['targs']
    c_imgs = data_imgs['c_imgs_list']
    d_imgs = data_imgs['d_imgs_list']
    assert len(y_pred) == len(y_targ) == len(c_imgs) == len(d_imgs)
    if 'cv_indices' in data_other:
        cv_ids = data_other['cv_indices']
    print("Now dealing with CV (rollout) indices: {}".format(cv_ids))

    # Index into y_pred, y_targ, c_imgs, d_imgs.
    idx = 0

    for rnum in cv_ids:
        print("\n=====================================================================")
        path = os.path.join(ROLLOUT_HEAD, 'rollout_{}/rollout.p'.format(rnum))
        if not os.path.exists(path):
            print("Error: {} does not exist".format(path))
            sys.exit()
        data = pickle.load(open(path,'rb'))
        print("loaded: {}".format(path))
        print("rollout {}, len(data): {}".format(rnum, len(data)))
        assert len(data) == 4
        assert data[0]['type'] == 'grasp'
        assert data[1]['type'] == 'success'
        assert data[2]['type'] == 'grasp'
        assert data[3]['type'] == 'success'

        # Unfortunately we'll assume that we have two grasps. For now we know this is the case, but
        # we can also load the rollout (as we do) and further inspect within that just to confirm.
        for g_in_rollout in range(2):
            # Get these from our training run.
            pred = y_pred[idx]
            targ = y_targ[idx]
            cimg = c_imgs[idx].copy()
            dimg = d_imgs[idx].copy()
            idx += 1

            l2 = np.sqrt( (pred[0]-targ[0])**2 + (pred[1]-targ[1])**2)
            c_suffix = 'rollout_{}_grasp_{}_rgb_L2_{:.0f}.png'.format(rnum,g_in_rollout,l2)
            d_suffix = 'rollout_{}_grasp_{}_depth_L2_{:.0f}.png'.format(rnum,g_in_rollout,l2)
            c_path = os.path.join(OUTPUT_PATH, c_suffix)
            d_path = os.path.join(OUTPUT_PATH, d_suffix)

            # For visualization later; good idea, Ron, Ajay, Honda :-).
            all_targs.append(targ)
            all_preds.append(pred)
            all_L2s.append(l2)
            all_x.append(targ[0])
            all_y.append(targ[1])

            # Alternatively could get from rollout paths. Good to double check. Unfortunately again
            # this assumes I did grasp then success then grasp then success ... yeah.
            pose = data[g_in_rollout*2]['pose']
            print("pose: {}, targ: {} (should match)".format(pose, targ))
            pose = (int(pose[0]), int(pose[1]))
            targ = (int(targ[0]), int(targ[1]))
            #assert pose[0] == targ[0] and pose[1] == targ[1], "pose {}, targ {}".format(pose, targ)
            preds = (int(pred[0]), int(pred[1]))
            print("predictions: {}".format(preds))

            # Overlay the pose to the image (red circle, black border).
            cv2.circle(cimg, center=pose, radius=INNER, color=(0,0,255), thickness=-1)
            cv2.circle(dimg, center=pose, radius=INNER, color=(0,0,255), thickness=-1)
            cv2.circle(cimg, center=pose, radius=OUTER, color=(0,0,0), thickness=1)
            cv2.circle(dimg, center=pose, radius=OUTER, color=(0,0,0), thickness=1)
    
            # The PREDICTION, though, will be a large blue circle (yellow border?).
            cv2.circle(cimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
            cv2.circle(dimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
            cv2.circle(cimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
            cv2.circle(dimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
    
            cv2.imwrite(c_path, cimg)
            cv2.imwrite(d_path, dimg)
    
    print("=====================================================================")
print("\nDone with creating overlays, look at {}".format(OUTPUT_PATH))


# ------------------------------------------------------------------------------
# Next part: plots!
# ------------------------------------------------------------------------------
all_targs = np.array(all_targs)
all_preds = np.array(all_preds)
all_L2s = np.array(all_L2s)
print("\nall_targs.shape: {}".format(all_targs.shape))
print("all_preds.shape: {}".format(all_preds.shape))
print("all_L2s.shape:   {}\n".format(all_L2s.shape))

# Create scatter plots and heat maps.
nrows = 2
ncols = 2
fig, ax = plt.subplots(nrows, ncols, figsize=(16,12))
tsize = 20
tick_size = 18
xlim = [0, 640]
ylim = [480, 0]
alpha = 0.5

# ------------------------------------------------------------------------------
# Heat-map of the L2 losses. I was going to use the contour code but that
# requires a value exists for every (x,y) pixel pair from our discretization.
# To get background image in matplotlib:
# https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
# This might be helpful:
# https://stackoverflow.com/questions/48472227/how-can-one-create-a-heatmap-from-a-2d-scatterplot-data-in-python
# TODO: double check these values are what I want ... also, colorbars?
# ------------------------------------------------------------------------------
I = cv2.imread("scripts/imgs/image_example.png")

# Create a scatter plot of where the targets are located.
ax[0,0].imshow(I, alpha=alpha)
ax[0,0].scatter(all_targs[:,0], all_targs[:,1], color='black')
ax[0,0].set_title("Ground Truth", fontsize=tsize)

# Along with predictions.
ax[0,1].imshow(I, alpha=alpha)
ax[0,1].scatter(all_preds[:,0], all_preds[:,1], color='black')
ax[0,1].set_title("Grasp Network Predictions", fontsize=tsize)

# Heat map now.
ax[1,0].imshow(I, alpha=alpha)
cf = ax[1,0].tricontourf(all_x, all_y, all_L2s, cmap='YlOrRd')
fig.colorbar(cf, ax=ax[1,0])
mean = np.mean(all_L2s)
std = np.std(all_L2s)
ax[1,0].set_title("L2 Losses (Heat Map). {:.1f} +/- {:.1f}".format(mean, std), fontsize=tsize)

# Original image.
ax[1,1].imshow(I)
ax[1,1].set_title("Original Image", fontsize=tsize)

# Bells and whistles
for i in range(nrows):
    for j in range(ncols):
        ax[i,j].set_xlim(xlim)
        ax[i,j].set_ylim(ylim)
        ax[i,j].tick_params(axis='x', labelsize=tick_size)
        ax[i,j].tick_params(axis='y', labelsize=tick_size)

plt.tight_layout()
figname = "check_predictions_scatter_map.png"
plt.savefig(figname)
print("Look at this figure: {}".format(figname))
