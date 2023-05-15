import sys
import cv2
import torch
import math
import time
import numpy as np
from math import ceil
from torch.autograd import Variable
from imageio import imread
from model.PWCNet import PWCNet
import matplotlib.pyplot as plt

UNKNOWN_FLOW_THRESH = 1e7

# Code to generate flow image by Daigo Hirooka, from:
# https://github.com/daigo0927/pwcnet/blob/master/flow_utils.py
def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            print(data.shape)
            data = 1.0 * data
            data = torch.from_numpy(data)
            return data
        return None

    
def epe(flow_pred, flow_gt):
    distance = torch.norm(flow_gt - flow_pred, p=2, dim=1)
    e = torch.mean(distance)
    return e


# Code to scale images and scale estimated flow from original PWC-Net implementation
# https://github.com/NVlabs/PWC-Net

im1_fn = '.data\\MPI-Sintel-complete\\training\\clean\\alley_1\\frame_0001.png'
im2_fn = '.data\\MPI-Sintel-complete\\training\\clean\\alley_1\\frame_0002.png'
flow_fn = 'test_frame.flo'
flow_gt_path = '.data\\MPI-Sintel-complete\\training\\flow\\alley_1\\frame_0001.flo'

saved_pwc_model = 'best_model_100_epochs_2023_05_08_17_10_33_model.pt'

im_all = [imread(img) for img in [im1_fn, im2_fn]]
im_all = [im[:, :, :3] for im in im_all]

flow_gt = torch.squeeze(load_flow(flow_gt_path)).cpu()

# rescale the image size to be multiples of 64
divisor = 64.
H = im_all[0].shape[0]
W = im_all[0].shape[1]

H_ = int(ceil(H/divisor) * divisor)
W_ = int(ceil(W/divisor) * divisor)
for i in range(len(im_all)):
	im_all[i] = cv2.resize(im_all[i], (W_, H_))

for _i, _inputs in enumerate(im_all):
	im_all[_i] = im_all[_i][:, :, ::-1]
	im_all[_i] = 1.0 * im_all[_i]/255.0
	im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
	im_all[_i] = torch.from_numpy(im_all[_i])
	im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
	im_all[_i] = im_all[_i].float()
    
im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

model = PWCNet()
if saved_pwc_model is not None:
    data = torch.load(saved_pwc_model)
    model.load_state_dict(data)

model = model.cuda()
model.eval()

start = time.time()
flow = model(im_all)[0]
end = time.time()
print("inference time: {}".format(end-start))
flow = flow[0] * 20.0
flow = flow.cpu().data.numpy()

# scale the flow back to the input size 
flow = np.swapaxes(np.swapaxes(flow, 0, 1), 1, 2) # 
u_ = cv2.resize(flow[:,:,0],(W,H))
v_ = cv2.resize(flow[:,:,1],(W,H))
u_ *= W/ float(W_)
v_ *= H/ float(H_)
flow = np.dstack((u_,v_))

flow = torch.from_numpy(flow)

print("flow: {}, flow_gt: {}".format(flow.size(), flow_gt.size()))
err = epe(flow, flow_gt)
print("epe: {}".format(err))

flow = flow.numpy()

flow_im = flow_to_image(flow)

# Visualization
plt.imshow(flow_im)
plt.axis('off')
plt.show()