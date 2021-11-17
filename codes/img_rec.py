import numpy as np
import cv2
import torch
from torch import optim
import torch.nn.functional as F
from GaussianSmoothing import GaussianSmoothing
from copy import deepcopy
from math import inf, nan
from matplotlib import pyplot as plt
import pandas as pd

width = 346
height = 260

def accumulate_event(events, P_matrix):
    device = events.device

    xx = events[:, 0].floor()
    yy = events[:, 1].floor()
    dx = events[:, 0] - xx
    dy = events[:, 1] - yy

    H = height + 1
    W = width + 1
    zero_v = torch.tensor([0.], device=device)
    ones_v = torch.tensor([1.], device=device)
    index = torch.where(xx>=0, ones_v, zero_v)*torch.where(yy>=0, ones_v, zero_v)*torch.where(xx<(W - 1), ones_v, zero_v)*torch.where(yy<(H - 1), ones_v, zero_v)
    index = index.bool()
    xx = xx[index].long()
    yy = yy[index].long()
    dx = dx[index]
    dy = dy[index]
    events[events[:, 3] == 0, 3] = -1
    polarity = events[index, 3]
    weights = P_matrix[index]
    iwe = accumulate_operation(xx, yy, dx, dy, polarity, weights, H, W, device)
    return iwe[:-1, :-1]

def accumulate_operation(pxx, pyy, dx, dy, polarity, weights, height, width, device):
    iwe = torch.zeros(size = torch.Size([height, width]), device = device)
    iwe.index_put_((pyy    , pxx    ), weights*polarity*(1.-dx)*(1.-dy), accumulate=True)
    iwe.index_put_((pyy    , pxx + 1), weights*polarity*dx*(1.-dy), accumulate=True)
    iwe.index_put_((pyy + 1, pxx    ), weights*polarity*(1.-dx)*dy, accumulate=True)
    iwe.index_put_((pyy + 1, pxx + 1), weights*polarity*dx*dy, accumulate=True)
    return iwe

def events2frame(warped_events_batch, P_matrix):
    device = warped_events_batch.device
    pos_batch = warped_events_batch[warped_events_batch[:,3]==1]
    neg_batch = warped_events_batch[warped_events_batch[:,3]==0]
    P_matrix_pos = P_matrix[warped_events_batch[:,3]==1]
    P_matrix_neg = P_matrix[warped_events_batch[:,3]==0]

    frame1 = accumulate_event(pos_batch, P_matrix_pos)
    frame0 = accumulate_event(neg_batch, P_matrix_neg)

    frame = torch.stack((frame1, frame0))
    return frame

def variance(iwe):
    return -iwe.std()**2

def convGaussianFilter(frame, k_size = 5, sigma = 1):
    device = frame.device
    dim = frame.size()
    smoothing = GaussianSmoothing(dim[0], k_size, sigma, device)
    frame = torch.unsqueeze(frame, 0)
    frame = F.pad(frame, (2, 2, 2, 2), mode='reflect')
    frame = torch.squeeze(smoothing(frame))
    return frame

def iwe_weights_per_event(frames, warped_events):
    frame_pos = frames[0].unsqueeze(0).unsqueeze(0).float()
    frame_neg = frames[1].unsqueeze(0).unsqueeze(0).float()
    warped_x = warped_events[:, 0].unsqueeze(0).unsqueeze(0)
    warped_y = warped_events[:, 1].unsqueeze(0).unsqueeze(0)
    warped_x = 2 * warped_x / (width - 1) - 1
    warped_y = 2 * warped_y / (height - 1) - 1
    grid_pos = torch.cat([warped_x, warped_y], dim=1).permute(0, 2, 1).unsqueeze(0)
    pos_weights = F.grid_sample(frame_pos, grid_pos, mode="bilinear", padding_mode="zeros", align_corners = True)
    neg_weights = F.grid_sample(frame_neg, grid_pos, mode="bilinear", padding_mode="zeros", align_corners = True)
    pos_weights = pos_weights.squeeze()
    neg_weights = neg_weights.squeeze()

    weights = pos_weights - neg_weights
    return weights

def loss_func(vels, P_matrix, events_batch, it):
    #mask = P_matrix >= 0.5
    # xs_batch = events_batch[:,0]
    # ys_batch = events_batch[:,1]
    # ts_batch = events_batch[:,2]
    # ps_batch = events_batch[:,3]
    # ts_batch = events_batch[:,0]
    # xs_batch = events_batch[:,1]
    # ys_batch = events_batch[:,2]
    # ps_batch = events_batch[:,3]
    # xs_cluster1 = xs_batch[mask]
    # ys_cluster1 = ys_batch[mask]
    # ts_cluster1 = ts_batch[mask]
    # ps_cluster1 = ps_batch[mask]
    # xs_cluster2 = xs_batch[~mask]
    # ys_cluster2 = ys_batch[~mask]
    # ts_cluster2 = ts_batch[~mask]
    # ps_cluster2 = ps_batch[~mask]
    xs_wrapped1 = events_batch[:,0] + events_batch[:,2] * 87.17457525692306 #vels[0]
    xs_wrapped2 = events_batch[:,0] + events_batch[:,2] * vels[1]
    warped_events_cluster1 = torch.stack((xs_wrapped1, events_batch[:,1], events_batch[:,2], events_batch[:,3]), dim=1)
    warped_events_cluster2 = torch.stack((xs_wrapped2, events_batch[:,1], events_batch[:,2], events_batch[:,3]), dim=1)
    # warped_events_batch1 = events_batch.clone()
    # warped_events_batch2 = events_batch.clone()

    # warped_events_xs1 = warped_events_batch1[:, 0] - warped_events_batch1[:, 2] * vels[0]
    # warped_events_xs2 = warped_events_batch2[:, 0] - warped_events_batch2[:, 2] * vels[1]
    # warped_events_batch1[:, 0] = warped_events_xs1.clone()
    # warped_events_batch2[:, 0] = warped_events_xs2.clone()
    
    frame1 = events2frame(warped_events_cluster1, P_matrix)
    frame2 = events2frame(warped_events_cluster2, 1.-P_matrix)

    iwe1 = torch.sum(frame1, dim=0).detach()
    iwe2 = torch.sum(frame2, dim=0).detach()
    iwe1_np = iwe1.float().cpu().numpy()
    iwe2_np = iwe2.float().cpu().numpy()
    # plt.imshow(iwe1_np, 'gray')
    # plt.title('Motion 1')
    plt.imsave('../../results/esai/iteration/motion1/{:09d}.png'.format(it), iwe1_np, cmap='gray')
    plt.imsave('../../results/esai/iteration/motion2/{:09d}.png'.format(it), iwe2_np, cmap='gray')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(iwe2_np, 'gray')
    # plt.title('Motion 2')
    # plt.colorbar()
    # plt.show()

    # P_matrix_np = P_matrix.detach().float().cpu().numpy().reshape(150, 200)
    # plt.matshow(P_matrix_np)
    # plt.title('P_matrix')
    # plt.colorbar()
    # plt.show()

    
    frame1 = convGaussianFilter(frame1)
    frame2 = convGaussianFilter(frame2)

    weights_batch1 = iwe_weights_per_event(frame1, warped_events_cluster1)
    weights_batch2 = iwe_weights_per_event(frame2, warped_events_cluster2)
    updated_P_matrix = weights_batch1 / (weights_batch1 + weights_batch2 + 1e-6)
    
    loss1 = variance(frame1.abs())
    loss2 = variance(frame2.abs())

    return loss1 + loss2, updated_P_matrix

def optimization(init_vels, P_matrix, events_tensor, device):
    optimizer_name = 'Adam'
    lr = 0.5
    lr_step = 250
    lr_decay = 0.1
    iters = 250
    if lr_step <= 0:
        lr_step = max(1, iters)
    vels = torch.from_numpy(init_vels.copy()).float().to(device)
    vels.requires_grad = True
    # initializing optimizer
    optimizer = optim.__dict__[optimizer_name]([vels],lr =lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
    print_interval = 10
    min_loss = inf
    best_vels = vels
    best_it = 0
    # optimization process
    if optimizer_name == 'Adam':
        for it in range(iters):
            optimizer.zero_grad()
            vels_val = vels.cpu().detach().numpy()
            if nan in vels_val:
                print("nan in the estimated values, something wrong takes place, please check!")
                exit()
            loss, P_matrix = loss_func(vels, P_matrix.detach(), events_tensor, it)
            # loss = loss_func(vels, P_matrix, events_tensor)
            if it == 0:
                print('[Initial]\tloss: {:.12f}\tposes: {}'.format(loss.item(), vels_val))
            elif (it + 1) % print_interval == 0:
                print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}'.format(it + 1, iters, loss.item(), vels_val))
            if loss < min_loss:
                best_vels = vels
                min_loss = loss.item()
                best_it = it
            try:
                loss.backward(retain_graph=True)
            except Exception as e:
                print(e)
                return vels_val, loss.item()
            optimizer.step()
            scheduler.step()
    else:
        print("The optimizer is not supported.")

event_path = "Example_data/Raw/Event/0001.npy"
data = np.load(event_path,allow_pickle=True).item()
eventData = data.get('events')
reference_time = data.get('ref_t')
fx = data.get('fx')
d = data.get('depth')
v = 0.177 * 3.0
x = eventData[:,0]
print(len(x))
shift_x = np.round((eventData[:,2] - reference_time) * fx * v / d)
valid_ind = (x+shift_x >= 0) * (x+shift_x < 346)
x[valid_ind] += shift_x[valid_ind]
x_filtered = x[valid_ind]
print(len(x_filtered))
eventData[:,0] = x

print(0.177 * fx / d)

# eventData = pd.read_csv('events_5.txt', sep=' ', header=None, engine='python')
# print(eventData.shape)
# eventData.columns = ["ts", "x", "y", "p"]
# events_set = eventData.to_numpy()

#events = deepcopy(eventData[:30000, :])
events = deepcopy(eventData)
#events[:,2] = events[:,2] - reference_time
print(events[:10, :])
# events[:,0] = events[:,0] - events[0,0]
num_events = events.shape[0]
para0 = np.array([87.17457525692306, 87.17457525692306 * 7.0])
# para0 = np.array([800.0])
P_matrix = np.ones(num_events, dtype=np.float32) / 2.0
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  
torch.cuda.empty_cache()
events_tensor = torch.from_numpy(events).float().to(device)
P_matrix = torch.from_numpy(P_matrix).float().to(device)
frame_tmp = events2frame(events_tensor, P_matrix)
iwe_tmp = torch.sum(frame_tmp, dim=0).detach()
iwe_np = iwe_tmp.float().cpu().numpy()
plt.imshow(iwe_np, 'gray')
plt.title('Demo')
plt.colorbar()
plt.show()
#res, loss = optimization(para0, P_matrix, events_tensor, device)