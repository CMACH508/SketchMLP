import cv2
import numpy as np
import random
from Hyper_params import hp
import torch
import math
import warnings
from typing import Callable, Iterable, Optional, Tuple, Union
from PIL import Image
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# cv2.setNumThreads(0)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  偶数正弦
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  奇数余弦

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return ((sinusoid_table + 1) / 2 * 255).astype(np.int16)


def draw_three(tsketch, random_color=False, img_size=hp.img_size, stroke_flag=1):
    color_idx = 0
    thickness = int(img_size * 0.025)
    sketch = tsketch.copy()
    if hp.Dataset == 'QuickDraw':
        sketch[:, 0:2] = sketch[:, 0:2] * img_size / 256 + thickness
    else:
        sketch[:, 0:2] = sketch[:, 0:2] * img_size / 256 + thickness

    canvas = np.ones((img_size + 3 * (thickness + 1), img_size + 3 * (thickness + 1), hp.img_ch), dtype='uint8') * 255

    if random_color:
        ds = dual_state(sketch)
        length = len(ds)
        cl_tab = np.zeros([length,3])
        cl_tab[:,0] = sketch[:length,2].copy()
        cl_tab[:,1:] = ds
        cl_tab = ((cl_tab)*255).astype(np.int16)
    else:
        color = (0, 0, 0)
    pen_now = np.array([sketch[0, 0], sketch[0, 1]])
    first_zero = False

    for stroke in sketch:
        delta_x_y = stroke[0:0 + 2] - pen_now
        state = stroke[2:]

        if int(state) == -1:
            break
        if first_zero:  # 首个零是偏移量, 不画
            pen_now += delta_x_y
            first_zero = False
            continue

        if random_color:
            color = tuple([int(x) for x in cl_tab[color_idx]])
            color_idx = color_idx + 1

        cv2.line(canvas, tuple([int(pen_now[0]), int(pen_now[1])]), tuple([int((pen_now + delta_x_y)[0]),int((pen_now + delta_x_y)[1])]), color, thickness=thickness)
        #cv2.line(canvas1, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=1)
        #cv2.line(canvas2, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=10)


        if int(state) == stroke_flag:  # next stroke
            first_zero = True

        pen_now += delta_x_y
    #canvas[:,:,0] = canvas1[:,:,0]
    #canvas[:,:,2] = canvas2[:,:,2]
    return Image.fromarray(cv2.resize(canvas, (img_size, img_size)))


def save_checkpoint(net=None, optimizer=None, epoch=None, train_losses=None, train_acc=None, val_loss=None,
                    val_acc=None, check_loss=None, savepath=None, m_name=None, GPUdevices=1):
    if GPUdevices > 1:
        net_weights = net.module.state_dict()
    else:
        net_weights = net.state_dict()
    save_json = {
        'net_state_dict': net_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    if check_loss > val_loss:
        savepath = savepath + '/{}_best_params.pkl'.format(m_name)
        check_loss = val_loss
    else:
        savepath = savepath + '/{}_epoch_{}.pkl'.format(m_name, epoch)
    torch.save(save_json, savepath)
    print("checkpoint of {}th epoch saved at {}".format(epoch, savepath))

    return check_loss


def load_checkpoint(model=None, optimizer=None, checkpoint_path=None, losses_flag=None):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['net_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    if not losses_flag:
        return model, optimizer, start_epoch
    else:
        losses = checkpoint['train_losses']
        return model, optimizer, start_epoch, losses


def off2abs(tvector, img_size=256):
    off_vector = tvector.copy()
    new_sketch = -np.ones_like(off_vector)
    new_sketch[:, 2] = off_vector[:, 2]
    a = off_vector[:, :2]
    h_s = np.max(a[:,0])-np.min(a[:,0])
    w_s = np.max(a[:,1])-np.min(a[:,1])
    if h_s>w_s:
        scale = h_s
    else:
        scale = w_s
    new_sketch[:,0] = (a[:,0]-np.min(a[:,0]))/scale*img_size
    new_sketch[:,1] = (a[:,1]-np.min(a[:,1]))/scale*img_size
    return new_sketch


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(hp.lr_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def points3to5(sketch):
    sketch_len = len(sketch)
    new_sketch = np.zeros([hp.seq_len, 5], dtype=np.int16)
    new_sketch[0, :2] = sketch[0, :2]
    new_sketch[1:sketch_len, :2] = sketch[1:sketch_len, :2] - sketch[:sketch_len - 1, :2]
    new_sketch[:sketch_len - 1, 2] = 1 - sketch[:-1, 2]
    new_sketch[:sketch_len - 1, 3] = sketch[:-1, 2]
    new_sketch[sketch_len - 1:, 4] = 1
    return new_sketch

def dual_state(sketch):
    record = -np.ones([len(sketch), 2])
    begin_idx = 0
    end_idx = 0
    stroke_idx = 1
    stroke_num = 1
    for i, points in enumerate(sketch):
        record[i, 0] = stroke_idx
        record[i, 1] = stroke_num
        end_idx = i
        if sketch[i, -1] == 0:
            record[begin_idx:end_idx + 1, 0] = record[begin_idx:end_idx + 1, 0] / stroke_idx
            begin_idx = end_idx + 1
            stroke_num = stroke_num + 1
            stroke_idx = 1
        elif sketch[i, -1] == 1:
            stroke_idx = stroke_idx + 1
        elif sketch[i, -1] == -1 or (i + 1) == len(sketch):
            record[:i + 1, 1] = record[:i + 1, 1] / stroke_num
            break
    return record

