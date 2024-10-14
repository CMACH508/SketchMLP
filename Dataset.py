import os
import lmdb
import torch

import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from Utils import draw_three, off2abs
from Hyper_params import hp
from Data.sketch_util import SketchUtil
from PIL import Image


class Dataset_Quickdraw(data.Dataset):
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'Train':
            self.TrainData_ENV = lmdb.open('./Data/train_QuickDraw', max_readers=4,
                                           readonly=True, lock=False, readahead=False, meminit=False)
        elif self.mode == 'Test':
            self.TestData_ENV = lmdb.open('./Data/test_QuickDraw', max_readers=4,
                                          readonly=True, lock=False, readahead=False, meminit=False)

        elif self.mode == 'Valid':
            self.ValidData_ENV = lmdb.open('./Data/valid_QuickDraw', max_readers=4,
                                           readonly=True, lock=False, readahead=False, meminit=False)

        with open("./Data/train_QuickDraw.pkl", "rb") as handle:
            self.Train_keys = pickle.load(handle)
        with open("./Data/test_QuickDraw.pkl", "rb") as handle:
            self.Test_keys = pickle.load(handle)
        with open("./Data/valid_QuickDraw.pkl", "rb") as handle:
            self.Valid_keys = pickle.load(handle)

        print('Total Training Sample {}'.format(len(self.Train_keys)))
        print('Total Valid Sample {}'.format(len(self.Valid_keys)))
        print('Total Testing Sample {}'.format(len(self.Test_keys)))

        self.train_transform = get_ransform('Train')
        self.valid_transform = get_ransform('Valid')
        self.test_transform = get_ransform('Test')

    def __getitem__(self, item):

        if self.mode == 'Train':
            with self.TrainData_ENV.begin(write=False) as txn:
                sample = txn.get(str(item).encode())
                sketch_points = np.fromstring(sample, dtype=np.uint16).reshape(-1, 3).copy().astype(np.float32)
                sketch_points[:,0:2] = sketch_points[:,0:2]/128-1

                if random.uniform(0, 1) > 0.5 and self.mode == 'Train':
                    sketch_points[:, 0:2] = SketchUtil.random_affine_transform(sketch_points[:, 0:2], scale_factor=0.2,
                                                                               rot_thresh=45.0)
                sketch_points[:, 0:2] = (sketch_points[:, 0:2]+1)*128
                sketch_points[:, 0:2] = SketchUtil.Q414k_horizontal_flip(sketch_points[:, 0:2]/256)*256

            sketch_points[:, -1] = 1 - sketch_points[:, -1]
            sketch_points = np.concatenate([sketch_points,[[0,0,0]]],axis=0)
            sketch_img = draw_three(sketch_points, stroke_flag=0)
            sketch_img = self.train_transform(sketch_img)

            padded_sketch = -np.ones([hp.seq_len, hp.sf_num], dtype=np.float32) 
            padded_sketch[:len(sketch_points), :] = sketch_points
            
            padded_sketch[:len(sketch_points), :2] = padded_sketch[:len(sketch_points), :2]/256

            padded_sketch = torch.tensor(padded_sketch / 1.0).to(torch.float32)
            sample = {'sketch_img': sketch_img, 'sketch_points': padded_sketch,
                      'sketch_label': self.Train_keys[str(item)], 'seq_len': len(sketch_points)}


        elif self.mode == 'Test':
            with self.TestData_ENV.begin(write=False) as txn:
                sample = txn.get(str(item).encode())
                sketch_points = np.fromstring(sample, dtype=np.uint16).reshape(-1, 3).copy().astype(np.float32)

            sketch_points[:, 0:2] = sketch_points[:, 0:2]
            sketch_points[:, -1] = 1 - sketch_points[:, -1]
            sketch_points = np.concatenate([sketch_points,[[0,0,0]]])

            sketch_img = draw_three(sketch_points, stroke_flag=0)
            sketch_img = self.test_transform(sketch_img)

            padded_sketch = -np.ones([hp.seq_len, hp.sf_num], dtype=np.float32)
            padded_sketch[:len(sketch_points), :] = sketch_points
            padded_sketch[:len(sketch_points), :2] = padded_sketch[:len(sketch_points), :2]/256

            padded_sketch = torch.tensor(padded_sketch / 1.0).to(torch.float32)
            sample = {'sketch_img': sketch_img, 'sketch_points': padded_sketch,
                      'sketch_label': self.Test_keys[str(item)], 'seq_len': len(sketch_points)}

        elif self.mode == 'Valid':
            with self.ValidData_ENV.begin(write=False) as txn:
                sample = txn.get(str(item).encode())
                sketch_points = np.fromstring(sample, dtype=np.uint16).reshape(-1, 3).copy().astype(np.float32)

            sketch_points[:, -1] = 1 - sketch_points[:, -1]
            sketch_points = np.concatenate([sketch_points,[[0,0,0]]])

            sketch_img = draw_three(sketch_points, stroke_flag=0)
            sketch_img = self.valid_transform(sketch_img)

            padded_sketch = -np.ones([hp.seq_len, hp.sf_num], dtype=np.float32)
            padded_sketch[:len(sketch_points), :] = sketch_points
            padded_sketch[:len(sketch_points), :2] = padded_sketch[:len(sketch_points), :2]/256

            padded_sketch = torch.tensor(padded_sketch / 1.0).to(torch.float32)
            sample = {'sketch_img': sketch_img, 'sketch_points': padded_sketch,
                      'sketch_label': self.Valid_keys[str(item)], 'seq_len': len(sketch_points)}
        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_keys)
        elif self.mode == 'Test':
            return len(self.Test_keys)
        elif self.mode == 'Valid':
            return len(self.Valid_keys)


class Dataset_TUBerlin(data.Dataset):
    def __init__(self, mode, drop_strokes=True):
        self.pkl_file = 'Data/TUBerlin.pkl'
        self.mode = mode
        self.drop_strokes = drop_strokes

        with open(self.pkl_file, 'rb') as fh:
            saved = pickle.load(fh)
            self.categories = saved['categories']
            self.sketches = saved['sketches']
            self.cvxhulls = saved['convex_hulls']
            self.folds = saved['folds']

        self.fold_idx = None
        self.indices = list()
        self.set_fold(2)

        self.train_transform = get_ransform('Train')
        self.valid_transform = get_ransform('Valid')
        self.test_transform = get_ransform('Test')

    def set_fold(self, idx):
        self.fold_idx = idx
        self.indices = list()

        if self.mode == 'Train':
            for i in range(len(self.folds)):
                if i != idx:
                    self.indices.extend(self.folds[i])
        else:
            self.indices = self.folds[idx]

        print('[*] Created a new {} dataset with {} fold as validation data'.format(self.mode, idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cid, sid = self.indices[idx]
        sid_points = np.copy(self.sketches[cid][sid])

        if self.mode == 'train':
            cvxhull = self.cvxhulls[cid][sid]
            pts_xy = sid_points[:, 0:2]
            if cvxhull is not None:
                if random.uniform(0, 1) > 0.5:
                    pts_xy = SketchUtil.random_cage_deform(np.copy(cvxhull), pts_xy, thresh=0.1)
                    pts_xy = SketchUtil.normalization(pts_xy)
                if random.uniform(0, 1) > 0.5:
                    pts_xy = SketchUtil.random_affine_transform(pts_xy, scale_factor=0.2, rot_thresh=45.0)
            pts_xy = SketchUtil.random_horizontal_flip(pts_xy)
            sid_points[:, 0:2] = pts_xy
            if self.drop_strokes:
                sid_points = self._random_drop_strokes(sid_points)

        sketch_points = sid_points.astype(np.float32)
        sketch_points[:,0:2] = (sketch_points[:,0:2]+1)*128
        sketch_points[:,-1] = 1-sketch_points[:,-1]
        
        sketch_points = np.concatenate([sketch_points,[0,0,0]],axis=0)
        sketch_img = draw_three(sketch_points, flag=0)
        if hp.sf_num == 5:
            dual_state = Utils.dual_state(sketch_points)
            sketch_points = np.concatenate([sketch_points,dual_state], axis=-1).astype('float32')

        elif hp.sf_num == 4:
            dual_state = Utils.dual_state(sketch_points)
            sketch_points = np.concatenate([sketch_points,dual_state[:,0:1]], axis=-1).astype('float32')

        sketch_points[:, 0:2] = sketch_points[:, 0:2] / 256

        if self.mode == 'Train':
            sketch_img = self.train_transform(sketch_img)
        else:
            sketch_img = self.test_transform(sketch_img)
        padded_sketch = -np.ones([hp.seq_len, hp.sf_num], dtype=np.float32)  # 搴斾娇鐢╥nt16锛屼箣鍓嶄娇鐢ㄤ簡uint16
        padded_sketch[:len(sketch_points), :] = sketch_points
        padded_sketch = torch.tensor(padded_sketch / 1.0).to(torch.float32)
        sample = {'sketch_img': sketch_img, 'sketch_points': padded_sketch,
                  'sketch_label': cid, 'seq_len': len(sketch_points)}
        return sample

    def _random_drop_strokes(self, points3):
        strokes = SketchUtil.to_stroke_list(points3)
        num_strokes = len(strokes)
        if num_strokes < 2:
            return points3
        sort_idxes = SketchUtil.compute_stroke_orders([s[:, 0:2] for s in strokes])
        keep_prob = np.random.uniform(0, 1, num_strokes)
        keep_prob[:(num_strokes // 2)] = 1
        keep_idxes = np.array(sort_idxes, np.int32)[keep_prob > 0.5]
        keep_strokes = [strokes[i] for i in sorted(keep_idxes.tolist())]
        return np.concatenate(keep_strokes, axis=0)

    def num_categories(self):
        return len(self.categories)

    def dispose(self):
        pass

    def get_name_prefix(self):
        return 'TUBerlin-{}-{}'.format(self.mode, self.fold_idx)


class Quickdraw414k(data.Dataset):

    def __init__(self, mode='Train'):
        self.mode = mode
        if mode == 'Train':
            sketch_list = "Data/QuickDraw414k/picture_files/tiny_train_set.txt"
            path_root1 = 'Data/QuickDraw414k/picture_files/train'
            path_root2 = 'Data/QuickDraw414k/coordinate_files/train'
        elif mode == 'Test':
            sketch_list = "Data/QuickDraw414k/picture_files/tiny_test_set.txt"
            path_root1 = 'Data/QuickDraw414k/picture_files/test'
            path_root2 = 'Data/QuickDraw414k/coordinate_files/test'
        elif mode == 'Valid':
            sketch_list = "Data/QuickDraw414k/picture_files/tiny_val_set.txt"
            path_root1 = 'Data/QuickDraw414k/picture_files/val'
            path_root2 = 'Data/QuickDraw414k/coordinate_files/val'

        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            self.img_urls = [os.path.join(path_root1, sketch_url.strip().split(' ')[
                0]) for sketch_url in sketch_url_list]
            self.coordinate_urls = [os.path.join(path_root2, (sketch_url.strip(
            ).split(' ')[0]).replace('png', 'npy')) for sketch_url in sketch_url_list]

            self.labels = [int(sketch_url.strip().split(' ')[-1])
                           for sketch_url in sketch_url_list]
        print('Total ' + mode + ' Sample {}'.format(len(self.labels)))

        self.train_transform = get_ransform('Train')
        self.valid_transform = get_ransform('Valid')
        self.test_transform = get_ransform('Test')

    def __len__(self):
        return len(self.img_urls)

    def __getitem__(self, item):
        sketch_url = self.img_urls[item]
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]
        # img = Image.open(sketch_url, 'r').resize((224, 224))

        seq = np.load(coordinate_url, encoding='latin1', allow_pickle=True)
        if seq.dtype == 'object':
            seq = seq[0]
        assert seq.shape == (100, 4)
        seq = seq.astype('float32')
        seq = seq[:, 0:3]
        index_neg = np.where(seq == -1)[0]
        

        if len(index_neg) == 0:
            seq = off2abs(seq)

            if random.uniform(0, 1) > 0.5 and self.mode == 'Train':
               seq[:, 0:2] = SketchUtil.random_affine_transform(seq[:, 0:2], scale_factor=0.2, rot_thresh=45.0)

            if self.mode == 'Train':
               seq[:, 0:2] = SketchUtil.Q414k_horizontal_flip(seq[:, 0:2]/256)*256

            img = draw_three(seq, stroke_flag=0)
            seq[:, 0:2] = seq[:, 0:2] / 256

        else:
            index_neg = index_neg[0]
            seq[:index_neg,:] = off2abs(seq)[:index_neg,:]

            if random.uniform(0, 1) > 0.5 and self.mode == 'Train':
               seq[:, 0:2] = SketchUtil.random_affine_transform(seq[:, 0:2], scale_factor=0.2, rot_thresh=45.0)
            if self.mode == 'Train':
               seq[:index_neg, 0:2] = SketchUtil.Q414k_horizontal_flip(seq[:index_neg, 0:2]/256)*256

            img = draw_three(seq, stroke_flag=0)
            seq[:index_neg, 0:2] = seq[:index_neg, 0:2] / 256


        if self.mode == 'Train':
            sketch_img = self.train_transform(img)
        elif self.mode == 'Test':
            sketch_img = self.test_transform(img)
        elif self.mode == 'Valid':
            sketch_img = self.valid_transform(img)
        sample = {'sketch_img': sketch_img, 'sketch_points': seq,
                  'sketch_label': label, 'seq_len': 100}
        return sample


def collate_self(batch):
    batch_mod = {'sketch_img': [], 'sketch_points': [],
                 'sketch_label': [], 'seq_len': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['seq_len'].append(i_batch['seq_len'])

        padded_sketch = -np.ones([hp.seq_len, 3], dtype=np.int16)  # 搴斾娇鐢╥nt16锛屼箣鍓嶄娇鐢ㄤ簡uint16
        padded_sketch[:i_batch['seq_len'], :] = i_batch['sketch_points']

        batch_mod['sketch_points'].append(torch.tensor(padded_sketch / 1.0))

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])
    batch_mod['sketch_points'] = torch.stack(batch_mod['sketch_points'], dim=0).to(torch.float32)
    return batch_mod


def get_dataloader():
    if hp.Dataset == 'TUBerlin':

        dataset_Train = Dataset_TUBerlin(mode='Train')
        dataset_Test = Dataset_TUBerlin(mode='Test')
        dataset_Valid = Dataset_TUBerlin(mode='Valid')


    elif hp.Dataset == 'QuickDraw':

        dataset_Train = Dataset_Quickdraw(mode='Train')
        dataset_Test = Dataset_Quickdraw(mode='Test')
        dataset_Valid = Dataset_Quickdraw(mode='Valid')

    elif hp.Dataset == 'QuickDraw414k':
        dataset_Train = Quickdraw414k(mode='Train')
        dataset_Test = Quickdraw414k(mode='Test')
        dataset_Valid = Quickdraw414k(mode='Valid')

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True, pin_memory=True,
                                       num_workers=int(hp.nThreads))

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                      num_workers=int(hp.nThreads))

    dataloader_Valid = data.DataLoader(dataset_Valid, batch_size=300, shuffle=False,
                                       num_workers=int(hp.nThreads))
    # dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
    #                                      num_workers=int(hp.nThreads), collate_fn=collate_self)
    #
    # dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
    #                                      num_workers=int(hp.nThreads), collate_fn=collate_self)
    #
    # dataloader_Valid = data.DataLoader(dataset_Valid, batch_size=hp.batchsize, shuffle=False,
    #                                   num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test, dataloader_Valid


def get_ransform(type):
    transform_list = []
    # if type is 'Train':
    # transform_list.extend([transforms.RandomRotation(45), transforms.RandomHorizontalFlip()])
    # elif type is 'Test':
    #     transform_list.extend([transforms.Resize(256)])
    # elif type is 'Valid':
    #     transform_list.extend([transforms.Resize(256)])
    # transform_list.extend(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms.Compose(transform_list)
