import torch
import os

class HParams:
    def __init__(self):
        self.Dataset = 'QuickDraw'
        if self.Dataset == 'QuickDraw':
            self.categories = 345
            self.max_seqlen = 321

        elif self.Dataset == 'QuickDraw414k':
            self.categories = 345
            self.max_seqlen = 100

        self.mode = 'Train'
        self.model_save = 'ckpts-huge'
        self.model_name = 'MoE'
        self.load_model = False
        self.log = 'log'     

        if not os.path.exists(self.model_save):
            os.mkdir(self.model_save)
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self.img_ch = 3

        self.net_struct =[10, 2, 2, 2, 2]
        self.img_size = 224
        self.drop_rate = 0.1
        self.patch_size = 16
        self.ps = [16,8,4,2,1]
        self.d_ffn = 334*4
        self.d_model =334
        self.seq_len = 322 #QuickDraw 350, TU_Berlin 448, QuickDraw414k 100
        self.num_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size)
        self.pos_enc= False
        self.c_com = 128
        self.expert = True
        self.sf_num = 3 #3 original

        self.mix_weight = 1
        self.seq_weight = 0.8
        self.img_weight = 0.8
        self.cv_weight = 0.0

        self.batchsize = 384
        self.learning_rate = 0.0004
        self.schedule_step = 1
        self.lr_gamma = 0.1
        self.lr_schedule = None
        self.lr_min = 1e-6
        self.nThreads = 4
        self.epochs = 16
        self.gpus = [0, 1,2,3,4,5]
        self.warmup_step = 4e3

        self.save_epoch_freq = 1
        self.save_step_freq = 200
        self.valid_epoch_freq = 1
        self.start_epoch = 0
        self.model_path = './ckpts/MoE_epoch_'+str(self.start_epoch)+'.pkl'
hp = HParams()
