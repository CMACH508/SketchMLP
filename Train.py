import os
from Dataset import get_dataloader
import torch
import numpy as np
import torch.nn as nn
from Utils import save_checkpoint, load_checkpoint
from Networks5 import net
from Hyper_params import hp
from tensorboardX import SummaryWriter
import torch.optim as optim
import random
from metrics import AverageMeter,accuracy
from tqdm.auto import tqdm
from Utils import get_cosine_schedule_with_warmup
seed = 1010
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
print("***********- ***********- READ DATA and processing-*************")
dataloader_Train, dataloader_Test, dataloader_Valid = get_dataloader()

print("***********- loading model -*************")
if(len(hp.gpus)==0):#cpu
    model = net()
elif(len(hp.gpus)==1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpus[0])
    model = net().cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = net().cuda()
    gpus = [i for i in range(len(hp.gpus))]
    model = torch.nn.DataParallel(model, device_ids=gpus)

optimizer = optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=0.005) #5e-3
scheduler = get_cosine_schedule_with_warmup(optimizer, hp.warmup_step, int(len(dataloader_Train)*hp.epochs))
loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)

writer = SummaryWriter('log/'+hp.model_name)

class trainer:
    def __init__(self, loss_f, model, optimizer, scheduler):
        self.loss_f = loss_f
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iter = 0

    def batch_train(self, batch_imgs, batch_seqs, batch_labels, epoch):
        predicted, img_logsoftmax, seq_logsoftmax, cv_important = self.model(batch_imgs, batch_seqs)
        cv_important = cv_important.mean()
        loss, mix_loss, img_loss, seq_loss = self.myloss(predicted, img_logsoftmax, seq_logsoftmax, batch_labels)
        loss = loss + hp.cv_weight * cv_important
        del batch_imgs, batch_labels
        return loss, mix_loss, img_loss, seq_loss, cv_important, predicted

    def train_epoch(self, loader, epoch):
        self.model.train()
        tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("\n************Training*************")
        for batch_idx, (batch) in enumerate(tqdm_loader):
            imgs = batch['sketch_img']
            seqs = batch['sketch_points']
            labels = batch['sketch_label']
            if (len(hp.gpus) > 0):
                imgs, seqs, labels = imgs.cuda(non_blocking=True), seqs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                #imgs, seqs, labels = imgs.cuda(), seqs.cuda(), labels.cuda()
            loss, mix_loss, img_loss, seq_loss, cv_important, predicted = self.batch_train(imgs, seqs, labels, epoch)
            losses.update(loss.item(), imgs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()

            err1, err5 = accuracy(predicted.data, labels, topk=(1, 5))
            top1.update(err1.item(), imgs.size(0))
            top5.update(err5.item(), imgs.size(0))

            writer.add_scalar('loss (iter)', loss, self.iter)
            writer.add_scalar('mix loss (iter)', mix_loss, self.iter)
            writer.add_scalar('img loss (iter)', img_loss, self.iter)
            writer.add_scalar('seq loss (iter)', seq_loss, self.iter)
            self.iter = self.iter+1
            tqdm_loader.set_description('Training: loss:{:.4}/{:.4} lr:{:.4} err1:{:.4} err5:{:.4}'.
                                        format(loss, losses.avg, self.optimizer.param_groups[0]['lr'],top1.avg, top5.avg))
        return top1.avg, top5.avg, losses.avg

    def valid_epoch(self, loader, epoch):
        self.model.eval()
        loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("\n************Evaluation*************")
        for batch_idx, (batch) in enumerate(loader):
            with torch.no_grad():
                imgs = batch['sketch_img']
                seqs = batch['sketch_points']
                labels = batch['sketch_label']
                if (len(hp.gpus) > 0):
                    batch_imgs, batch_seqs, batch_labels = imgs.cuda(), seqs.cuda(), labels.cuda()
                predicted, img_logsoftmax, seq_logsoftmax, cv_important = self.model(batch_imgs, batch_seqs)
                loss, mix_loss, img_loss, seq_loss = self.myloss(predicted, img_logsoftmax, seq_logsoftmax, batch_labels)
                cv_important = cv_important.mean()
                loss = loss + hp.cv_weight * cv_important
                loss = loss.detach().cpu().numpy()
                mix_loss = mix_loss.detach().cpu().numpy()
                img_loss = img_loss.detach().cpu().numpy()
                seq_loss = seq_loss.detach().cpu().numpy()
                losses.update(loss.item(), batch_imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), batch_imgs.size(0))
                top5.update(err5.item(), batch_imgs.size(0))

        return top1.avg, top5.avg, losses.avg

    def myloss(self,predicted, img_ls, seq_ls, labels):
        mix_loss = self.loss_f(predicted, labels)
        img_loss = self.loss_f(img_ls, labels)
        seq_loss = self.loss_f(seq_ls, labels)
        loss = hp.mix_weight * mix_loss + hp.img_weight * img_loss + hp.seq_weight * seq_loss
        return loss, mix_loss, img_loss, seq_loss

    def run(self, train_loder, val_loder, test_loder):
        best_err1, best_err5 = 100, 100
        start_epoch = hp.start_epoch
        top_score = np.ones([5, 3], dtype=float) * 100
        top_score5 = np.ones(5, dtype=float) * 100
        if start_epoch!=0:
            print("-------model loading-----------")
            model_path = hp.model_path
            model, optimizer, start_epoch = load_checkpoint(self.model,self.optimizer,model_path)
        for e in range(hp.epochs):
            e = e + start_epoch + 1
            print("------model:{}----Epoch: {}--------".format(hp.model_name, e))
            #torch.cuda.empty_cache()
            _, _, train_loss = self.train_epoch(train_loder, e)
            err1, err5, val_loss = self.valid_epoch(val_loder, e)
            test_err1, test_err5, test_loss = self.valid_epoch(test_loder, e)
            print("\nval_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(val_loss, err1, err5))
            if err1 <= best_err1:
                best_err1 = err1
                print('Current Best (top-1 error):', best_err1)
            if err5 <= best_err5:
                best_err5 = err5
                print('Current Best (top-5 error):', best_err5)

            if err1 < top_score[4][2]:
                top_score[4] = [e, val_loss, err1]
                z = np.argsort(top_score[:, 2])
                top_score = top_score[z]
                best_err1 = save_checkpoint(self.model, self.optimizer, e, val_loss=err1, check_loss=best_err1,
                                            savepath=hp.model_save, m_name=hp.model_name)
            if err5 < top_score5[4]:
                top_score5[4] = err5
                z = np.argsort(top_score5)
                top_score5 = top_score5[z]

            writer.add_scalar('training loss', train_loss, e)
            writer.add_scalar('valing loss', val_loss, e)
            writer.add_scalar('valid_err1', err1, e)
            writer.add_scalar('valid_err5', err5, e)
            writer.add_scalar('test_err1', test_err1, e)
            writer.add_scalar('test_err5', test_err5, e)

        writer.close()
        print('\nbest score:{}'.format(hp.model_name))
        for i in range(5):
            print(top_score[i])
        print(top_score5, top_score[:, 0])
        print('Best(top-1 and 5 error):', top_score[:, 1].mean(), best_err1, best_err5)

        print("best accuracy:\n avg_acc1:{:.4f} | best_acc1:{:.4f} | avg_acc5:{:.4f} | | best_acc5:{:.4f} ".
              format(100 - top_score[:, 2].mean(), 100 - best_err1, 100 - top_score5.mean(), 100 - best_err5))


print('''***********- training -*************''')
params_total = sum(p.numel() for p in model.parameters())
print("Number of parameter: %.2fM"%(params_total/1e6))
Trainer = trainer(loss_f, model, optimizer, scheduler)
Trainer.run(dataloader_Train, dataloader_Valid, dataloader_Test)
