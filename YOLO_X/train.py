import os
import time
from model.yolox import YOLOX
from model.yolo_pafpn import YOLOPAFPN
from model.yolo_head import YOLOXHead
from model.yolo_v4.yolov4 import YOLOv4
from model.CSPdarknet import CSPDarknet
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import numpy as np
from model.yolo_loss import YOLOXLoss
from utils.cosine_decay_lr import cosine_decay_with_warmup
from tensorflow.keras.callbacks import (EarlyStopping,ReduceLROnPlateau,TensorBoard)
from datasets.data_augment import TrainTransform,ValTransform
from datasets.voc2007_dataset import VOCDataset




class Trainer:
    def __init__(self,):
        # ---------------- model config ---------------- #
        # voc2007
        self.num_classes = 20
        # # yolo_x
        # self.depth = 1.33
        # self.width = 1.25
        # yolo_m
        # self.depth = 0.67
        # self.width = 0.75
        # # # yolo_l
        # self.depth = 1.0
        # self.width = 1.0
        # # yolo_s
        self.depth = 0.33
        self.width = 0.50


        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.random_size = (14, 26)
        self.ann_path = "VOC2007/Annotations"
        self.train_idx_path="datasets/data/train.txt"
        self.val_idx_path="datasets/data/val.txt"
        self.log_dir="log/01train"
        self.num_samples=9963
        self.val_split=0.1

        # --------------  training config --------------------- #
        self.batch_size=4
        self.warmup_epochs = 5
        self.total_epoch = 100
        self.warmup_lr = 0.0001
        self.basic_lr_per_img =0.005
        self.no_aug_epochs = 15
        self.enale_aug=False
        self.min_lr_ratio = 1e-4
        self.momentum = 0.9

        # -----------------  model init ------------------ #
        self.backbone=YOLOPAFPN(depth=self.depth,width=self.width)
        self.head=YOLOXHead(num_classes=self.num_classes,width=self.width)
        self.yolo_loss=YOLOXLoss(self.num_classes)

    def data_generate(self,file_path,trainging=True):
        idx_list = []
        with open(file_path, "r")as train_file:
            for x in train_file.readlines():
                idx_list.append(int(x.strip()))
        if trainging:
            Dataset = VOCDataset(input_size=self.input_size,
                                 index_list=idx_list,
                                 batch_size=self.batch_size,
                                 epochs=1,
                                 num_samples=self.num_samples,
                                 preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225),
                                                        max_labels=50, ),
                                 enable_mosiac=self.enale_aug,
                                 enable_mixup=False
                             )
        else:
            Dataset=VOCDataset(input_size=self.input_size,
                                 index_list=idx_list,
                                 batch_size=self.batch_size,
                                 epochs=1,
                                 num_samples=self.num_samples,
                                 preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225),
                                                     ),
                               enable_mosiac=False,
                               enable_mixup=False
                             )


        dataset = Dataset.get_dataset()
        return dataset
    def get_idx_list(self,file_path):
        list=[]
        with open(file_path, "r")as train_file:
            for x in train_file.readlines():
                list.append(int(x.strip()))
        return list

    def train(self):
        # model=YOLOv4(self.num_classes)

        self.model=YOLOX(num_classes=self.num_classes,backbone=self.backbone,head=self.head)
        writer = tf.summary.create_file_writer(self.log_dir)
        warmup_steps = int(self.warmup_epochs * self.num_samples / self.batch_size)
        total_steps = int(self.total_epoch * self.num_samples / self.batch_size)

        global_steps=0
        optimizer = SGD(momentum=self.momentum,nesterov=True,)
        loss_metric_train=tf.keras.metrics.Mean()
        loss_metric_val=tf.keras.metrics.Mean()
        for epoch in range(self.total_epoch):
            loss_metric_train.reset_state()
            loss_metric_val.reset_state()
            print("\nStart of epoch %d" % (epoch,))

            # 余弦退火lr
            global_steps += 1
            lr = cosine_decay_with_warmup(global_steps,
                                          self.basic_lr_per_img,
                                          total_steps,
                                          self.warmup_lr,
                                          warmup_steps,
                                          hold_base_rate_steps=warmup_steps * 3,
                                          min_learn_rate=self.min_lr_ratio)
            optimizer.lr.assign(lr)

            # warmup和最后15epochs，关闭mosiac,mixup
            if epoch>self.warmup_epochs and epoch<=(self.total_epoch-self.no_aug_epochs):
                self.enale_aug=True
            start_time = time.time()

            for step, (image_data_train, label_train) in enumerate(self.data_generate(self.train_idx_path)):
                with tf.GradientTape() as tape:
                    y_pred_train=self.model(image_data_train)
                    loss =self.yolo_loss(label_train,y_pred_train)

                grads=tape.gradient(loss,self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Update training metric.
                loss_metric_train.update_state(loss)

                # Log every 200 batches.
                if step % 100== 0:
                    print(
                        "Training loss at step %d: %.4f"
                        % (step, float(loss))
                    )


                with writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("loss/total_loss", loss, step=global_steps)
                writer.flush()
            print("Training loss over epoch: %.4f" % (float(loss_metric_train.result())))

            for step,(image_data_val,label_val) in enumerate(self.data_generate(self.val_idx_path,trainging=False)):
                global_steps+=1
                y_pred_val=self.model(image_data_val)
                loss_val=self.yolo_loss(label_val,y_pred_val)
                loss_metric_val.update_state(loss_val)

            print("Val loss over epoch: %.4f" % (float(loss_metric_val.result())))
            print("Time taken: %.2fs" % (time.time() - start_time))

            self.model.save_weights(self.log_dir + 'yolox_m.h5')

        self.model.save_weights(self.log_dir + 'yolox_m.h5')



if __name__=="__main__":
    yolox_train=Trainer()
    yolox_train.train()
