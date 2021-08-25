import math
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from utils.iou_losses import IOUloss
from utils.iou import boxes_iou
from model.layers import BaseConv, DWConv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer,Conv2D,Concatenate

class YOLOXHead(Layer):
    def __init__(self,
                 num_classes, width=1.0, strides=(8, 16, 32), act="silu", depthwise=False,prior_prob=1e-2
                 ):
        super(YOLOXHead, self).__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.stems = []
        self.cls_convs = []
        self.reg_convs = []
        self.cls_preds = []
        self.reg_preds = []
        self.obj_preds = []
        self.bias=-tf.math.log((1-prior_prob)/prior_prob)
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(strides)):
            self.stems.append(
                BaseConv(
                       filters=int(256 * width),
                        kernel_size=1,
                        strides=1,
                        act=act,
                )
            )
            self.cls_convs.append(
                Sequential(
                [
                    Conv(filters=int(256 * width),
                        kernel_size=3,
                        strides=1,
                        act=act,
                    ),
                    Conv(filters=int(256 * width),
                        kernel_size=3,
                        strides=1,
                        act=act,
                    ),
                ]
                )
            )

            self.reg_convs.append(
                Sequential(
                    [
                        Conv(
                            filters=int(256 * width),
                            kernel_size=3,
                            strides=1,
                            act=act,
                        ),
                        Conv(
                            filters=int(256 * width),
                            kernel_size=3,
                            strides=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(Conv2D(
                    filters=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    bias_initializer=tf.keras.initializers.constant(self.bias)
                ))

            self.reg_preds.append(
                Conv2D(
                  filters=4,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                )
            )
            self.obj_preds.append(
                Conv2D(
                   filters=self.n_anchors * 1,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    bias_initializer=tf.keras.initializers.constant(self.bias)
                )
            )



    def call(self, inputs, *args, **kwargs):
        out_puts=[]
        for k,x in enumerate(inputs):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = self.cls_convs[k](cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            out_put=Concatenate(-1)([reg_output,obj_output,cls_output])
            out_puts.append(out_put)

        return out_puts
