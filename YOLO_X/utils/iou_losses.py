import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class IOUloss():
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def __call__(self, b1, b2):
        assert b1.shape[0] == b2.shape[0]
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_mins = b1_xy - b1_wh / 2.
        b1_maxes = b1_xy + b1_wh / 2.
        # 1,n,4
        # 计算左上角和右下角的坐标
        b2_xy = b2[..., 0:2]
        b2_wh = b2[..., 2:4]
        b2_mins = b2_xy - b2_wh / 2.
        b2_maxes = b2_xy + b2_wh / 2.

        intersect_mins = K.maximum(b1_mins, b2_mins)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)


        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            enclose_mins = K.minimum(b1_mins, b2_mins)
            enclose_maxes = K.maximum(b1_maxes, b2_maxes)
            enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
            area_c=enclose_wh[...,0]*enclose_wh[...,1]

            g_iou=iou-(area_c-intersect_area)/K.maximum(area_c,1e-16)
            loss=1-tf.clip_by_value(g_iou,-1.0,1.0)
        if self.reduction == "mean":
            loss = K.mean(loss)
        elif self.reduction == "sum":
            loss = K.sum(loss)
        return loss



if __name__=="__main__":
    a=np.array([4,5,54,12],dtype="float32")
    b=np.array([4,5,12,24],dtype="float32")
    iou=IOUloss()
    loss=iou(a,b)
    print(loss)