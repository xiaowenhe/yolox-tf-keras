import tensorflow.keras.backend as K

def boxes_iou(b1,b2):
    # 计算左上角的坐标和右下角的坐标
    b1=K.expand_dims(b1,-2)
    b1_xy=b1[...,:2]
    b1_wh=b1[...,2:4]
    b1_mins=b1_xy-b1_wh/2.
    b1_maxes=b1_xy+b1_wh/2.
    # 1,n,4
    # 计算左上角和右下角的坐标
    b2=K.expand_dims(b2,0)
    b2_xy=b2[...,0:2]
    b2_wh=b2[...,2:4]
    b2_mins=b2_xy-b2_wh/2.
    b2_maxes=b2_xy+b2_wh/2.

    intersect_mins=K.maximum(b1_mins,b2_mins)
    intersect_maxes=K.minimum(b1_maxes,b2_maxes)
    intersect_wh=K.maximum(intersect_maxes-intersect_mins,0.)
    intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
    b1_area=b1_wh[...,0]*b1_wh[...,1]
    b2_area=b2_wh[...,0]*b2_wh[...,1]
    iou=intersect_area/(b1_area+b2_area-intersect_area)
    return iou