import math
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from utils.iou_losses import IOUloss
from utils.iou import boxes_iou

from tensorflow.keras.layers import Concatenate

class YOLOXLoss():
    def __init__(self,
                 num_classes, strides=(8, 16, 32),
                 filters=(256, 512, 1024),
                 ):
        super(YOLOXLoss, self).__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        # self.decode_in_inference = True  # for deploy, set to False

        self.use_l1 = False
        self.l1_loss = tf.keras.regularizers.L1()
        self.iou_loss = IOUloss()
        self.strides = strides
        self.grids = [np.zeros(1)] * len(filters)
        self.expanded_strides = [None] * len(filters)
        self.BEC=tf.keras.losses.BinaryCrossentropy(reduction="none",from_logits=False)
    def __call__(self, y_true, y_pred):
        # inputs:[
        #   [b,w / 8,h / 8 ,num_classes+5],
        #   [b, w / 16, h / 16, num_classes+5],
        #   [b, w / 32, h / 32,num_classes+5],
        # ]
        # labels:[b,50,5/mix_up=6]
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (stride_this_level, input) in enumerate(zip(self.strides, y_pred)):

            # output:(b,w*h,5+num_classes) grid:(1,w*h,2)
            output,grid=self.get_output_and_grid(input,k,stride_this_level,y_pred[0].dtype)

            x_shifts.append(grid[:,:,0])
            y_shifts.append(grid[:,:,1])
            expanded_strides.append(
                tf.cast(np.full((1, grid.shape[1]),stride_this_level),y_pred[0].dtype)
            )
            if self.use_l1:
                reg_output=input[...,0:4]
                batch_size=reg_output.shape[0]
                hsize,wsize=reg_output.shape[1:3]
                reg_output=K.reshape(reg_output,[batch_size,self.n_anchors * hsize * wsize,-1])
                origin_preds.append(reg_output)
            outputs.append(output)

        loss,iou,obj,cls,l1,num= self.get_losses(x_shifts,y_shifts,expanded_strides,y_true,
                               Concatenate(1)(outputs),origin_preds,dtype=y_pred[0].dtype
            )
        # print(loss,obj,iou,cls)
        return loss
    def get_output_and_grid(self,output,k,stride,dtype):
        # output:(b,h,w,c)

        grid=self.grids[k]
        batch_size=output.shape[0]
        hsize,wsize=output.shape[1:3]
        if grid.shape[1:3]!=output.shape[1:3]:

            grid_y = K.tile(K.reshape(K.arange(0, stop=hsize), [1, -1, 1, 1]),
                            [1,1, wsize, 1])
            grid_x = K.tile(K.reshape(K.arange(0, stop=wsize), [1, 1,-1, 1]),
                            [1,hsize, 1, 1])
            grid = K.concatenate([grid_x, grid_y])
            grid = K.cast(grid,dtype)
            self.grids[k] = grid
        output=K.reshape(output,[batch_size,self.n_anchors * hsize * wsize,-1])
        grid=K.reshape(grid,[1,-1,2])
        xy = (output[..., :2] + grid) * stride
        wh = K.exp(output[..., 2:4]) * stride
        obj_cls=output[..., 4:]
        output=Concatenate()([xy,wh,obj_cls])
        return output, grid

    def get_losses(self,x_shifts,y_shifts,expanded_strides,labels,outputs,origin_preds,dtype):
        # [batch, n_anchors_all, 4] n_anchors_all:所有layer，n_anchor*wsize*hsize之和
        bbox_preds=outputs[:,:,:4]
        # [batch, n_anchors_all, 1]
        obj_preds=outputs[:,:,4:5]
        # [batch, n_anchors_all, n_cls]
        cls_preds=outputs[:,:,5:]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = K.sum(tf.cast(K.sum(label_cut,axis=2) > 0,tf.float32),axis=1)  # number of objects
        # n_anchors_all
        total_num_anchors=outputs.shape[1]
        # (1,n_anchors_all)
        x_shifts=Concatenate(1)(x_shifts)
        y_shifts=Concatenate(1)(y_shifts)
        # (1,n_anchors_all)
        expanded_strides=Concatenate(1)(expanded_strides)
        if self.use_l1:
            # (b,n_anchors_all,4)
            origin_preds=Concatenate(1)(origin_preds)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0
        num_gts = 0.0

        # 对batch每一张图片处理
        for batch_idx in range(outputs.shape[0]):
            num_gt=(nlabel[batch_idx])
            num_gts+=num_gt
            num_gt=int(num_gt)
            if num_gt==0:
                cls_target=K.cast(np.zeros((0,self.num_classes)),dtype)
                reg_target=K.cast(np.zeros((0,4)),dtype)
                l1_target = K.cast(np.zeros((0, 4)), dtype)
                obj_target = K.cast(np.zeros((total_num_anchors,1)), dtype)
                fg_mask = K.cast(np.zeros(total_num_anchors), bool)

            else:
                gt_bboxes_per_image=labels[batch_idx,:num_gt,1:5]
                gt_classes=labels[batch_idx,:num_gt,0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                    cls_preds, bbox_preds, obj_preds, dtype
                )

                num_fg+=num_fg_img
                # (num_fg,num_classes)
                cls_target=tf.one_hot(tf.cast(gt_matched_classes,'int32'),self.num_classes)*tf.expand_dims(pred_ious_this_matching,axis=-1)
                # (all_anchors,1)
                obj_target=tf.expand_dims(fg_mask,axis=-1)
                # (num_fg,4)
                reg_target=tf.convert_to_tensor([gt_bboxes_per_image[ids] for ids in matched_gt_inds])
                if self.use_l1:
                    l1_target=self.get_l1_target(
                        K.cast(np.zeros((num_fg_img,4)), dtype),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask]
                    )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(tf.cast(obj_target,dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = Concatenate(0)(cls_targets)
        reg_targets = Concatenate(0)(reg_targets)
        obj_targets = Concatenate(0)(obj_targets)
        fg_masks = Concatenate(0)(fg_masks)
        if self.use_l1:
            l1_targets = Concatenate(0)(l1_targets)

        num_fg=K.maximum(num_fg,1)
        reg_fg=K.reshape(bbox_preds,[-1,4])[fg_masks]
        loss_iou=K.sum(self.iou_loss(reg_fg,reg_targets))/num_fg
        loss_obj=K.sum(K.binary_crossentropy(obj_targets,K.reshape(obj_preds,[-1,1]),from_logits=True))/num_fg
        loss_cls=K.sum(K.binary_crossentropy(cls_targets,K.reshape(cls_preds,[-1,self.num_classes])[fg_masks],from_logits=True))/num_fg
        if self.use_l1:
            loss_l1 = K.sum(self.l1_loss(K.reshape(origin_preds,[-1, 4])[fg_masks], l1_targets))/ num_fg
        else:
            loss_l1 = 0.0
        reg_weight=5.0
        loss=reg_weight*loss_iou+loss_obj+loss_cls+loss_l1
        # print("total_loss:%.4f,iou_loss:%.4f,obj_loss:%.4f,cls_loss:%.4f"%(loss,loss_iou,loss_obj,loss_cls))
        return loss,reg_weight*loss_iou,loss_obj,loss_cls,loss_l1,num_fg/max(num_gts,1)

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = K.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = K.log(gt[:, 3] / stride + eps)
        return l1_target

    def get_assignments(self,batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
        cls_preds, bbox_preds, obj_preds,dtype):
        """

        :param batch_idx:
        :param num_gt:
        :param total_num_anchors:
        :param gt_bboxes_per_image:(num_gt,4)
        :param gt_classes:(num_gt,1)
        :param bboxes_preds_per_image:(n_anchors_all,4)
        :param expanded_strides:(1,n_anchors_all)
        :param x_shifts:(1,n_anchors_all)
        :param y_shifts:(1,n_anchors_all)
        :param cls_preds:(b,n_anchors_all,num_classes)
        :param bbox_preds:(b,n_anchors_all,4)
        :param obj_preds:(b,n_anchors_all,1)
        :param dtype:
        :return:
        """
        fg_mask,is_in_boxes_and_center=self.get_in_boxes_info(
            gt_bboxes_per_image,expanded_strides,x_shifts,y_shifts,total_num_anchors,num_gt
        )
        # (num_in_boxes_anchor,4)
        bboxes_preds_per_image=tf.boolean_mask(bboxes_preds_per_image,fg_mask)
        # (num_in_boxes_anchor,num_classes)
        cls_preds_=tf.boolean_mask(cls_preds[batch_idx],fg_mask)
        # (num_in_boxes_anchor, 1)
        obj_preds_=tf.boolean_mask(obj_preds[batch_idx],fg_mask)
        num_in_boxes_anchor=bboxes_preds_per_image.shape[0]
        # (num_gt,num_in_boxes_anchor)
        pair_wise_ious = boxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image
        )
        # print(pair_wise_ious)
        gt_cls_per_image=tf.cast(tf.one_hot(tf.cast(gt_classes,'int32'),self.num_classes),dtype)
        # (num_gt,num_in_boxes_anchor,num_classes)
        gt_cls_per_image=tf.tile(tf.expand_dims(gt_cls_per_image,axis=1),(1,num_in_boxes_anchor,1))
        pair_wise_ious_loss= -K.log(pair_wise_ious + 1e-8)
        # (num_gt,num_in_boxes_anchor,num_classes)
        cls_preds_=tf.tile(tf.expand_dims(cls_preds_,axis=0),(num_gt,1,1))
        # (num_gt,num_in_boxes_anchor,1)
        obj_preds_=tf.tile(tf.expand_dims(obj_preds_,axis=0),(num_gt,1,1))
        cls_preds_=tf.sigmoid(cls_preds_)*tf.sigmoid(obj_preds_)
        # (num_gt,num_in_boxes_anchor)
        pair_wise_cls_loss=self.BEC(gt_cls_per_image,tf.math.sqrt(cls_preds_))*self.num_classes
        del cls_preds_
        # (num_gt,num_in_boxes_anchor)
        cost=(pair_wise_cls_loss+3.0*pair_wise_ious_loss+100000.0*tf.cast((~is_in_boxes_and_center),tf.float32))
        num_fg,gt_matched_classes,pred_ious_this_matching,matched_gt_inds,fg_mask= self.dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        return gt_matched_classes,fg_mask,pred_ious_this_matching,matched_gt_inds,num_fg


    def get_in_boxes_info(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
    ):
        """

        :param gt_bboxes_per_image: (num_gt,4)
        :param expanded_strides: (1,n_anchors_all)
        :param x_shifts: (1,n_anchors_all)
        :param y_shifts: (1,n_anchors_all)
        :param total_num_anchors: n_anchors_all
        :param num_gt:
        :return:
        """
        expanded_strides_per_image = expanded_strides[0]
        # (n_anchors_all)
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        # 取每个网格的中心点坐标
        x_centers_per_image = tf.tile(
            tf.expand_dims(x_shifts_per_image + 0.5 * expanded_strides_per_image,axis=0),
            (num_gt,1)
        )  # [n_anchor] -> [n_gt, n_anchor_all]

        y_centers_per_image = tf.tile(
            tf.expand_dims(y_shifts_per_image + 0.5 * expanded_strides_per_image,axis=0),
            (num_gt, 1)
        )
        # 取每个gt_bboxes 左右上下边界
        # （n_gt,n_anchor_all)
        gt_bboxes_per_image_l = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2],axis=1),
            (1, total_num_anchors)
        )
        gt_bboxes_per_image_r = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2],axis=1),
            (1, total_num_anchors)
        )
        gt_bboxes_per_image_t = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3],axis=1),
            (1, total_num_anchors)
        )
        gt_bboxes_per_image_b = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3],axis=1),
            (1, total_num_anchors)
        )
        # 每个网格中心与n_gt各边界之差
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # (4,n_gt,n_anchors_all) > (n_gt,n_anchor_all,4)
        bbox_deltas = K.stack([b_l, b_t, b_r, b_b], 2)
        # 差值都为正说明网格中心位于gt_bbox内 (n_gt,n_anchor_all)
        is_in_boxes = K.min(bbox_deltas,axis=-1) > 0.0
        # 计算每个网格中包含gt_bbox的个数 (n_anchor_all)
        is_in_boxes_all = K.sum(tf.cast(is_in_boxes,tf.float32),axis=0)>0
        # in fixed center

        center_radius = 2.5
        # 每个gt的cx与cy向外扩展2.5 * expanded_strides距离
        gt_bboxes_per_image_l = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 0],axis=1),(1, total_num_anchors))\
                                - center_radius *tf.expand_dims( expanded_strides_per_image,axis=0)
        gt_bboxes_per_image_r = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 0],axis=1),(1, total_num_anchors))\
                                 + center_radius * tf.expand_dims(expanded_strides_per_image,axis=0)
        gt_bboxes_per_image_t = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 1],axis=1),(1, total_num_anchors))\
                                - center_radius * tf.expand_dims(expanded_strides_per_image,axis=0)
        gt_bboxes_per_image_b = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 1],axis=1),(1, total_num_anchors)) \
                                + center_radius *tf.expand_dims( expanded_strides_per_image,axis=0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = K.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = K.min(center_deltas,axis=-1)>0.0
        is_in_centers_all = K.sum(tf.cast(is_in_centers,tf.float32),axis=0)>0

        # in boxes and in centers
        # 在box或者在center,(n_anchors_all)
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        # shape:[num_gt, num_in_boxes_anchor]，注意：这里是每一个gt与每一个候选区域的关系
        #     # 这里一个anchor可能与多个gt存在候选关系
        is_in_boxes_and_center = tf.boolean_mask(is_in_boxes, is_in_boxes_anchor,axis=1) \
                                 & tf.boolean_mask(is_in_centers, is_in_boxes_anchor,axis=1)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        :param cost:  (num_gt,num_in_boxes_anchor)
        :param pair_wise_ious:  (num_gt,num_in_boxes_anchor)
        :param gt_classes: (num_gt,1)
        :param num_gt:
        :param fg_mask: (n_anchors_all)
        :return:num_fg,正样本个数
                gt_matched_classes,
                pred_ious_this_matching, 正样本iou
                matched_gt_inds, 候选区域内正样本gt_index :(num_fg)
        """

        matching_matrix = np.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = 10
        topk_ious, _ = tf.nn.top_k(ious_in_boxes_matrix, n_candidate_k)
        dynamic_ks = K.maximum(K.sum(topk_ious, axis=1),1.0)
        for gt_idx in range(num_gt):
            _, pos_idx = tf.nn.top_k(
                -cost[gt_idx], k=int(dynamic_ks[gt_idx])
            )
            for item in pos_idx:
                matching_matrix[gt_idx][item] = 1.0

        del topk_ious, dynamic_ks, pos_idx
        # (num_in_boxes_anchor)
        anchor_matching_gt = matching_matrix.sum(axis=0)
        if (anchor_matching_gt > 1).sum() > 0:

            cost_argmin = K.argmin(tf.boolean_mask(cost, anchor_matching_gt > 1,axis=1), axis=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        # 候选区域内正样本mask：(num_in_boxes_anchor)
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = K.sum(tf.cast(tf.convert_to_tensor(fg_mask_inboxes),tf.float32))
        #  (n_anchors_all)
        fg_mask=fg_mask.numpy()
        fg_mask[fg_mask] = fg_mask_inboxes
        fg_mask=tf.convert_to_tensor(fg_mask)

        # 候选区域内正样本gt_index,classes,iou   shape:(num_fg,)
        matched_gt_inds = K.argmax(matching_matrix[:, fg_mask_inboxes],axis=0)
        gt_matched_classes = tf.convert_to_tensor([gt_classes[ids] for ids in matched_gt_inds])
        pred_ious_this_matching =tf.boolean_mask(K.sum(matching_matrix * pair_wise_ious,axis=0),fg_mask_inboxes)

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds,fg_mask



if __name__=="__main__":
    y_head=YOLOXLoss(80)
    input1=tf.ones((1, 52, 52, 85),)*0.5
    input2 = tf.ones((1, 26, 26, 85), ) * 0.6
    input3 = tf.ones((1, 13, 13, 85), ) * 0.7
    inputs=[input1,input2,input3]
    labels=tf.constant([[[4.0,25.0,120.0,14.4,25.3],
                        [2.0,75.0,250.0,21.4,8.5],
                        [12.0,340.0,59.0,4.6,18.0]]],dtype=tf.float32
                       )
    img=tf.random.uniform((1,416,416,3),0,1,dtype=tf.float32)
    loss=y_head(labels,inputs)
    print(loss)
