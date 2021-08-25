#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from model.yolox import YOLOX
import sys
import tempfile
import time
from loguru import logger
# from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from datasets.data_augment import ValTransform
from datasets.voc2007_dataset import VOCDataset
from evaluators.voc import VOCDetection
from utils.postprocess_nms import postprocess

class VOCEvaluator:
    """
    VOC AP Evaluation class.
    """

    def __init__(
        self, img_size,batch_size, confthre, nmsthre, num_classes,model_weight_path,rootpath,test_path
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """

        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_samples=9963
        self.batch_size=batch_size
        self.model_weight_path=model_weight_path
        self.model=YOLOX(num_classes)
        self.rootpath=rootpath
        self.test_path=test_path
        # self.test_path="C:/Users/xiongdada/PycharmProjects/TF_DL/YOLO_X/datasets/data/test.txt"
        self.VOCdetect=VOCDetection(self.rootpath,self.test_path)

    def dataloader(self, file_path):
        idx_list = []
        with open(file_path, "r")as train_file:
            for x in train_file.readlines():
                idx_list.append(int(x.strip()))
        self.num_images=len(idx_list)
        Dataset = VOCDataset(input_size=self.img_size,
                             index_list=idx_list,
                             batch_size=self.batch_size,
                             epochs=1,
                             num_samples=self.num_samples,
                             preproc=ValTransform(rgb_means=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225),
                                                  ),
                             enable_mosiac=False,
                             detect=True
                             )
        dataset = Dataset.get_dataset()
        return dataset
    def decode_output(self,outputs):
        hw = [x.shape[1:3] for x in outputs]
        dtype=outputs[0].dtype
        for x in range(len(hw)):
            outputs[x] = K.reshape(outputs, [self.batch_size,  hw[x][0]*hw[x][1], -1])
        outputs=tf.concat(outputs,axis=1)
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw, [8,16,32]):
            grid_y = K.tile(K.reshape(K.arange(0, stop=hsize), [1, -1, 1, 1]),
                            [1, 1, wsize, 1])
            grid_x = K.tile(K.reshape(K.arange(0, stop=wsize), [1, 1, -1, 1]),
                            [1, hsize, 1, 1])
            grid = K.concatenate([grid_x, grid_y])
            grid = K.cast(grid, tf.float32)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(tf.cast(np.full((*shape, 1), stride),dtype))

        grids = tf.cast(tf.concat(grids, dim=1),dtype)
        strides = tf.cast(tf.concat(strides, dim=1),dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = K.exp(outputs[..., 2:4]) * strides
        return outputs


    def evaluate(self,):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        data_generate=self.dataloader(self.test_path)
        self.model.load_weights(self.model_weight_path)

        ids = []
        data_list = {}
        inference_time = 0
        nms_time = 0
        n_samples = len(data_generate) - 1

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(data_generate):
            # skip the the last iters since batchsize might be not enough for batch inference
            is_time_record = cur_iter < len(data_generate) - 1
            if is_time_record:
                start = time.time()

            outputs = self.model(imgs)

            outputs = self.decode_output(outputs)

            if is_time_record:
                infer_end =time.time()
                inference_time += (infer_end - start)

            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            if is_time_record:
                nms_end = time.time()
                nms_time += (nms_end - infer_end)

        data_list.update(self.convert_to_voc_format(outputs, info_imgs, ids))

        statistics = tf.convert_to_tensor([inference_time, nms_time, n_samples])

        eval_results = self.evaluate_prediction(data_list, statistics)
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions

    def evaluate_prediction(self, data_dict, statistics):

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.batch_size)

        time_info = ", ".join(
            ["Average {} time: {:.2f} ms".format(k, v) for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)]
            )]
        )

        info = time_info + "\n"

        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = K.concatenate((bboxes, scores.unsqueeze(1)), axis=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images)
            )
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.VOCdetect.evaluate_detections(all_boxes,tempdir)
            return mAP50, mAP70, info
