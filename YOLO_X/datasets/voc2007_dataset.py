import tensorflow as tf
from datasets.mosaic_with_mixup import MosaicDetection
from datasets.data_augment import TrainTransform,ValTransform
import numpy as np

class VOCDataset():
    def __init__(self,input_size,index_list,batch_size,epochs,num_samples,preproc,
                 enable_mixup = True,enable_mosiac=True,detect=False):

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = enable_mixup
        self.enable_mosiac=enable_mosiac
        self.detect=detect

        self.input_size = input_size  # [height, width]
        self.batch_size = batch_size
        self.epochs=epochs
        self.index_list=index_list
        self._mosaic = MosaicDetection(self.input_size,num_samples=num_samples,
                                       preproc=preproc,
                                       degrees=self.degrees,
                                       translate=self.translate,
                                       scale=self.scale,
                                       mscale=self.mscale,
                                       shear=self.shear,
                                       perspective=self.perspective,
                                       enable_mosaic=self.enable_mosiac,
                                       enable_mixup=self.enable_mixup,
                                       )


    def data_argument(self,index):
        image, label,img_info,idx =self._mosaic(int(index))
        return image,label,img_info,idx
    def map_func(self,index,):
        h,w=self.input_size
        image, label,img_info,idx = tf.py_function(self.data_argument,[index],[tf.float32,tf.float32,tf.float32,tf.float32])
        image.set_shape((h,w,3))
        label.set_shape((50,5))
        if self.detect:
            return image,label,img_info,idx
        else:
            return image,label
    def get_dataset(self,):
        dataset = tf.data.Dataset.from_tensor_slices(self.index_list) \
            .shuffle(len(self.index_list)) \
            .map(self.map_func)\
            .batch(self.batch_size) \
            .prefetch(self.batch_size) \

        return dataset


if __name__=="__main__":
    # 划分训练集、验证集
    np.random.seed(10101)
    list_samples = np.random.permutation(range(9963))
    np.random.seed(None)
    num_val = int(9963 * 0.1)
    num_train = 9963 - num_val
    train_idx_list = list_samples[0:num_train]
    val_idx_list = list_samples[num_train:]
    Dataset=VOCDataset(input_size=(640,640,),
                       index_list=val_idx_list,
                       batch_size=2,epochs=300,
                       num_samples=9963,
                       preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225),
                                             ),
                       enable_mosiac=False,
                       detect=False
                       )
    dataset=Dataset.get_dataset()
    print(len(dataset))
    for item in dataset:
       print(item)