import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Concatenate

from model.yolo_v4.layer import MyConv2D,CSPStage, SPP, DownSampling, UpSampling,Conv5Block
from model.yolo_head import YOLOXHead

class CSPDarknet53(Layer):
    def __init__(self,name="CSPDarknet53",**kwargs):
        super(CSPDarknet53, self).__init__(name=name,**kwargs)
        self.conv=MyConv2D(filters=32,kernel_size=3,activation="mish",apply_dropblock=True)
        self.stages=[
            CSPStage(filters=[32,64],num_blocks=1),
            CSPStage(filters=[64, 128], num_blocks=2),
            CSPStage(filters=[128, 256], num_blocks=8),
            CSPStage(filters=[256, 512], num_blocks=8),
            CSPStage(filters=[512, 1024], num_blocks=4),
        ]

    def call(self,inputs,training=False,**kwargs):
        x=self.conv(inputs,training=training)
        x=self.stages[0](x,training=training)
        x=self.stages[1](x,training=training)
        x=self.stages[2](x,training=training)
        output_large=x
        x=self.stages[3](x,training=training)
        output_medium=x
        x = self.stages[4](x, training=training)
        output_small=x

        return output_small,output_medium,output_large

class PANet(Layer):
    def __init__(self,num_classes,name="PANet",**kwargs):
        super(PANet, self).__init__(name=name,**kwargs)
        self.num_classes=num_classes
        self.concat=Concatenate()
        self.block_1=Sequential([
            MyConv2D(filters=512,kernel_size=1,apply_dropblock=True, name="pa_net_block1_conv1"),
            MyConv2D(filters=1024, kernel_size=3, apply_dropblock=True, name="pa_net_block1_conv2"),
            MyConv2D(filters=512, kernel_size=1, apply_dropblock=True, name="pa_net_block1_conv3"),
        ])
        self.spp=SPP()
        self.block_2=Sequential([
            MyConv2D(filters=512, kernel_size=1, apply_dropblock=True, name="pa_net_block2_conv1"),
            MyConv2D(filters=1024, kernel_size=3, apply_dropblock=True, name="pa_net_block2_conv2"),
            MyConv2D(filters=512, kernel_size=1, apply_dropblock=True, name="pa_net_block2_conv3"),
        ])
        self.up_sampling_1=UpSampling(filters=256,apply_dropblock=True)
        self.medium_entry_conv=MyConv2D(filters=256,kernel_size=1,apply_dropblock=True)
        self.block_3=Conv5Block(filters=256,index_block=3,apply_dropblock=True)

        self.up_sampling_2=UpSampling(filters=128,apply_dropblock=True)
        self.large_entry_conv=MyConv2D(filters=128,kernel_size=1,apply_dropblock=True)
        self.block_4 = Conv5Block(filters=128,index_block=4, apply_dropblock=True)

        # self.large_out_conv=Sequential([MyConv2D(filters=256, kernel_size=3),
        #                 MyConv2D(filters=3 * (self.num_classes + 5), kernel_size=1, activation="linear",
        #                                apply_batchnorm=False, apply_dropblock=False,name="large_output_conv" )])
        self.down_sampling_1=DownSampling(filters=256,apply_dropblock=True)
        self.block_5 = Conv5Block(filters=256, index_block=5,apply_dropblock=True)

        # self.medium_out_conv =Sequential([MyConv2D(filters=512, kernel_size=3),
        #                 MyConv2D(filters=3 * (self.num_classes + 5), kernel_size=1, activation="linear",
        #                                apply_batchnorm=False, apply_dropblock=False,name="medium_output_conv" )])
        self.down_sampling_2 = DownSampling(filters=512, apply_dropblock=True)
        self.block_6 = Conv5Block(filters=512, index_block=6,apply_dropblock=True)

        # self.small_out_conv = Sequential([MyConv2D(filters=1024, kernel_size=3),
        #                                    MyConv2D(filters=3 * (self.num_classes + 5), kernel_size=1,
        #                                             activation="linear",
        #                                             apply_batchnorm=False, apply_dropblock=False,name="small_output_conv" )])

    def call(self,inputs,training=False,**kwargs):
        input_small,input_medium,input_large=inputs

        input_small=self.block_1(input_small,training=training)
        input_small=self.spp(input_small,training=training)
        input_small = self.block_2(input_small, training=training)

        up_small=self.up_sampling_1(input_small,training=training)
        input_medium=self.medium_entry_conv(input_medium,training=training)
        input_medium=self.concat([up_small,input_medium])
        input_medium=self.block_3(input_medium,training=training)

        up_medium=self.up_sampling_2(input_medium,training=training)
        input_large=self.large_entry_conv(input_large,training=training)
        input_large=self.concat([up_medium,input_large])
        input_large=self.block_4(input_large,training=training)

        # output_large=self.large_out_conv(input_large,training=training)

        down_large=self.down_sampling_1(input_large,training=training)
        input_medium = self.concat([down_large, input_medium])
        input_medium = self.block_5(input_medium,training=training)

        # output_medium = self.medium_out_conv(input_medium,training=training)

        down_medium=self.down_sampling_2(input_medium,training=training)
        input_small = self.concat([down_medium,input_small])
        input_small = self.block_6(input_small,training=training)

        # output_small = self.small_out_conv(input_small,training=training)

        return input_small,input_medium,input_large


class YOLOv4(tf.keras.layers.Layer):
    def __init__(self,num_classes,name="yolov4",**kwargs):
        super(YOLOv4, self).__init__(name=name,**kwargs)
        self.backbone=CSPDarknet53()
        self.yolobody=PANet(num_classes=num_classes)
        self.yolo_head=YOLOXHead(num_classes)
    def call(self,inputs,training=False,**kwargs):
        x=self.backbone(inputs,training=training)
        x=self.yolobody(x,training=training)
        x=self.yolo_head(x)


        return x


if __name__=="__main__":
    from tensorflow.keras import Input
    from tensorflow.keras import Model
    yolo=YOLOv4(20)
    h, w = (416,416)
    inputs = Input(shape=(h, w,3))
    # labels = Input(shape=(None, 50, None))
    out_puts=yolo(inputs)
    print(out_puts)
    model=Model(inputs,out_puts)
    model.summary()