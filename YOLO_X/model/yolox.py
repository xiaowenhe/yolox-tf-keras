from model.yolo_head import YOLOXHead
from model.yolo_pafpn import YOLOPAFPN
from model.CSPdarknet import CSPDarknet
from tensorflow.keras import Model
import tensorflow as tf
from model.yolo_loss import YOLOXLoss
class YOLOX(Model):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, num_classes=80,backbone=None, head=None,act="silu"):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN(act=act)
        if head is None:
            head = YOLOXHead(num_classes,act=act)

        self.backbone = backbone
        self.head = head
        self.yolo_loss=YOLOXLoss(num_classes=num_classes)

    def call(self,inputs,**kwargs):
        # fpn output content features of [dark3, dark4, dark5]
        x=self.backbone(inputs)
        outputs=self.head(x)
        return outputs

    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     x, y=data
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss = self.yolo_loss(y_pred,y)
    #         # loss += self.losses
    #
    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #
    #     # Compute our own metrics
    #     return {m.name: m.result() for m in self.metrics}

if __name__=="__main__":
    from tensorflow.keras import Input
    from tensorflow.keras import Model
    backbone = YOLOPAFPN(depth=0.67, width=0.75)
    head = YOLOXHead(num_classes=80, width=0.75)
    yolo = YOLOX(num_classes=80,backbone=backbone, head=head)
    h, w = (640,640)
    inputs = Input(shape=(h, w,3))
    # outputs=backbone(inputs)
    # model=Model(inputs,outputs)
    # model.summary()
    labels = Input(shape=( 50, 6))
    outputs = yolo(inputs)
    model=Model(inputs,outputs)
    model.summary()