import tensorflow as tf
import tensorflow.keras.backend as K

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = tf.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = K.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = tf.squeeze(image_pred[:, 4] * tf.squeeze(class_conf) >= conf_thre)
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = tf.concat([image_pred[:, :5], class_conf, class_pred.float()], 1)
        detections = tf.boolean_mask(detections,conf_mask)
        if not detections.size(0):
            continue

        nms_out_index = tf.image.non_max_suppression(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = tf.concat([output[i], detections])

    return output