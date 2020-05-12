#!/usr/bin/python

import keras.backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TerminateOnNaN
from keras.optimizers import Adam, SGD
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import math
import argparse

from ssd import AnchorBoxes, L2Normalization, ssd_512

from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


def run(train_dir, valid_dir, set_dir):
    # train_dir = arguments.train_dir
    # valid_dir = arguments.valid_dir

    train_dataset_dir = train_dir + '/train/'
    train_annot_dir = train_dir + '/train_annot/'
    train_set = set_dir + 'img_set.txt'

    valid_dataset_dir = valid_dir + '/valid/'
    valid_annot_dir = valid_dir + '/valid_annot/'
    valid_set = set_dir + 'valid_set.txt'

    # Set Training and Validation dataset paths
    batch_size = 32
    model_path = 'COCO_512.h5'
    # model_path = 'saved_model.h5'
    # Needs to know classes and order to map to integers
    classes = ['background','car', 'bus', 'truck','motorbike',
               'train', 'traffic light', 'building', 'person']
    # Set required parameters for training of SSD
    img_height = 512

    img_width = 512
    img_channels = 3  # Colour image
    mean_color = [123, 117, 104]  # DO NOT CHANGE
    swap_channels = [2, 1, 0]  # Original SSD used BGR
    n_classes = 80  # 80 for COCO
    scales_coco = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
    scales = scales_coco
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 128, 256, 512]
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    clip_boxes = False
    variances = [0.1, 0.1, 0.2, 0.2]
    normalize_coords = True
    K.clear_session()

    model = ssd_512(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)
    model.load_weights(model_path, by_name=True)

    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

    # model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
    #                                   'L2Normalization': L2Normalization,

    #                                   'compute_loss': ssd_loss.compute_loss})
    # Create Data Generators for train and valid sets
    train_dataset = DataGenerator(
        load_images_into_memory=False, hdf5_dataset_path=None)
    valid_dataset = DataGenerator(
        load_images_into_memory=False, hdf5_dataset_path=None)
    train_dataset.parse_xml(images_dirs=[train_dataset_dir],
                            image_set_filenames=[train_set],
                            annotations_dirs=[train_annot_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    valid_dataset.parse_xml(images_dirs=[valid_dataset_dir],
                            image_set_filenames=[valid_set],
                            annotations_dirs=[valid_annot_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    # Will speed up trainig but requires more memory
    # Can comment out to avoid memory requirements
    '''
    train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_trainval.h5',
                                      resize=False,
                                      variable_image_size=True,
                                      verbose=True)

    valid_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
                                      resize=False,
                                      variable_image_size=True,
                                      verbose=True)
    '''

    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=[
                                                 ssd_data_augmentation],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = valid_dataset.generate(batch_size=batch_size,
                                           shuffle=False,
                                           transformations=[
                                               convert_to_3_channels, resize],
                                           label_encoder=ssd_input_encoder,
                                           returns={'processed_images',
                                                    'encoded_labels'},
                                           keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    valid_dataset_size = valid_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(
        train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(
        valid_dataset_size))

    model_checkpoint = ModelCheckpoint(filepath='ssd_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    #csv_logger = CSVLogger(filename='ssd512_training_log.csv',
    #                       separator=',',
    #                       append=True)

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                    verbose=1)

    terminate_on_nan = TerminateOnNaN()

    #callbacks = [model_checkpoint,
    #             csv_logger,
    #             learning_rate_scheduler,
    #             terminate_on_nan]

    callbacks = [learning_rate_scheduler,
                 terminate_on_nan]

    initial_epoch = 0
    final_epoch = 30  # 120
    steps_per_epoch = math.ceil(119/32)  # ceil(num_samples/batch_size)

    # Training
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=math.ceil(valid_dataset_size/batch_size),
                                  initial_epoch=initial_epoch)

    # Make predictions
    predict_generator = valid_dataset.generate(batch_size=1,
                                               shuffle=True,
                                               transformations=[convert_to_3_channels,
                                                                resize],
                                               label_encoder=None,
                                               returns={'processed_images',
                                                        'filenames',
                                                        'inverse_transform',
                                                        'original_images',
                                                        'original_labels'},
                                               keep_images_without_gt=False)

    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(
        predict_generator)

    i = 0  # Which batch item to look at

    print("Image:", batch_filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(np.array(batch_original_labels[i]))

    y_pred = model.predict(batch_images)
    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    y_pred_decoded_inv = apply_inverse_transforms(
        y_pred_decoded, batch_inverse_transforms)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded_inv[i])


    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
    classes = ['background',
               'car', 'bus', 'truck', 'motorbike']

    plt.figure(figsize=(20, 12))
    plt.imshow(batch_original_images[i])

    current_axis = plt.gca()

    for box in batch_original_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large',
                          color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

    for box in y_pred_decoded_inv[i]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large',
                          color='white', bbox={'facecolor': color, 'alpha': 1.0})

    plt.show()

    return


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


class SSDLoss:
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.
        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.
        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0),
                           square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.
        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.
        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.
        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.
        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.
        n_boxes = tf.shape(y_pred)[1]

        # 1: Compute the losses for class and box predictions for every box.

        # classification_loss = tf.to_float(self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))  # Output shape: (batch_size, n_boxes)
        classification_loss = tf.cast(self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]), dtype=tf.float32)
        # localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)
        localization_loss = tf.cast(self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]), dtype=tf.float32)

        # 2: Compute the classification losses for the positive and negative targets.

        # Create masks for the positive and negative ground truth classes.
        negatives = y_true[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
        # Tensor of shape (batch_size, n_boxes)
        # positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))
        positives = tf.cast(tf.reduce_max(y_true[:, :, 1:-12], axis=-1), dtype=tf.float32)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        # Tensor of shape (batch_size,)
        pos_class_loss = tf.reduce_sum(
            classification_loss * positives, axis=-1)

        # Compute the classification loss for the negative default boxes (if there are any).

        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * \
            negatives  # Tensor of shape (batch_size, n_boxes)
        # The number of non-zero loss entries in `neg_class_loss_all`
        n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        # n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)
        n_negative_keep = tf.minimum(tf.maximum(
            self.neg_pos_ratio * tf.cast(n_positive, dtype=tf.int32), self.n_neg_min), n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.

        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            # Tensor of shape (batch_size * n_boxes,)
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)  # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(
                                               indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
            # Tensor of shape (batch_size, n_boxes)
            #negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes]))
            negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]), dtype=tf.float32)
            # ...and use it to keep only those boxes and mask all other classification losses
            # Tensor of shape (batch_size,)
            neg_class_loss = tf.reduce_sum(
                classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_class_loss = tf.cond(
            tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        # Tensor of shape (batch_size,)
        class_loss = pos_class_loss + neg_class_loss

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

        # Tensor of shape (batch_size,)
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        # 4: Compute the total loss.

        total_loss = (class_loss + self.alpha * loc_loss) / \
            tf.maximum(1.0, n_positive)  # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        # total_loss = total_loss * tf.to_float(batch_size)
        total_loss = total_loss * tf.cast(batch_size, dtype=tf.float32)

        return total_loss

# uncomment if running in terminal
#if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('train_dir', help='Base directory for training data and annotations.')
    #parser.add_argument('test_dir', help='Base directory for testing/validation data and annotations.')
    #args = parser.parse_args()
    #main(args)
    #run('/Users/justinbutler/Desktop/school/Calgary/Thesis Work/Datasets/aerial-cars-dataset-master/aerial', '/Users/justinbutler/Desktop/school/Calgary/Thesis Work/Datasets/aerial-cars-dataset-master/aerial', '/Users/justinbutler/Desktop/school/Calgary/Thesis Work/Datasets/aerial-cars-dataset-master/aerial/')
    
    