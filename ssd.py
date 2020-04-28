import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Activation, Lambda, Input, ZeroPadding2D
from keras.layers import Reshape, Concatenate, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K

from keras.engine.topology import InputSpec, Layer


class L2Normalization(Layer):

    def __init__(self, gamma_init=20, **kwargs):
        if K.common.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1

        self.gamma_init = gamma_init
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super().build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma

    def get_config(self):
        config = {'gamma_init': self.gamma_init}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class AnchorBoxes(Layer):
    def __init__(self, img_h, img_w, this_scale, next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0], two_boxes_for_ar1=True,
                 this_steps=None, this_offsets=None, clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids', normalize_coords=False, **kwargs):
        # Raise exceptions

        self.img_h = img_h
        self.img_w = img_w
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords

        if 1 in aspect_ratios and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def call(self, x, mask=None):
        size = min(self.img_h, self.img_w)
        wh_list = []
        for ar in self.aspect_ratios:
            if ar == 1:
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(
                        self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        if K.common.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        else:
            batch_size, feature_map_channels, feature_map_height, feaure_map_width = x._keras_shape

        if self.this_steps is None:
            step_height = self.img_h / feature_map_height
            step_width = self.img_w / feature_map_height
        else:
            if isinstance(self.this_steps, (list, tuple)) and len(self.this_steps) == 2:
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps

        if self.this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and len(self.this_offsets) == 2:
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets

        cy = np.linspace(offset_height * step_height,
                         (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width +
                                                     feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        boxes_tensor = np.zeros(
            (feature_map_height, feature_map_width, self.n_boxes, 4))

        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_w] = self.img_w - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords

            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_h] = self.img_h - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_w
            boxes_tensor[:, :, :, [1, 3]] /= self.img_h

        if self.coords == 'centroids':
            boxes_tensor = convert_coordinates(
                boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            boxes_tensor = convert_coordinates(
                boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances
        boxes_tensor = np.concatenate(
            (boxes_tensor, variances_tensor), axis=-1)

        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(
            boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.common.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else:
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape

        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_h': self.img_h,
            'img_w': self.img_w,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def ssd_512(image_size, n_classes, mode, l2_regularization=0.0005,
            min_scale=None, max_scale=None, scales=None,
            aspect_ratios_global=None, aspect_ratios_per_layer=[],
            two_boxes_for_ar1=True, steps=[], offsets=None, clip_boxes=False,
            variances=[], coords='centroids', normalize_coords=True,
            subtract_mean=[], divide_by_stddev=None, swap_channels=[],
            confidence_thresh=0.01, iou_thresh=0.45,
            top_k=200, nms_max_output_size=400, return_predictor_sizes=False):
    '''
    Build a SSD architecture for 512x512 input images.

    Arguments:



    Returns:


    '''

    n_predictor_layers = 7  # From original SSD512 design
    n_classes += 1  # Account for background
    img_h, img_w, img_ch = image_size[0], image_size[1], image_size[2]

    # Handle Exceptions

    # Compute Anchor Boxes
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                # +1 for additional box with aspect ratio 1
                n_boxes.append(len(ar) + 1)
            else:
                n_boxes.append(len(ar))

    else:
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    # Define Lambda Layer
    def indentity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[..., swap_channels[0]],
                            tensor[..., swap_channels[1]],
                            tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]],
                            tensor[..., swap_channels[1]],
                            tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    # Build Model Architecture
    x = Input(shape=(image_size))

    x1 = Lambda(indentity_layer, output_shape=image_size,
                name='identity_layer')(x)
    if subtract_mean is not None:
        x1 = Lambda(input_mean_normalization, output_shape=image_size,
                    name='input_mean_normalization')(x1)
    if divide_by_stddev is not None:
        x1 = Lambda(input_stddev_normalization, output_shape=image_size,
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=image_size,
                    name='input_channel_swap')(x1)

    # VGG-16 Base Network
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), padding='same',
                         name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), padding='same',
                         name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv5_3')(conv5_2)
    # SSD converts pool5 to 3x3 w stride 1
    pool5 = MaxPooling2D(pool_size=(3, 3),
                         strides=(1, 1), padding='same', name='pool5')(conv5_3)

    # SSD converts the fully connected layers 6 and 7 (fc6 and fc7) of
    # VGG to Convolution layers
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu',
                 padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_regularization), name='fc6')(pool5)
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_regularization), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu',
                     padding='valid',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                     kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization), name='conv8_1')(conv7_2)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv8_padding')(conv7_1)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                     kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization), name='conv9_1')(conv8_2)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv9_padding')(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                     kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name='conv9_2')(conv9_1)

    conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_regularization), name='conv10_1')(conv9_2)
    conv10_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                             name='conv10_padding')(conv10_1)
    conv10_2 = Conv2D(256, (4, 4), activation='relu', padding='valid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_regularization), name='conv10_2')(conv10_1)

    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    # Build predictor layers (convolutional layers)
    # Confidence layers have output (batch, height, width, n_boxes * n_classes)
    # Each box has n_classes confidence values and ea. predictor has depth n_boxes * n_classes
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_regularization), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_regularization), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_regularization), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_regularization), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_regularization), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_regularization), name='conv9_2_mbox_conf')(conv9_2)
    conv10_2_mbox_conf = Conv2D(n_boxes[6] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=l2(l2_regularization), name='conv10_2_mbox_conf')(conv10_2)

    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_regularization), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_regularization), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_regularization), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_regularization), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_regularization), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_regularization), name='conv9_2_mbox_loc')(conv9_2)
    conv10_2_mbox_loc = Conv2D(n_boxes[6] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_regularization), name='conv10_2_mbox_loc')(conv10_2)

    # Generate anchor boxes
    conv4_3_norm_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[0], next_scale=scales[1],
                                              aspect_ratios=aspect_ratios[0], two_boxes_for_ar1=two_boxes_for_ar1,
                                              this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                              variances=variances, coords=coords, normalize_coords=normalize_coords,
                                              name='conv4_3_norm_mbox_anchorbox')(conv4_3_norm_mbox_loc)

    fc7_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[1], next_scale=scales[2],
                                     aspect_ratios=aspect_ratios[1], two_boxes_for_ar1=two_boxes_for_ar1,
                                     this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords,
                                     name='fc7_mbox_loc_anchorbox')(fc7_mbox_loc)

    conv6_2_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[2], next_scale=scales[3],
                                         aspect_ratios=aspect_ratios[2], two_boxes_for_ar1=two_boxes_for_ar1,
                                         this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv6_2_mbox_anchorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[3], next_scale=scales[4],
                                         aspect_ratios=aspect_ratios[3], two_boxes_for_ar1=two_boxes_for_ar1,
                                         this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv7_2_mbox_anchorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[4], next_scale=scales[5],
                                         aspect_ratios=aspect_ratios[4], two_boxes_for_ar1=two_boxes_for_ar1,
                                         this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv8_2_mbox_anchorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[5], next_scale=scales[6],
                                         aspect_ratios=aspect_ratios[5], two_boxes_for_ar1=two_boxes_for_ar1,
                                         this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv9_2_mbox_anchorbox')(conv9_2_mbox_loc)
    conv10_2_mbox_anchorbox = AnchorBoxes(img_h, img_w, this_scale=scales[6], next_scale=scales[7],
                                          aspect_ratios=aspect_ratios[6], two_boxes_for_ar1=two_boxes_for_ar1,
                                          this_steps=steps[6], this_offsets=offsets[6], clip_boxes=clip_boxes,
                                          variances=variances, coords=coords, normalize_coords=normalize_coords,
                                          name='conv10_2_mbox_anchorbox')(conv10_2_mbox_loc)

    # Reshape class predictions to shape (batch, height*width*n_boxes, n_classes)
    conv4_3_norm_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    conv10_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv10_2_mbox_conf_reshape')(conv10_2_mbox_conf)

    # # Reshape box predictions to (batch, height*width*n_boxes, 4)
    conv4_3_norm_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape(
        (-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    conv10_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv10_2_mbox_loc_reshape')(conv10_2_mbox_loc)

    # Reshape anchor box to tensor (batch, height*width*n_boxes, 8)
    conv4_3_norm_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='conv4_3_norm_mbox_anchorbox_reshape')(conv4_3_norm_mbox_anchorbox)
    fc7_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_anchorbox)
    conv6_2_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='conv6_2_mbox_anchorbox_reshape')(conv6_2_mbox_anchorbox)
    conv7_2_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='conv7_2_mbox_anchorbox_reshape')(conv7_2_mbox_anchorbox)
    conv8_2_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='conv8_2_mbox_anchorbox_reshape')(conv8_2_mbox_anchorbox)
    conv9_2_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='conv9_2_mbox_anchorbox_reshape')(conv9_2_mbox_anchorbox)
    conv10_2_mbox_anchorbox_reshape = Reshape(
        (-1, 8), name='conv10_2_mbox_anchorbox_reshape')(conv10_2_mbox_anchorbox)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape,
                                                       conv10_2_mbox_conf_reshape])

    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape,
                                                     conv10_2_mbox_loc_reshape])

    mbox_anchorbox = Concatenate(axis=1, name='mbox_anchorbox')([conv4_3_norm_mbox_anchorbox_reshape,
                                                                 fc7_mbox_anchorbox_reshape,
                                                                 conv6_2_mbox_anchorbox_reshape,
                                                                 conv7_2_mbox_anchorbox_reshape,
                                                                 conv8_2_mbox_anchorbox_reshape,
                                                                 conv9_2_mbox_anchorbox_reshape,
                                                                 conv10_2_mbox_anchorbox_reshape])

    # Apply softmax to class predictions
    mbox_conf_softmax = Activation(
        'softmax', name='mbox_conf_softmax')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_anchorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_thresh,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_h,
                                               img_width=img_w,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError('Mode must be training or inference.')

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                    fc7_mbox_conf._keras_shape[1:3],
                                    conv6_2_mbox_conf._keras_shape[1:3],
                                    conv7_2_mbox_conf._keras_shape[1:3],
                                    conv8_2_mbox_conf._keras_shape[1:3],
                                    conv9_2_mbox_conf._keras_shape[1:3],
                                    conv10_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.
    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format
    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.
    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind+1]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] +
                               tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - \
            tensor[..., ind] + d  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - \
            tensor[..., ind+2] + d  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - \
            tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + \
            tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - \
            tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + \
            tensor[..., ind+3] / 2.0  # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind+2]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] +
                               tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - \
            tensor[..., ind] + d  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - \
            tensor[..., ind+1] + d  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - \
            tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - \
            tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + \
            tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + \
            tensor[..., ind+3] / 2.0  # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError(
            "Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1


class DecodeDetections(Layer):
    '''
    A Keras layer to decode the raw SSD prediction output.
    Input shape:
        3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.
    Output shape:
        3D tensor of shape `(batch_size, top_k, 6)`.
    '''

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):
        '''
        All default argument values follow the Caffe implementation.
        Arguments:
            confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
                positive class in order to be considered for the non-maximum suppression stage for the respective class.
                A lower value will result in a larger part of the selection process being done by the non-maximum suppression
                stage, while a larger value will result in a larger part of the selection process happening in the confidence
                thresholding stage.
            iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
                with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
                to the box score.
            top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
                non-maximum suppression stage.
            nms_max_output_size (int, optional): The maximum number of predictions that will be left after performing non-maximum
                suppression.
            coords (str, optional): The box coordinate format that the model outputs. Must be 'centroids'
                i.e. the format `(cx, cy, w, h)` (box center coordinates, width, and height). Other coordinate formats are
                currently not supported.
            normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
                and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
                relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
                Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
                coordinates. Requires `img_height` and `img_width` if set to `True`.
            img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
            img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError(
                "This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        if coords != 'centroids':
            raise ValueError(
                "The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

        # We need these members for the config.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # We need these members for TensorFlow.
        self.tf_confidence_thresh = tf.constant(
            self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(
            self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(
            self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(
            self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(
            self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(
            self.nms_max_output_size, name='nms_max_output_size')

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Returns:
            3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
            to always yield `top_k` predictions per batch item. The last axis contains
            the coordinates for each predicted box in the format
            `[class_id, confidence, xmin, ymin, xmax, ymax]`.
        '''

        #####################################################################################
        # 1. Convert the box coordinates from predicted anchor box offsets to predicted
        #    absolute coordinates
        #####################################################################################

        # Convert anchor box offsets to image offsets.
        # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cx = y_pred[..., -12] * y_pred[..., -4] * \
            y_pred[..., -6] + y_pred[..., -8]
        # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        cy = y_pred[..., -11] * y_pred[..., -3] * \
            y_pred[..., -5] + y_pred[..., -7]
        # w = exp(w_pred * variance_w) * w_anchor
        w = tf.exp(y_pred[..., -10] * y_pred[..., -2]) * y_pred[..., -6]
        # h = exp(h_pred * variance_h) * h_anchor
        h = tf.exp(y_pred[..., -9] * y_pred[..., -1]) * y_pred[..., -5]

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        # If the model predicts box coordinates relative to the image dimensions and they are supposed
        # to be converted back to absolute coordinates, do that.
        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1

        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(
            self.tf_normalize_coords, normalized_coords, non_normalized_coords)

        # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
        y_pred = tf.concat(
            values=[y_pred[..., :-12], xmin, ymin, xmax, ymax], axis=-1)

        #####################################################################################
        # 2. Perform confidence thresholding, per-class non-maximum suppression, and
        #    top-k filtering.
        #####################################################################################

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1]
        n_classes = y_pred.shape[2] - 4
        class_indices = tf.range(1, n_classes)

        # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):

            # Create a function that filters the predictions for one single class.
            def filter_single_class(index):

                # From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract
                # a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
                # confidnece values for just one class, determined by `index`.
                confidences = tf.expand_dims(batch_item[..., index], axis=-1)
                class_id = tf.fill(dims=tf.shape(
                    confidences), value=tf.to_float(index))
                box_coordinates = batch_item[..., -4:]

                single_class = tf.concat(
                    [class_id, confidences, box_coordinates], axis=-1)

                # Apply confidence thresholding with respect to the class defined by `index`.
                threshold_met = single_class[:, 1] > self.tf_confidence_thresh
                single_class = tf.boolean_mask(tensor=single_class,
                                               mask=threshold_met)

                # If any boxes made the threshold, perform NMS.
                def perform_nms():
                    scores = single_class[..., 1]

                    # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                    xmin = tf.expand_dims(single_class[..., -4], axis=-1)
                    ymin = tf.expand_dims(single_class[..., -3], axis=-1)
                    xmax = tf.expand_dims(single_class[..., -2], axis=-1)
                    ymax = tf.expand_dims(single_class[..., -1], axis=-1)
                    boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                    maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                  scores=scores,
                                                                  max_output_size=self.tf_nms_max_output_size,
                                                                  iou_threshold=self.iou_threshold,
                                                                  name='non_maximum_suppresion')
                    maxima = tf.gather(params=single_class,
                                       indices=maxima_indices,
                                       axis=0)
                    return maxima

                def no_confident_predictions():
                    return tf.constant(value=0.0, shape=(1, 6))

                single_class_nms = tf.cond(
                    tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

                # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
                padded_single_class = tf.pad(tensor=single_class_nms,
                                             paddings=[
                                                 [0, self.tf_nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
                                             mode='CONSTANT',
                                             constant_values=0.0)

                return padded_single_class

            # Iterate `filter_single_class()` over all class indices.
            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                                elems=tf.range(1, n_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128,
                                                back_prop=False,
                                                swap_memory=False,
                                                infer_shape=True,
                                                name='loop_over_classes')

            # Concatenate the filtered results for all individual classes to one tensor.
            filtered_predictions = tf.reshape(
                tensor=filtered_single_classes, shape=(-1, 6))

            # Perform top-k filtering for this batch item or pad it in case there are
            # fewer than `self.top_k` boxes left at this point. Either way, produce a
            # tensor of length `self.top_k`. By the time we return the final results tensor
            # for the whole batch, all batch items must have the same number of predicted
            # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
            # predictions are left after the filtering process above, we pad the missing
            # predictions with zeros as dummy entries.
            def top_k():
                return tf.gather(params=filtered_predictions,
                                 indices=tf.nn.top_k(
                                     filtered_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=filtered_predictions,
                                            paddings=[
                                                [0, self.tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(
                                     padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[
                                  0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        # Iterate `filter_predictions()` over all batch items.
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        # Last axis: (class_ID, confidence, 4 box coordinates)
        return (batch_size, self.tf_top_k, 6)

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
