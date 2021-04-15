from torchvision import transforms
from torch.utils import data

from PIL import Image

import os
import re
import csv
import h5py
import glob
import shutil
import random
import numpy as np
import torch
import tensorflow as tf
import tensorflow.keras.backend as B

def filterMask(mask, valList):
    newmask = tf.cast(mask == valList[0], tf.float32)
    for val in valList[1:]:
        newmask += tf.cast(mask == val, tf.float32)
    return newmask

def _parse(proto):
    keys_to_features = {
        "personno": tf.io.FixedLenFeature([], tf.int64),
        "person": tf.io.FixedLenFeature([], tf.string),
        "personMask": tf.io.FixedLenFeature([], tf.string),
        "densepose": tf.io.FixedLenFeature([], tf.string),
        "is_shorts": tf.io.FixedLenFeature([], tf.string),
        "is_short_sleeve": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    person = (tf.cast(tf.image.decode_jpeg(parsed_features["person"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5
    grapy = tf.cast(tf.image.decode_png(parsed_features["personMask"], channels=1), tf.float32)
    mapping = tf.constant([0, 1, 1, 13, 1, 2, 4, 2, 14, 3, 5, 12, 3, 1, 6, 7, 8, 9, 10, 11, 2, 2])
    grapy = tf.gather(mapping, tf.cast(grapy, dtype=tf.int32))
    grapy = tf.cast(grapy, tf.float32)

    densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
    dp_seg = tf.cast(densepose[..., 0], tf.int32)
    dp_seg = tf.cast(tf.one_hot(dp_seg, depth=25), tf.float32)
    dp_uv = (tf.cast(densepose[..., 1:], tf.float32) / 255.0 - 0.5) / 0.5

    person_no = parsed_features["personno"]
    is_shorts = parsed_features["is_shorts"]
    is_short_sleeve = parsed_features["is_short_sleeve"]
    
    p_sil = tf.cast(grapy > 0, tf.float32)
    p_sil = erode_grapy(p_sil)
    grapy = grapy*p_sil
    person = person * p_sil

    # Multiplying by cloth mask to get torso
    occluded_cloth_mask = filterMask(grapy, [2, 4])
    person_cloth = person*occluded_cloth_mask

    g = tf.random.uniform([])
    if g > 0.75:
        pass_skin = tf.zeros([256, 192, 1], tf.float32)
    else:
        pass_skin = tf.ones([256, 192, 1], tf.float32)

    shape_mask = tf.cast(filterMask(grapy, [1, 3, 8, 9, 10, 11, 12, 13, 14]), dtype=tf.float32)
    lower_body = shape_mask
    upper_clothes = tf.cast(filterMask(grapy, [2, 4]), dtype=tf.float32)
    upper_body = tf.stack([filterMask(grapy[..., 0], [i]) for i in [5, 6, 7]], axis=-1)
    exp_seg = tf.concat([filterMask(grapy, [0]), upper_clothes, upper_body, lower_body], axis=-1)
    exp_seg = B.argmax(exp_seg, axis=-1)[..., None]
    exp_seg = tf.cast(exp_seg, tf.float32)
    # exp_seg id : bg, upper_clothes, torsoskin, leftarm, rightarm, lowerbody

    densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
    dp_seg = tf.cast(densepose[..., 0], tf.int32)
    dp_seg = tf.one_hot(dp_seg, depth=25)
    dp_seg = tf.cast(dp_seg, tf.float32)
    dp_uv = (tf.cast(densepose[..., 1:], tf.float32) / 255.0 - 0.5) / 0.5
    densepose = tf.concat([dp_seg, dp_uv], axis=-1)
    random_cloth = prep_cloth_random(occluded_cloth_mask, person_cloth)

    if g < 0.75:
        left_hand = tf.cast(filterMask(grapy, [6]) * filterMask(exp_seg, [3]), dtype=tf.bool)
        right_hand = tf.cast(filterMask(grapy, [7]) * filterMask(exp_seg, [4]), dtype=tf.bool)

        left_palm = tf.cast(filterMask(tf.cast(B.argmax(densepose[:, :, :25], -1), tf.float32), [3])[:, :, None], dtype=tf.bool)
        right_palm = tf.cast(filterMask(tf.cast(B.argmax(densepose[:, :, :25], -1), tf.float32), [4])[:, :, None], dtype=tf.bool)

        left_hand = tf.cast(tf.math.logical_or(left_hand, left_palm), dtype=tf.float32)
        right_hand = tf.cast(tf.math.logical_or(right_hand, right_palm), dtype=tf.float32)

        prior_left, eroded_part_cloth_left = simulate_misalign(left_hand)
        prior_left = prior_left * pass_skin 
        eroded_part_cloth_left = eroded_part_cloth_left*random_cloth

        prior_right, eroded_part_cloth_right = simulate_misalign(right_hand)
        prior_right = prior_right * pass_skin
        eroded_part_cloth_right = eroded_part_cloth_right*random_cloth

        # arm is passed after top part cutoff
        left_hand = tf.cast(curvy_cut(prior_left) * prior_left, dtype=tf.bool)
        right_hand = tf.cast(curvy_cut(prior_right) * prior_right, dtype=tf.bool)
        hand_mask = tf.math.logical_or(left_hand, right_hand)

        if g > 0.3:
            # Passing in full palms when rest of the side arm is eroded 
            palms = tf.math.logical_or(left_palm, right_palm)
            hand_mask = tf.math.logical_or(hand_mask, palms)
        
        hand_mask = tf.cast(hand_mask, tf.float32)

        print_prior = hand_mask
        
    else:
        # No arm is passed
        palms = filterMask(tf.cast(B.argmax(densepose[:, :, :25], -1), tf.float32), [3, 4])[:, :, None]
        print_prior = tf.cast(palms, dtype=tf.float32)
        hand_mask = tf.cast(palms, dtype=tf.float32)

        eroded_part_cloth_left = tf.zeros([256,192,3])
        eroded_part_cloth_right = tf.zeros([256,192,3])

    # Torso Skin Prior
    torso_prior, eroded_part_torso = simulate_misalign(filterMask(grapy, [5]) * filterMask(exp_seg, [2]))
    torso_prior = torso_prior*pass_skin
    random_cloth2 = prep_cloth_random(occluded_cloth_mask, person_cloth)
    eroded_part_torso = eroded_part_torso*random_cloth2
    if g < 0.5:
        torso_mask = tf.cast(curvy_cut(torso_prior) * torso_prior, dtype=tf.float32)
    else:
        torso_mask = torso_prior

    # Model inputs 
    # Isolating the skin inpainting by giving tryon cloth as input
    person_priors_mask = filterMask(exp_seg, [1, 5]) + hand_mask + torso_mask
    person_priors = person * person_priors_mask 

    added_priors = person * filterMask(exp_seg, [1, 5])

    person_cloth_mask = filterMask(exp_seg, [1])
    warped_cloth = person * person_cloth_mask        

    skin_prior_mask = filterMask(grapy, [8, 9]) + hand_mask + torso_mask
    skin_prior = person * skin_prior_mask
    skin_mask = filterMask(grapy, [5, 6, 7]) * filterMask(exp_seg, [2, 3, 4])

    # Adding grapy misalignment
    #Arms
    k = tf.random.uniform([])
    if k < 0.33:
        person_priors = person_priors + eroded_part_cloth_left
        skin_prior = skin_prior + eroded_part_cloth_left
    elif k < 0.66:
        person_priors = person_priors + eroded_part_cloth_right
        skin_prior = skin_prior + eroded_part_cloth_right

        #Torso
        person_priors = person_priors + eroded_part_torso
        skin_prior = skin_prior + eroded_part_torso
    else: 
        person_priors = person_priors + eroded_part_cloth_left + eroded_part_cloth_right
        skin_prior = skin_prior + eroded_part_cloth_left + eroded_part_cloth_right


    is_short_sleeve = parsed_features["is_short_sleeve"]

    data = {
        "warped_cloth": warped_cloth,
        "person_priors": person_priors,
        "added_priors": added_priors,
        "exp_seg": exp_seg,
        "skin_mask": skin_mask,
        "person": person,
        "densepose": densepose,
        "person_no": person_no,
        "skin_priors": skin_prior,
        "is_short_sleeve": is_short_sleeve
    }

    model_inputs = {
        "person_priors": person_priors,
        "added_priors": added_priors,
        "exp_seg": exp_seg,
        "skin_priors": skin_prior,
        "skin_mask": skin_mask,
    }
    return data, model_inputs


def simulate_misalign(mask):        
    mask = tf.expand_dims(mask, axis=0)

    k = tf.random.uniform([])
    if k < 0.16:
        kernel = tf.cast(tf.ones((2, 3, 1)), tf.double)
    elif k < 0.32:
        kernel = tf.cast(tf.ones((2, 2, 1)), tf.double)
    elif k < 0.48:
        kernel = tf.cast(tf.ones((3, 2, 1)), tf.double)
    elif k < 0.64:
        kernel = tf.cast(tf.ones((1, 2, 1)), tf.double)
    elif k < 0.80:
        kernel = tf.cast(tf.ones((2, 1, 1)), tf.double)
    else:
        kernel = tf.cast(tf.ones((1, 1, 1)), tf.double)

    eroded_mask = tf.nn.erosion2d(
            value=tf.cast(mask, tf.double),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 3, 3, 1),
            padding="SAME",
            data_format="NHWC",
    )
    eroded_mask = eroded_mask[0] + 1
    eroded_mask = tf.cast(eroded_mask, tf.float32)

    # To account for misalignment of grapy
    eroded_part = mask[0] - eroded_mask
    eroded_part_cloth = eroded_part

    return eroded_mask, eroded_part_cloth


def erode_grapy(mask):        
    mask = tf.expand_dims(mask, axis=0)

    kernel = tf.cast(tf.ones((2, 2, 1)), tf.double)

    eroded_mask = tf.nn.erosion2d(
            value=tf.cast(mask, tf.double),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 3, 3, 1),
            padding="SAME",
            data_format="NHWC",
    )
    eroded_mask = eroded_mask[0] + 1
    eroded_mask = tf.cast(eroded_mask, tf.float32)

    return eroded_mask

def dilate_skin(mask):
    mask = tf.expand_dims(mask, axis=0)

    kernel = tf.cast(tf.ones((2, 2, 1)), tf.double)

    dilated_mask = tf.nn.dilation2d(
            input=tf.cast(mask, tf.double),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 3, 3, 1),
            padding="SAME",
            data_format="NHWC"
    )
    dilated_mask = dilated_mask[0] - 1
    dilated_mask = tf.cast(dilated_mask, tf.float32)

    return dilated_mask

def prep_cloth_random(clothMask, cloth):

    k = tf.random.uniform([])

    if k < 0.2:
        i_cloth_ten = tf.ones([256, 192, 3])
    elif k < 0.6:
        i_cloth_ten = tf.random.uniform([256, 192, 3], minval=-1, maxval=1)
    else:
        px = tf.where(clothMask == 1)

        try:
            # Crop Cloth for accurate scaling
            x = tf.cast(tf.minimum(tf.maximum(tf.reduce_min(px[:, 0]), 10), 254), tf.int32)
            y = tf.cast(tf.minimum(tf.maximum(tf.reduce_min(px[:, 1]), 10), 190), tf.int32)
            height = tf.cast(tf.minimum(tf.maximum(tf.reduce_max(px[:, 0]) - tf.reduce_min(px[:, 0]), 10), 244), tf.int32)
            width = tf.cast(tf.minimum(tf.maximum(tf.reduce_max(px[:, 1]) - tf.reduce_min(px[:, 1]), 10), 180), tf.int32)
            #tf.print(x, y, height, width, 'before')

            x = tf.minimum(x, 254-height)
            y = tf.minimum(y, 190-width)
            #tf.print(x, y, height, width, 'after')

            i_cloth_ten = tf.image.crop_to_bounding_box(cloth, x, y, height, width)

            i_cloth_ten = tf.image.resize(i_cloth_ten, [256, 192], method='nearest')
        except:
            i_cloth_ten = tf.ones([256, 192, 3])

    return i_cloth_ten

def curvy_cut(cols):
    """
    Code to produce random curvy cut below the sleeve
    """
    image = tf.random.uniform(shape=(1, 600, 1))
    image = B.concatenate((tf.zeros((1, 600, 1)), image), 0)
    image = B.concatenate((image, tf.ones((1, 600, 1))), 0)
    image = tf.image.resize(image, [30, 10000])
    image = tf.cast(image + 0.5, tf.int32)

    diff = tf.cast(tf.random.uniform(shape=()) * 25.0 + 25.0, tf.int64)
    px = tf.where(cols == 1)
    y1 = tf.reduce_min(px[:, 1])
    y2 = tf.reduce_max(px[:, 1])
    y1 = tf.maximum(tf.minimum(y1 - 1, 190), 1)
    y2 = tf.maximum(tf.minimum(y2 - 1, 191), y1 + 1)
    x = tf.reduce_min(tf.where(cols)[:, 0])
    y = tf.maximum(tf.minimum(x + diff, 254), 1)
    y_coord = tf.cast(tf.random.uniform(shape=()) * tf.cast((10000 - (y2 - y1)), tf.float32), tf.int64)
    k = tf.maximum(tf.minimum(tf.cast(10, tf.int64), y), 1)
    sliced = tf.slice(image, [20-k, y_coord, 0], [k, y2 - y1, 1])

    hand_mask = tf.cast(
        B.concatenate(
            (
                B.concatenate(
                    (
                        tf.ones([y, y1, 1]),
                        B.concatenate((tf.zeros([y - k, y2 - y1, 1]), tf.cast(sliced, tf.float32)), axis=0),
                        tf.ones([y, 192 - y2, 1]),
                    ),
                    1,
                ),
                tf.ones([256 - y, 192, 1]),
            ),
            0,
        ),
        tf.float32,
    )

    return hand_mask

def _filter(data, model_inputs):
    #string = data["cloth_category"]
    return tf.strings.regex_full_match("stripes", "stripes")

def create_dataset(parse_func, filter_func, tfrecord_path, num_data, batch_size, mode, data_split, device):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    if mode == "train":
        dataset = dataset.take(int(data_split * num_data))
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

    elif mode == "val":
        dataset = dataset.skip(int(data_split * num_data))

    elif mode == "k_worst":
        dataset = dataset.take(data_split * num_data)

    dataset = dataset.map(
        parse_func,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.filter(filter_func)

    if mode != "k_worst":
        num_lines = sum(1 for _ in dataset)
        # num_lines = 15000
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        num_lines = num_data  # doesn't get used anywhere
        dataset = dataset.batch(batch_size, drop_remainder=False)

    if device != "colab_tpu":
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, num_lines
    else:
        return dataset

def define_dataset(tfrecord_path, batch_size, train=True, test=False):
    per_replica_train_batch_size = batch_size
    per_replica_val_batch_size = batch_size
    if test:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=5000,
            batch_size=per_replica_train_batch_size,
            mode="k_worst",
            data_split=1,
            device='gpu',
        )
        return data_gen, dataset_length

    if train:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=37129,
            batch_size=per_replica_train_batch_size,
            mode="train",
            data_split=0.8,
            device='gpu',
        )
    else:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=37129,
            batch_size=per_replica_val_batch_size,
            mode="val",
            data_split=0.8,
            device='gpu',
        )
    return data_gen, dataset_length