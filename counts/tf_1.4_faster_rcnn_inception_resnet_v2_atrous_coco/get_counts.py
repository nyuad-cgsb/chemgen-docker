#!/usr/bin/env python3

# coding: utf-8
# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will
# walk you step by step through the process of using a pre-trained model to
# detect objects in an image. Make sure to follow the [installation
# instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
# before you start.

import time
import collections
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
# from pascal_voc_io import PascalVocWriter

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image
from scipy import misc
import cv2
import numpy as np
# from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('model_path', '', 'Path to the model')
flags.DEFINE_string('image_path', '', 'Path to the testing images')
flags.DEFINE_string('label_path', '', 'Path to the output labelled images')
flags.DEFINE_string('counts', '', 'Path to the output counts')
flags.DEFINE_string(
    'labels', '', 'Path to image labels (child, adult, larvae)')
FLAGS = flags.FLAGS

##########################################################################
### Begin Command Line Args
# # Model preparation
# Any model exported using the `export_inference_graph.py` tool
# can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb
# file.
PATH_TO_CKPT = FLAGS.model_path
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = FLAGS.labels
# Number of classes to test for - this is stupid should just be the number we have in the classes config
NUM_CLASSES = 5

TEST_IMAGE_PATHS = []

if not os.path.exists(FLAGS.image_path):
    BMP_TEST_IMAGE_PATHS = glob.glob(os.path.join(FLAGS.image_path, '*autolevel.bmp'))
    PNG_TEST_IMAGE_PATHS = glob.glob(os.path.join(FLAGS.image_path, '*autolevel.png'))

    for bmp in BMP_TEST_IMAGE_PATHS:
        png = bmp.replace('bmp', 'png')
        if png in PNG_TEST_IMAGE_PATHS:
            TEST_IMAGE_PATHS.append(png)
        else:
            TEST_IMAGE_PATHS.append(bmp)
else:
    TEST_IMAGE_PATHS = [os.path.abspath(FLAGS.image_path)]

### End Command Line Args
##########################################################################

##########################################################################
### Begin Config Options
## Egg size config is fairly unscientific
## I just opened up the images in gimp and tried to count the number of pixels

EGG_SIZE_CONFIG = {2000: 70, 1600: 70, 1104: 70}


# # Size, in inches, of the display images.
# IMAGE_SIZE = (24, 12)

def read_detection_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_labelmap():
    """Loading label map Label maps map indices to category names, so that when
    our convolution network predicts `5`, we know that this corresponds to
    `airplane`.  Here we use internal utility functions, but anything that returns
    a dictionary mapping integers to appropriate string labels would be fine"""
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return (label_map, categories, category_index)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata())
    return image_np.reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def post_process_egg_clump(image_path, height, coordinates):
    """
    image_path: Str to path of image
    coordinates: List of coordinates [(xmin, ymin, xmax, ymax)]

    This function gets the coordinates of each egg clump,
    crops the whole image to that just the egg clump, applies some thresholding magic,
    and then counts the number of black pixels, which it divides by the average number of pixels per egg
    """
    img = cv2.imread(image_path, 0)
    if height in EGG_SIZE_CONFIG:
        EGG_SIZE = EGG_SIZE_CONFIG[height]
    else:
        EGG_SIZE = 70

    total_count = 0
    for coordinate in coordinates:
        xmin, xmax, ymin, ymax = coordinate
        # Crop it here
        crop_img = img[ymin:ymax, xmin:xmax]
        blur = cv2.GaussianBlur(crop_img, (5, 5), 0)
        ret3, th3 = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        count = 0
        for i in range(0, th3.shape[0] - 1):
            row = th3[i]
            for cell in row:
                if cell == 0:
                    count += 1
                    total_count += 1
            if count is not 0:
                this_clump_eggs = round(EGG_SIZE / count)

    # Divide count by the average number of pixels per egg

    if total_count == 0:
        total_eggs = 0
    else:
        total_eggs = round(total_count / EGG_SIZE)

    return total_eggs


def get_normalized_coordinates(image, image_path, image_np, boxes, classes, scores, category_index):
    """
    The code for getting the coordinates comes from the vis_util.visualize_boxes_and_labels_on_image_array
    The code for writing the coordinates to pascalVoc format comes from the labelImg program
    PascalVocWriter
        foldername
        filename
        imageSize
        localImgPath=image_path
    """

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    min_score_thresh = .4
    max_boxes_to_draw = boxes.shape[0]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    xml_name = base_name + '.xml'
    full_name = os.path.join(FLAGS.label_path, xml_name)
    full_name = os.path.abspath(full_name)
    if os.path.exists(full_name):
        os.remove(full_name)

    image_path = os.path.abspath(image_path)
    # xml_writer = PascalVocWriter(
    #     image_path, image_path, image.size, localImgPath=image_path)
    im_width, im_height = image.size

    full_output = {}
    counts_output = {}
    egg_clump_coordinates = []
    count = {"image_path": image_path, "worm": 0, 'larva': 0,  'egg': 0, 'egg_clump': 0}

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None:
            continue
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if classes[i] in category_index.keys():
                # This is the label
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'
            # Add to xml file
            ymin, xmin, ymax, xmax = box
            # xmin, xmax, ymin, ymax
            (left, right, bottom, top) = (round(xmin * im_width), round(xmax * im_width),
                                          round(ymin * im_height), round(ymax * im_height))
            # xml_writer.addBndBox(left, bottom, right, top, class_name, 0)

            # Post processing for the egg_clump here
            if 'egg_clump' in class_name:
                egg_clump_coordinates.append((left, right, bottom, top))
            count[class_name] += 1

    # xml_writer.save(targetFile=full_name)
    clump_egg_count = post_process_egg_clump(image_path, im_height, egg_clump_coordinates)
    count['egg'] += clump_egg_count

    print(count)
    return count


def label_image(image_path, image_np, boxes, classes, scores, category_index):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        use_normalized_coordinates=True,
        max_boxes_to_draw=None,
        min_score_thresh=.4,
        line_thickness=1)

    # Save the labelled image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    jpeg_name = base_name + '_labelled.bmp'
    # TODO Change this for use on centos server
    full_name = os.path.join(FLAGS.label_path, jpeg_name)

    # plt.imsave(full_name, image_np)


def process_images(image_path, image_tensor, detection_boxes, detection_score, detection_classes, num_detection, category_index):
    image = Image.open(image_path)

    try:
        image_np = load_image_into_numpy_array(image)
    except ValueError as err:
        print('We are going to try converting the image. This could go terribly wrong')
        image_np = load_image_into_numpy_array(image.convert('RGB'))
    except:
        print('Unable to convert the image to a numpy array. Aborting mission!!')
        raise
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # label_image(image_path, image_np, boxes, classes, scores, category_index)
    counts = get_normalized_coordinates(
        image, image_path, image_np, boxes, classes, scores, category_index)
    return counts


#########################################################################
# Main entry
# Read in the detection graph
print('Reading detection graph')
detection_graph = read_detection_graph()
print('Reading label map')
(label_map, categories, category_index) = get_labelmap()

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)

# if not os.path.exists(FLAGS.counts):
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=session_conf) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        os.makedirs(FLAGS.label_path, exist_ok=True)

        results = []
        for image_path in TEST_IMAGE_PATHS:
            print('Starting {} '.format(image_path), file=sys.stderr)
            tic = time.clock()
            counts = process_images(image_path, image_tensor, detection_boxes,
                           detection_scores, detection_classes, num_detections, category_index)
            results.append(counts)
            toc = time.clock()
            print('Finishing {} '.format(image_path), file=sys.stderr)
            print('Time {}'.format(toc-tic), file=sys.stderr)

        df = pd.DataFrame.from_dict(results)
        df.to_csv(FLAGS.counts, index=False)
        print(results)
# else:
#     print('Csv exists {}'.format(FLAGS.counts))
