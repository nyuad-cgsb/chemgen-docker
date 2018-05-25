#!/usr/bin/env python3

"""
WIP - run get_counts as a flask app
Because this is so long running it also needs to be run in the background
"""

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
from PIL import Image
from scipy import misc
import cv2
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

EGG_SIZE_CONFIG = {2000: 70, 1600: 70, 1104: 70}
NUM_CLASSES = 5

from celery import Celery


def make_celery(app):
    celery = Celery(
        app.import_name,
        broker='amqp://admin:mypass@rabbit:5672', backend='rpc://',
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


celery = make_celery(app)

# celery.purge()

@celery.task()
def add_together(a, b):
    print('In Add Together!')
    time.sleep(100)
    print('Finished add together!')
    return a + b


class TfArgs:
    def __init__(self, args):
        """
        Store the initial request in 'args'
        :param args:
        """
        self._args = args

    @property
    def config(self):
        """
        Put general configuration info in here - useful for incorporating this back into a pipeline
        :return:
        """
        if 'config' in self._args:
            return self._args['config']

    @property
    def model_path(self):
        """
        Path to the model exported by export_inference_graph.py
        Usually called frozen_inference.pb or something
        """
        if 'model_path' in self._args:
            return self._args['model_path']
        elif os.environ.get('MODEL_PATH'):
            return os.environ.get('MODEL_PATH')
        else:
            raise Exception('Model File Required!')

    @property
    def image_path(self):
        """
        Path to images to annotate
        :return:  file_path
        """
        if 'image_path' in self._args:
            return self._args['image_path']
        elif os.environ.get('IMAGE_PATH'):
            return os.environ.get('IMAGE_PATH')
        else:
            raise Exception('Image Path Required!')

    @property
    def labels(self):
        """
        Labels of things to integers
        1. worms
        2. larva
        3. egg
        4. egg_clump
        5. bubble
        :return: os path to label file
        """
        if 'labels' in self._args:
            return self._args['labels']
        elif os.environ.get('labels'):
            return os.environ.get('labels')
        else:
            raise Exception('Labels File Required!')

    @property
    def counts(self):
        """
        Optionally, define a path to output a csv of counts
        Otherwise just return them in the request
        :return: os path to output counts
        """
        if 'counts' in self._args:
            return self._args['counts']
        else:
            return False

    @property
    def label_path(self):
        """
        If outputting annotated images add a label_path
        Annotated images are images + boxes with a .xml file for relabelling
        :return: os path to output label_path
        """
        if 'label_path' in self._args:
            return self._args['label_path']
        else:
            return False


def read_detection_graph(args):
    """
    Read in the graph exported by export_inference_graph
    :param args:
    :return:
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_labelmap(args):
    """Loading label map Label maps map indices to category names, so that when
    our convolution network predicts `5`, we know that this corresponds to
    `airplane`.  Here we use internal utility functions, but anything that returns
    a dictionary mapping integers to appropriate string labels would be fine"""
    label_map = label_map_util.load_labelmap(args.labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return (label_map, categories, category_index)


def get_images(args):
    """
    Given an image path, which can be a glob pattern
    Get all the images
    Preferentially use pngs, because those shave off some of the processing time
    In a test of 1000 images results were identical between pngs and bmps
    :param args:
    :return: [image_paths]
    """
    BMP_TEST_IMAGE_PATHS = glob.glob(os.path.join(args.image_path, '*autolevel.bmp'))
    PNG_TEST_IMAGE_PATHS = glob.glob(os.path.join(args.image_path, '*autolevel.png'))
    TEST_IMAGE_PATHS = []

    for bmp in BMP_TEST_IMAGE_PATHS:
        png = bmp.replace('bmp', 'png')
        if png in PNG_TEST_IMAGE_PATHS:
            TEST_IMAGE_PATHS.append(png)
        else:
            TEST_IMAGE_PATHS.append(bmp)

    for png in PNG_TEST_IMAGE_PATHS:
        if png not in TEST_IMAGE_PATHS:
            TEST_IMAGE_PATHS.append(png)

    return TEST_IMAGE_PATHS


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


def get_normalized_coordinates(args, image, image_path, image_np, boxes, classes, scores, category_index):
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

    if args.label_path is not False:
        xml_name = base_name + '.xml'
        full_name = os.path.join(args.label_path, xml_name)
        full_name = os.path.abspath(full_name)
        if os.path.exists(full_name):
            os.remove(full_name)

    image_path = os.path.abspath(image_path)
    # xml_writer = PascalVocWriter(
    #     image_path, image_path, image.size, localImgPath=image_path)
    im_width, im_height = image.size

    egg_clump_coordinates = []
    count = {"image_path": image_path, "worm": 0, 'larva': 0, 'egg': 0, 'egg_clump': 0}

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
            # Add to xml file when relabeling
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
    return count


def process_images(args, sess, image_path, image_tensor, detection_boxes, detection_scores, detection_classes,
                   num_detections,
                   category_index):
    """
    Do the main initial image processing  - get counts and so on
    """

    image = Image.open(image_path)

    try:
        image_np = load_image_into_numpy_array(image)
    except ValueError as err:
        image_np = load_image_into_numpy_array(image.convert('RGB'))
    except Exception as e:
        print('Unable to convert the image to a numpy array. Aborting mission!!')
        raise e

    counts = {}
    print('Image is processing', file=sys.stderr)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # annotate_image(args, image_path, image_np, boxes, classes, scores, category_index)
    counts = get_normalized_coordinates(
        args, image, image_path, image_np, boxes, classes, scores, category_index)
    sys.stdout.flush()
    return counts


DETECTION_GRAPH = False
LABEL_MAP = False
CATEGORIES = False
CATEGORY_INDEX = False
if os.environ.get('MODEL_FILE') and os.environ.get('LABEL_FILE'):
    args = {'model_path': os.environ.get('MODEL_FILE'), 'labels': os.environ.get('LABEL_FILE')}
    args = TfArgs(args)
    DETECTION_GRAPH = read_detection_graph(args)
    (LABEL_MAP, CATEGORIES, CATEGORY_INDEX) = get_labelmap(args)


# WIP, trying to get retries working
## So far they are not
## @celery.task(autoretry_for=(Exception,),
##          retry_kwargs={'max_retries': 2})

@celery.task()
def run_tf_counts(request_args):
    sys.stdout.flush()

    args = TfArgs(request_args)
    print('Beginning job {}'.format(args.image_path), file=sys.stderr)

    ### WHY IS THIS HERE??
    # if args.counts is not False:
    #     open(args.counts, 'a').close()

    size = os.stat(args.counts).st_size == 0

    if args.counts is not False and size:

        print('No counts file found running job {}'.format(args.image_path), file=sys.stderr)
        images = get_images(args)

        if DETECTION_GRAPH is not False:
            detection_graph = DETECTION_GRAPH
            (label_map, categories, category_index) = (LABEL_MAP, CATEGORIES, CATEGORY_INDEX)
        else:
            detection_graph = read_detection_graph(args)
            (label_map, categories, category_index) = get_labelmap(args)

        if args.label_path is not False:
            os.makedirs(args.label_path, exist_ok=True)

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)

        results = []
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

                for image_path in images:
                    print('Starting {} '.format(image_path), file=sys.stderr)
                    sys.stdout.flush()
                    counts = process_images(args, sess, image_path, image_tensor, detection_boxes,
                                            detection_scores, detection_classes, num_detections,
                                            category_index)
                    results.append(counts)
                    sys.stdout.flush()

        if args.counts is not False:
            df = pd.DataFrame.from_dict(results)
            df.to_csv(args.counts, index=False)

        print(results, file=sys.stderr)
        sys.stdout.flush()
    else:
        print('Counts file already found. Run with force to regenerate. Exiting job {}'.format(args.image_path), file=sys.stderr)


@app.route('/tf_counts/1.0/api/count_worms/', methods=['GET', 'POST'])
def label_image():
    content = request.json
    print(content, file=sys.stderr)
    run_tf_counts.delay(content)
    return jsonify({'request': content, 'sent': 1})


@app.route('/tf_counts/1.0/api/health/', methods=['GET', 'POST'])
def health():
    content = request.json
    result = add_together.delay(23, 42)
    # result.wait()
    return jsonify({'request': content, 'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
