from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify
import os
import glob
import sys
import numpy as np
import tensorflow as tf

app = Flask(__name__)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


GLOBAL_GRAPH = False
if os.environ.get('GRAPH_FILE'):
    GLOBAL_GRAPH = load_graph(os.environ.get('GRAPH_FILE'))

GLOBAL_LABELS = False
if os.environ.get('LABEL_FILE'):
    GLOBAL_LABELS = load_labels(os.environ.get('LABEL_FILE'))


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


class TfArgs:
    def __init__(self, args):
        self._args = args

    @property
    def input_width(self):
        if 'input_width' in self._args:
            return self._args['input_width']
        else:
            return 299

    @property
    def input_height(self):
        if 'input_height' in self._args:
            return self._args['input_height']
        else:
            return 299

    @property
    def input_std(self):
        if 'input_std' in self._args:
            return self._args['input_std']
        else:
            return 255

    @property
    def input_mean(self):
        if 'input_mean' in self._args:
            return self._args['input_mean']
        else:
            return 0

    @property
    def input_layer(self):
        if 'input_layer' in self._args:
            return self._args['input_layer']
        else:
            return 'Mul'

    @property
    def output_layer(self):
        if 'output_layer' in self._args:
            return self._args['output_layer']
        else:
            return 'final_result'

    @property
    def graph_file(self):
        if 'graph_file' in self._args:
            return self._args['graph_file']
        elif os.environ.get('GRAPH_FILE'):
            return os.environ.get('GRAPH_FILE')
        else:
            raise Exception('Graph File Required!')

    @property
    def label_file(self):
        if 'label_file' in self._args:
            return self._args['label_file']
        elif os.environ.get('LABEL_FILE'):
            return os.environ.get('LABEL_FILE')
        else:
            raise Exception('Label File Required!')

    @property
    def image_files(self):
        if 'image_files' in self._args:
            return self._args['image_files']
        elif 'image_file' in self._args:
            return glob.glob(self._args['image_file'])
        else:
            raise Exception('Image File Required!')

    @property
    def image_file(self):
        if 'image_file' in self._args:
            return self._args['image_file']
        else:
            raise Exception('Image File Required!')


def do_work(args):
    print(args)
    args = TfArgs(args)

    if GLOBAL_GRAPH is not False:
        graph = GLOBAL_GRAPH
    else:
        graph = load_graph(args.graph_file)

    if GLOBAL_LABELS is not False:
        labels = GLOBAL_LABELS
    else:
        labels = load_labels(args.label_file)

    class_data = {}
    for image_file in args.image_files:
        t = read_tensor_from_image_file(
            image_file,
            input_height=args.input_height,
            input_width=args.input_width,
            input_mean=args.input_mean,
            input_std=args.input_std)

        input_name = "import/" + args.input_layer
        output_name = "import/" + args.output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]

        top_hits = {}
        for i in top_k:
            label = labels[i].replace(' ', '_')
            top_hits[label] = float(results[i])
            # print(labels[i], results[i])

        top_hits['conclusion'] = labels[top_k[0]].replace(' ', '_')
        class_data[image_file] = top_hits

    return class_data


@app.route('/api/label_image/', methods=['GET', 'POST'])
def label_image():
    content = request.json
    print(content, file=sys.stderr)
    top_hits = do_work(content)
    return jsonify({'request': content, 'top_hits': top_hits})

@app.route('/api/health/', methods=['GET', 'POST'])
def health():
    content = request.json
    return jsonify({'request': content, 'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
