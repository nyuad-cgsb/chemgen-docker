FROM continuumio/anaconda3

RUN apt-get update
RUN apt-get install -y curl
RUN conda install -y flask numpy pandas opencv scipy pip setuptools
RUN pip install  request jsonify tensorflow==1.4  gunicorn
RUN pip install pandas opencv-python matplotlib

RUN mkdir -p /model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco
RUN mkdir -p /images
RUN mkdir -p /scripts

ADD research.tar.gz /scripts
ADD frozen_inference_graph.pb /model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/
ADD worm_detection_labels.pbtxt /model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/
ADD get_counts.py /scripts/
ADD RNAiI.3A1E_M_H12-autolevel.png /images

ENV PYTHONPATH "/scripts/research:/scripts/research/slim"

RUN python /scripts/get_counts.py \
--model_path /model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/frozen_inference_graph.pb \
--label_path /images/annotate \
--labels /model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/worm_detection_labels.pbtxt \
--counts /images/counts.csv \
--image_path /images

