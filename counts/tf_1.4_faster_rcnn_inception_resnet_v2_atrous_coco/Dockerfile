FROM continuumio/anaconda3

RUN apt-get update
RUN apt-get install -y curl
RUN conda install -y flask numpy pandas opencv scipy pip setuptools
RUN pip install pandas opencv-python matplotlib celery
RUN pip install request jsonify tensorflow==1.4 gunicorn

RUN useradd -ms /bin/bash webuser
USER webuser
WORKDIR /home/webuser

RUN mkdir -p /home/webuser/model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco
RUN mkdir -p /home/webuser/images
RUN mkdir -p /home/webuser/scripts

ADD research.tar.gz /home/webuser/scripts
ADD frozen_inference_graph.pb /home/webuser/model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/
ADD worm_detection_labels.pbtxt /home/webuser/model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/
ADD get_counts.py /home/webuser/scripts/
ADD flask_app/flask_get_counts.py /home/webuser/scripts
ADD RNAiI.3A1E_M_H12-autolevel.png /home/webuser/images

ENV PYTHONPATH "/home/webuser/scripts/research:/home/webuser/scripts/research/slim"

ENV LABEL_FILE /home/webuser/model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/worm_detection_labels.pbtxt
ENV MODEL_FILE /home/webuser/model/tf_1.4_faster_rcnn_inception_resnet_v2_atrous_coco/frozen_inference_graph.pb
ENV IMAGE_PATH /home/user/images/RNAiI.3A1E_M_H12-autolevel.png

WORKDIR /home/webuser/scripts
EXPOSE 5000

##TODO Change this user - celery complains
USER root

ADD start_app.sh /home/webuser/scripts/start_app.sh

CMD ["/bin/bash", "-c", "./start_app.sh"]

