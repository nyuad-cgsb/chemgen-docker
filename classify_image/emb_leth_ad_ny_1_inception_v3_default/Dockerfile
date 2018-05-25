# This image has the inception_v3 model retrained with
# Images from NY and AD

FROM tensorflow/tensorflow:1.7.0-py3

RUN apt-get update
RUN apt-get install -y  curl
RUN pip install Flask flask_restful request jsonify tensorflow==1.7 tensorflow_hub numpy pandas gunicorn
RUN mkdir -p /model
RUN mkdir -p /images
RUN mkdir -p /scripts
ADD emb_leth_ad_ny_1_hub_inceptionv3 /model/emb_leth_ad_ny_1_hub_inceptionv3
ADD RNAiI.3A2E_M_D03-autolevel.jpeg  /images/RNAiI.3A2E_M_D03-autolevel.jpeg
ADD retrain.py /scripts
WORKDIR /scripts
RUN curl https://raw.githubusercontent.com/jerowe/chemgen-docker/master/classify_image/flask_app/flask_label_images.py -o flask_label_images.py

ENV LABEL_FILE /model/emb_leth_ad_ny_1_hub_inceptionv3/output_labels.txt
ENV GRAPH_FILE /model/emb_leth_ad_ny_1_hub_inceptionv3/output_graph.pb

EXPOSE 5000
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:5000", "flask_label_images:app"]
