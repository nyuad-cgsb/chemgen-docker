FROM ubuntu

RUN apt-get update -y
RUN apt-get install -y python3 build-essential cmake libpng-dev zlib1g-dev vim-common clang zip libpng12-dev
RUN mkdir -p /scripts
RUN mkdir -p /images
RUN mkdir -p /model
ADD model /model

ADD devstar-master.zip /scripts/

WORKDIR /scripts/
RUN unzip devstar-master.zip
WORKDIR /scripts/devstar-master
RUN make
RUN ./devstar -h
#RUN devstar analyze example/ad.json
