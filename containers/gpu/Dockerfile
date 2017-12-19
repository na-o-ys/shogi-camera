From nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y apt-utils git wget sudo g++ python-dev python-pip python3-dev python3-pip && \
    pip install --upgrade pip && \
    pip install numpy && \
    pip3 install --upgrade pip && \
    pip3 install numpy

# OpenCV
COPY opencv/dependencies.sh opencv/dependencies.sh
RUN cd opencv && ./dependencies.sh
COPY opencv/make_install.sh opencv/make_install.sh
RUN cd opencv && \
    ./make_install.sh && \
    cd ../ && \
    rm -rf opencv

RUN apt-get install -y python-pip python3-pip libhdf5-dev graphviz
RUN pip3 install scipy scikit-learn pandas matplotlib jupyter tensorflow-gpu Pillow h5py keras
RUN pip3 install click scikit-image

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH /app

WORKDIR /app
