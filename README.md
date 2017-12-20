# Shogi Camera

Shogi Camera is an experimental project which aims to extract shogi board information from pictures of shogi games.

![hero](https://dl.dropboxusercontent.com/s/n8pnfuklsnwaz3b/shogicamera_hero.png)

The accuracy is not better for now.

# System Requirements

There are several distinct environment settings on which you can execute the program. One is your native machine and another is a Docker container.

## To execute on a native machine

- Python 3.5
- Keras 2
- Tensorflow 1.1.0
- scipy, numpy, scikit-learn
- hdf5, h5py

## To execute on a Docker container (without GPU)

- Docker Engine

## To execute on a GPU container

Docker host requirements. (fit for AWS p2 instance)

- GPU (NVIDIA Tesla K80)
- CUDA Toolkit / CUDA Driver
- Docker Engine
- nvidia-docker

# Quick Start

```
$ python3 cli.py predict {image_file_path}
```

or on a Docker container,

```
$ docker run -it -v $PWD:/app naoys/shogi-camera:nogpu python3 cli.py predict {image_file_path}
```

```
$ docker run --runtime=nvidia -it -v $PWD:/app naoys/shogi-camera:latest python3 cli.py predict {image_file_path}
```

# Prediction Details

The prediction consists of two phases.

1. detecting a shogi board precisely from the picture
2. predicting single cell information using a pre-trained NN model and repeating it cell by cell

## 1. board detection

see [the notebook](https://github.com/na-o-ys/shogi-camera/blob/master/notebooks/Board%20Detection.ipynb) or [a program](https://github.com/na-o-ys/shogi-camera/blob/master/shogicam/preprocess/_detect_corners.py)

## 2. cell prediction

https://github.com/na-o-ys/shogi-camera/blob/master/shogicam/predict/_predict_board.py

# Model

The simple model contains three convolutional layers and two fully connected layers.

See [the notebook](https://github.com/na-o-ys/shogi-camera/blob/master/notebooks/Learn.ipynb) or [a program](https://github.com/na-o-ys/shogi-camera/blob/master/shogicam/learn/_purple.py).
