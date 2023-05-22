# Traffic Signs

# Abstract
Traffic signs support road safety and managing the flow of
traffic, hence are an integral part of any vision system for autonomous
driving. While the use of deep learning is well-known in traffic signs classification due to the high accuracy results obtained using convolutional
neural networks (CNNs) (state of the art is 99.46%), little is known
about binarized neural networks (BNNs). Compared to CNNs, BNNs reduce the model size and simplify convolution operations and have shown
promising results in computationally limited and energy-constrained devices which appear in the context of autonomous driving.
This work presents a bottom-up approach for architecturing BNNs by
studying characteristics of the constituent layers. These constituent layers (binarized convolutional layers, max pooling, batch normalization,
fully connected layers) are studied in various combinations and with different values of kernel size, number of filters and of neurons by using the
German Traffic Sign Recognition Benchmark (GTSRB) for training. As
a result, we propose BNNs architectures which achieve an accuracy of
more than 90% for GTSRB (the maximum is 96.45%) and an average
greater than 80% (the maximum is 88.99%) considering also the Belgian
and Chinese datasets for testing. The number of parameters of these architectures varies from 100k to less than 2M. The accompanying material
of this paper is publicly available at 
https://github.com/apostovan21/BinarizedNeuralNetwork

# VNN Competition

## ONNX Models
We chose our 3 best models, one for each image size we have trained (30, 48, 64).
The models can be found in `onnx/` folder.

## Dataset
Although we've tested the model on German/Belgium/Chinese datasets, for verification purpose we suggest starting with German (GTSRB) datatset for testing. You have two ways to get it:
  - Download the entire dataset from [kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?datasetId=82373&language=Python).
  - Download just the *test* set from [drive](https://drive.google.com/drive/folders/1vKvawIPsUdAddezZudXJ6HRJ1TtmbOw0?usp=sharing). You should unzip the `Test.zip` file from `GTSRB_dataset` folder.
In te end you should have the folder `GTSRB_dataset` at the same level with `onnx` and `vnnlib` folder.

## Script's arguments
The script `src/generate_properties.py` can be executed without any arguments.
In this case it will use default values:
  - **seed**: 42
  - **epsilon**: [1, 3, 5, 10, 15]. It will generate vnnlib files for each epsilon from the list. In case you want to pass an argument for epsilon it should be an integer not a list.
  - **network**: all three networks from `onnx/` folder.
  - **n**: 10 (number of samples to generate)
  - **negate_spec**: False
  - **dont_extend**: False

## VNN lib Files
We have generated the `vnnlib` files for all three models, with the `epsilon = 1, 3, 5, 10, 15`, but using different `seed` value for each model, as following:
  - Model `3_30_30_*` seed = 42 (default)
  - Model `3_48_48_*` seed = 0
  - Model `3_64_64_*` seed = 1

## Example of calling the script:
```
./generate_properties.py --network onnx/3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx --seed 1
```