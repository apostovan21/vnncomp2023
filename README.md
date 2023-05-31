# German Traffic Signs Recognition Benckmark (GTSRB)

This benchmark set includes 3 binarized neural networks (BNNs), provided in the `onnx/` folder, which were trained using GTSRB. The BNNs contain layers like binarized convolutions, max pooling, batch normalization and fully connected. For the binarized convolution layer we used Larq library [1]. For more details see [2].

The verification properties, provided in the `vnnlib/` folder, represent adversarial robustness to infinity norm perturbations around 0 whose radius of `epsilon` was randomly chosen, see below.

## Dataset details
Although we've tested the model on German/Belgium/Chinese datasets, for verification purpose we suggest starting with German (GTSRB) datatset for testing. As this dataset isn't included in any python package, we have added the test set into `GTSRB_dataset` folder of this repository. 

## ONNX Models
We chose our 3 best models in terms of accuracy, one for each image size we have trained: 30x30, 48x48, 64x64 (px x px). The models can be found in `onnx/` folder.

## VNNLIB Files
We have generated the `vnnlib` files for all three models, with the `epsilon = 1, 3, 5, 10, 15`, but using different `seed` value for each model, as following:
  - Model `3_30_30_*` seed = 42 (default)
  - Model `3_48_48_*` seed = 0
  - Model `3_64_64_*` seed = 1

## How to generate VNNLIB specifications 

### Script arguments
The script `src/generate_properties.py` can be executed without any arguments.
In this case it will use default values:
  - **seed**: 42
  - **epsilon**: [1, 3, 5, 10, 15]. It will generate vnnlib files for each epsilon from the list. In case you want to pass a specific value for epsilon it should be an integer not a list.
  - **network**: all three networks from `onnx/` folder.
  - **n**: 10 (number of samples to generate)
  - **negate_spec**: False
  - **dont_extend**: False

### Example of calling the script:
```
./generate_properties.py --network onnx/3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx --seed 1
```

## Citation
If you find the repo useful, fell free to cite us:

```
@article{postovan2023architecturing,
  title={Architecturing Binarized Neural Networks for Traffic Sign Recognition},
  author={Postovan, Andreea and Era{\c{s}}cu, M{\u{a}}d{\u{a}}lina},
  journal={arXiv preprint arXiv:2303.15005},
  year={2023}
}
```

## References
[1] https://docs.larq.dev/larq/

[2] Andreea Postovan and Mădălina Eraşcu. Architecturing Binarized Neural Networks for Traffic Sign Recognition. https://arxiv.org/abs/2303.15005
