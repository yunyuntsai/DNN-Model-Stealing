# Under Convnet

Implementation of the paper ["Adversarial Manipulation of Deep
Representations"](http://arxiv.org/abs/1511.05122).  The code can manipulate
the representation of an image in a deep neural network (DNN) to mimic those of
other natural images, with only minor, imperceptible perturbations to the
original image.

## Dependencies

This code is written in python. To use it you will need:

* [Caffe](http://caffe.berkeleyvision.org)

In `utils.py` set `caffe_root` to your installation path.

## Reference

If you found this code useful, please cite the following paper:

    @article{sabour2015adversarial,
        title={Adversarial Manipulation of Deep Representations},
        author={Sabour, Sara and Cao, Yanshuai and Faghri, Fartash and Fleet,
            David J},
        journal={arXiv preprint arXiv:1511.05122},
        year={2015}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
