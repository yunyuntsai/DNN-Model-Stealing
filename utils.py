import cv2
import PIL
from IPython.display import display
import numpy as np
import sys
from google.protobuf import text_format
import matplotlib
matplotlib.use('Agg')
import pylab as plt  # NOQA
caffe_root = '/home/exx/Documents/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe  # NOQA
caffe.set_mode_gpu()


def binary_to_nparray(binAdr):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(binAdr, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    arr = arr[0, ...]
    return arr


model_path_dict = {'caffenet': '/home/exx/Documents/pretrain_networks/AlexNet/retrained_caffe_model2150/',
                   }

net_fn_dict = {'caffenet': '/home/exx/Documents/pretrain_networks/AlexNet/alexnet_deploy.prototxt',
               }

param_fn_dict = {'caffenet': '/home/exx/Documents/pretrain_networks/AlexNet/retrained_caffe_model2150/alexnet_solver_iter_1000.caffemodel',
                 }

data_mean_dict = {'caffenet':
                  binary_to_nparray('/home/exx/Documents/dataset_all/randomdataset2150_train_mean.binaryproto'),
                  }


def init_model(model_name):
    model_path = model_path_dict[model_name]
    net_fn = net_fn_dict[model_name]
    param_fn = param_fn_dict[model_name]
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean=data_mean_dict[model_name],
                           input_scale=None
                           )

    return net


def read_image_rgb_noresize(fname):
    """
    Caffe doesn't keep the aspect ratio.

    NOTE: Keep the range of values in an image [0,255] before using
    preprocessing of a network. Make sure any function you use from scikit,
    PIL or opencv keeps that range. sckit for example always changes the output
    to [0,255] range regardless of the input range.
    """
    img = cv2.imread(fname)[:, :, ::-1]
    if img.ndim == 2 or img.shape[2] == 1:
        img = img.reshape((img.shape[0], img.shape[1], 1))
        img = np.concatenate((img, img, img), axis=2)
    return img


def image_resize(net, img):
    img = cv2.resize(img, tuple(net.image_dims))
    return img


def read_image_rgb(fname, net):
    img = read_image_rgb_noresize(fname)
    img = image_resize(net, img)
    return img


def preprocess(net, img):
    '''This is specific to imagenet classification nets
    that does only channel swapping and mean subtraction
    '''
    if hasattr(net, 'use_self_transformer') and net.use_self_transformer:
        return net.transformer.preprocess(net.inputs[0], img)
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    if hasattr(net, 'use_self_transformer') and net.use_self_transformer:
        return net.transformer.deprocess(net.inputs[0], img)
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def extract_feat(net, img, layer='prob'):
    src = net.blobs['data']
    src.data[0] = img
    net.forward()
    return net.blobs[layer].data[0].copy()


def showarray_noproc(a):
    """
    Note: uint8 is for visualization.
    """
    img = PIL.Image.fromarray(np.uint8(a))
    display(img)


def showarray_deproc(a, net):
    b = deprocess(net, a)
    img = PIL.Image.fromarray(np.uint8(b))
    display(img)
