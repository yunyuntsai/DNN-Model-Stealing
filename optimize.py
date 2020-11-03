import utils
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as cnstOpt
import sys


"""
Parameters to LBFGS.
The following values work well for the models listed in utils. We had to
change these for some other models.
"""
factr = 10000000.0
pgtol = 1e-05


def objective_guide_euclidean(dst, g_feat, verbose=True):
    """
    The objective function used when backward does all the computation.
    Exceptions are when there is an in-place operation or the operation was
    part of another layer that doesn't exist at deploy time.
    """
    x = dst.data[0].copy()
    y = g_feat
    ch = x.shape[0]
    x = x.reshape(ch, -1)
    y = y.reshape(ch, -1)
    A = 2 * (x - y)  # compute the matrix of dot-products with guide features
    diff = (x - y).flatten()
    obj = np.dot(diff, diff)
    if verbose is True:
        print np.sqrt(obj)
    dst.diff[0].reshape(ch, -1)[:] = A[:]
    return obj


def objective_guide_relu(dst, g_feat, rfeat, verbose=True):
    """
    To be used when forward proping until after relu (e.g. 'prob')
    and the objective is on the output after relu.

    dst: net.blobs[end] before relu
    rfeat: net.blobs[end].data after relu
    """

    x = rfeat
    y = g_feat
    ch = x.shape[0]
    x = x.reshape(ch, -1)
    y = y.reshape(ch, -1)
    A = 2 * (x - y)  # compute the matrix of dot-products with guide features
    diff = (x - y).flatten()
    obj = np.dot(diff, diff)
    if verbose is True:
        print np.sqrt(obj)
    lfeat = dst.data[0].copy()
    dst.diff[0].reshape(ch, -1)[:] = A[:] * (lfeat.reshape(ch, -1)[:] > 0)
    return obj


def objective_guide_label_noloss(prob_layer, diff_layer, g_feat, verbose=True):
    """
    To be used when optimizing on prediction but from 'prob' layer
    and there is no explicit loss layer.

    Taken from softmax_loss_layer.cpp forward()
    """
    g_label = np.argmax(g_feat)
    obj = -np.log(max(prob_layer.data[0][g_label], sys.float_info.min))
    if verbose:
        print obj, np.argmax(prob_layer.data[0])
    diff_layer.diff[0][...] = prob_layer.data[0]
    diff_layer.diff[0][g_label] = prob_layer.data[0][g_label] - 1.
    return obj


def get_layer_objective(layer, net):
    layer_names = [net._layer_names[i] for i in range(len(net._layer_names))]
    idx = layer_names.index(layer)
    if net.layers[idx].type == 'Softmax':
        print "Objective: label_noloss"
        return objective_guide_label_noloss
    if idx+1 < len(net.layers):
        if net.layers[idx+1].type == 'ReLU':
            print "Objective: relu"
            return objective_guide_relu
        else:
            print "Objective: euclidean"
            return objective_guide_euclidean
    return None


def get_next_layer(layer, net):
    layer_names = net.blobs.keys()
    idx = layer_names.index(layer)
    if idx == len(layer)-1:
        return layer_names[idx]
    return layer_names[idx+1]


def calc_gstep(cs_x, net, g_feat, end, objective, verbose=True):

    src = net.blobs['data']  # input image is stored in Net's 'data' blob
    dst = net.blobs[end]
    src.data[0][...] = cs_x.reshape(src.data[0].shape)
    if objective == objective_guide_euclidean:
        net.forward(end=end)
        obj = objective(dst, g_feat, verbose)
        net.backward(start=end)
    elif objective == objective_guide_relu:
        # two forwrad props to handle in place operations like relu
        next_layer = get_next_layer(end, net)
        net.forward(end=next_layer)
        rfeat = dst.data[0].copy()
        net.forward(end=end)
        # specify the optimization objective
        obj = objective(dst, g_feat, rfeat, verbose)
        net.backward(start=end)
    elif objective == objective_guide_label_noloss:
        diff_layer = net.blobs.keys()[net.blobs.keys().index('prob') - 1]
        net.forward(end='prob')
        obj = objective(net.blobs['prob'], net.blobs[diff_layer],
                        g_feat, verbose)
        net.backward(start=diff_layer)
    else:
        raise Exception("Unknown objective function.")

    g = src.diff[0]

    return obj, g.flatten().astype(float)


def constOptimize(net, base_img, guide_img, objective, iter_n, max_thres,
                  end, factr=factr, pgtol=pgtol, verbose=True):
    proc_base = utils.preprocess(net, base_img)
    proc_guide = utils.preprocess(net, guide_img)
    src = net.blobs['data']
    ch, h, w = proc_base.shape
    src.reshape(1, ch, h, w)
    # allocate image for network-produced details
    src, dst = net.blobs['data'], net.blobs[end]
    src.data[0] = proc_guide
    net.forward(end='prob')
    guide_features = dst.data[0].copy()

    up_bnd = proc_base + max_thres
    lw_bnd = proc_base - max_thres
    mean_arr = net.transformer.mean['data']
    if mean_arr.ndim == 1:
        mean_arr = mean_arr.reshape((3, 1, 1))
    up_bnd = np.minimum(up_bnd, 255 - mean_arr)
    lw_bnd = np.maximum(lw_bnd, 0 - mean_arr)
    bound = zip(lw_bnd.flatten(), up_bnd.flatten())
    src.data[0] = proc_base
    x, f, d = cnstOpt(calc_gstep, proc_base.flatten().astype(float),
                      args=(net, guide_features, end, objective, verbose),
                      bounds=bound, maxiter=iter_n, iprint=0, factr=factr,
                      pgtol=pgtol)

    return x.reshape(proc_base.shape), f, d


def optimize_single(seed_fname, guide_fname, model_name, layer, iter_n,
                    max_thres, net, verbose=True):
    seed = utils.read_image_rgb(seed_fname, net)
    guide = utils.read_image_rgb(guide_fname, net)
    objective = get_layer_objective(layer, net)
    ad, bc, cd = constOptimize(
        net, seed, guide, iter_n=iter_n, max_thres=max_thres,
        end=layer, objective=objective, verbose=verbose)
    print cd

    return utils.deprocess(net, ad), bc, cd
