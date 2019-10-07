import tensorflow as tf
import numpy as np
from copy import deepcopy


def _retrieve_data_links(op_scope, bn_scope=None, bias_scope=None, base_scope='efficientnet_b0'):
    data = {}
    
    if "depthwise" in op_scope:
        data['weights'] = base_scope + '/' + op_scope + '/depthwise_weights:0'
    else:
        data['weights'] = base_scope + '/' + op_scope + '/weights:0'
    
    if bn_scope is not None:
        data["gamma"] = base_scope + '/' + bn_scope + '/gamma:0'
        data["beta"] = base_scope + '/' + bn_scope + '/beta:0'
        data["mean"] = base_scope + '/' + bn_scope + '/moving_mean:0'
        data["variance"] = base_scope + '/' + bn_scope + '/moving_variance:0'
    
    if bias_scope is not None:
        data["bias"] = base_scope + '/' + bias_scope + '/biases:0'
    
    return data


def _gw(sess: tf.Session, op_scope, bn_scope=None, bias_scope=None, base_scope='efficientnet_b0'):
    return sess.run(_retrieve_data_links(op_scope, bn_scope, bias_scope, base_scope))


def _get_cascade_cell_name(operation_names, cell_index):
    substr = "expanded_conv_" + str(cell_index)
    for op_name in operation_names:
        if substr in op_name:
            return op_name.split('/')[1]
    return None


def _fold_bn(weights, moving_mean, moving_variance, gamma, beta, eps=10**-3):
    scale_factor = gamma / np.sqrt(moving_variance + eps)
    bias = beta - moving_mean * scale_factor
    weights = np.multiply(weights, scale_factor)
    return weights, bias


def _fold_bn_dws(weights, moving_mean, moving_variance, gamma, beta, eps=10**-3):
    scale_factor = gamma / np.sqrt(moving_variance + eps)
    bias = beta - moving_mean * scale_factor
    weights = np.multiply(weights, np.expand_dims(scale_factor, 1))
    return weights, bias


def _fold_weights(weights_dict):
    folded_weights = dict()
    
    for ln, w_data in weights_dict.items():
        new_w_data = {}
        w = w_data["weights"]

        print(ln)
        
        if "dws" in ln:
            w, b = _fold_bn_dws(w_data["weights"], 
                                w_data["mean"], 
                                w_data["variance"], 
                                w_data["gamma"], 
                                w_data["beta"])
            new_w_data["weights"] = w
            new_w_data["bias"] = b
        elif "se" in ln or 'output' in ln:
            new_w_data = w_data
        else:
            w, b = _fold_bn(w_data["weights"], 
                            w_data["mean"], 
                            w_data["variance"], 
                            w_data["gamma"], 
                            w_data["beta"])
            new_w_data["weights"] = w
            new_w_data["bias"] = b
        
        folded_weights[ln] = deepcopy(new_w_data)
    
    return folded_weights


def get_weights_for_effnetb0(model: tf.Graph) -> dict:
    """Extracts weights dictionary from any MNasNet model hosted at
    **www.tensorflow.org/lite/models**

    Weights are preprocessed in order to eliminate operations, related to batch normalization.

    Parameters
    ----------
    model: tf.Graph
        A static graph, from which weights data must be extracted.

    Returns
    -------
    dict:
        A dictionary containing layers' weights data (including biases)
    """
    with tf.Session(graph=model) as sess:
    
        operation_names = [op.name for op in model.get_operations() if op.type=="Const" and op.name.find('Mean')==-1]

        weights = {}

        # stem cell
        weights["Conv"] = _gw(sess, "Conv", "Conv/BatchNorm")

        # 0-th cascade cell
        weights["expanded_conv/dws"] = _gw(sess, "expanded_conv/depthwise", "expanded_conv/depthwise/BatchNorm")
        weights["expanded_conv/se_1"] = _gw(sess, "expanded_conv/se_1", bias_scope="expanded_conv/se_1")
        weights["expanded_conv/se_2"] = _gw(sess, "expanded_conv/se_2", bias_scope="expanded_conv/se_2")
        weights["expanded_conv/project"] = _gw(sess, "expanded_conv/project", "expanded_conv/project/BatchNorm")

        # intermediate cascade cells
        for cell_index in range(1, 16):
            cell_scope = _get_cascade_cell_name(operation_names, cell_index)
            # expand -> dws -> project
            weights[cell_scope + "/expand"] = _gw(sess, cell_scope + "/expand", cell_scope + "/expand/BatchNorm")
            weights[cell_scope + "/dws"] = _gw(sess, cell_scope + "/depthwise", cell_scope + "/depthwise/BatchNorm")
            weights[cell_scope + "/se_1"] = _gw(sess, cell_scope + "/se_1", bias_scope=cell_scope + "/se_1")
            weights[cell_scope + "/se_2"] = _gw(sess, cell_scope + "/se_2", bias_scope=cell_scope + "/se_2")
            weights[cell_scope + "/project"] = _gw(sess, cell_scope + "/project", cell_scope + "/project/BatchNorm")

        # output cell
        weights["Conv_1"] = _gw(sess, "Conv_1", "Conv_1/BatchNorm")
        weights["output"] = _gw(sess, "Logits/Conv2d_1c_1x1", bias_scope="Logits/Conv2d_1c_1x1")
    
    return _fold_weights(weights)
