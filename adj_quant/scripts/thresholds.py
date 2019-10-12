import types
from typing import Tuple, Union, Dict, Callable

import tensorflow as tf
import numpy as np


def _check_evaluation_args(
        sess: tf.Session,
        input_node: Union[tf.Tensor, str],
        reference_nodes: dict,
        calibration_data: Callable[[], types.GeneratorType]):

    if not isinstance(sess, tf.Session):
        raise TypeError("Specified session must be a valid tf.Session object")

    # if not isinstance(calibration_data, DataGenerator):
    #     raise TypeError("Specified calibration data must be an instance of DataGenerator class")

    if not isinstance(input_node, (tf.Tensor, str)):
        raise TypeError("Input node must be whether a tensor or a node name")

    if not isinstance(reference_nodes, dict):
        raise TypeError("Reference nodes must be presented via a dictionary")

    if not all([isinstance(node_name, str) for node_name in reference_nodes.keys()]):
        raise TypeError("One of the specified reference nodes has a non-string name")

    if not all([isinstance(node, tf.Tensor) for node in reference_nodes.values()]):
        raise TypeError("One of the specified reference nodes is not a tensor")


def _eval_threshold_for_one_node(
        output_node: tf.Tensor,
        input_node: Union[tf.Tensor, str],
        sess: tf.Session, batch_size: int) -> Tuple[float, float]:

    res_min = [np.Inf for i in range(output_node.shape[3].value)]
    res_max = [-np.Inf for i in range(output_node.shape[3].value)]

    print('Eval initial threshold:',output_node.name)
    for i in range(int(10000/batch_size)+1):
        res = sess.run(output_node)
        res_min = np.minimum(res_min, np.min(np.min(np.min(res,axis=0),axis=0),axis=0))
        res_max = np.maximum(res_max, np.max(np.max(np.max(res,axis=0),axis=0),axis=0))

    return res_min, res_max


def eval_thresholds(sess, input_node, reference_nodes, batch_size) -> Dict[str, Dict[str, float]]:
    """

    Parameters
    ----------
    sess: tf.Session
    input_node: Union[tf.Tensor, str]
    reference_nodes: Dict[str, tf.Tensor]
    calibration_data: Callable[[], types.GeneratorType]

    Returns
    -------
    dict:
        A dictionary containing min and max thresholds for the reference nodes
    """
    #_check_evaluation_args(sess, input_node, reference_nodes, calibration_data)

    thresholds = dict()

    output_nodes = [node_name for node_name in reference_nodes if "weights" not in node_name.split('/')[-1]]
    weights_nodes = [node_name for node_name in reference_nodes if "weights" in node_name.split('/')[-1]]

    for w_node in weights_nodes:
        res = sess.run(reference_nodes[w_node])

        if 'dws' not in w_node:
            thresholds[w_node] = {"min": np.min(np.min(np.min(res,axis=0),axis=0),axis=0),
                                  "max": np.max(np.max(np.max(res,axis=0),axis=0),axis=0)}
        else:
            thresholds[w_node] = {"min": np.min(np.min(np.min(res,axis=0),axis=0),axis=1),
                                  "max": np.max(np.max(np.max(res,axis=0),axis=0),axis=1)}

    for o_node in output_nodes:
        res_min, res_max = _eval_threshold_for_one_node(reference_nodes[o_node],
                                                        input_node,
                                                        sess,batch_size)

        thresholds[o_node] = {"min": res_min,
                              "max": res_max}

    return thresholds
