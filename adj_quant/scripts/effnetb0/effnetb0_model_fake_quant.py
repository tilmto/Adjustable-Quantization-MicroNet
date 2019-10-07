import tensorflow as tf
import numpy as np

from .effnetb0_model_float import EffNetB0ModelFloat
from .effnetb0_model_float import EffNetB0InitError


def _check_init_parameters(thresholds):
    if not isinstance(thresholds, dict):
        raise EffNetB0InitError("Precalculated thresholds must be presented via a dictionary")

    if not all([isinstance(th_name, str) for th_name in thresholds.keys()]):
        raise EffNetB0InitError("All names of thresholds must be strings")

    if not all([isinstance(th_data, dict) for th_data in thresholds.values()]):
        raise EffNetB0InitError("Thresholds must be packed in pairs into dictionaries")

    if not all([("min" in th_data and "max" in th_data) for th_data in thresholds.values()]):
        raise EffNetB0InitError("Each reference node must have corresponding minimal and maximal thresholds")


def _get_name_scope():
    return tf.get_default_graph().get_name_scope()


class EffNetB0ModelFakeQuant(EffNetB0ModelFloat):

    def __init__(self, input_node, weights, thresholds, mix_prec=False, output_node_name="output_node"):
        _check_init_parameters(thresholds)
        self._initial_thresholds = thresholds
        super().__init__(input_node, weights, mix_prec, output_node_name)

    def _get_thresholds(self, thresholds_name):
        return np.array(self._initial_thresholds[thresholds_name]["min"]), np.array(self._initial_thresholds[thresholds_name]["max"])

    def _create_weights_node(self, weights_data, quant=True):
        weights_name_scope = _get_name_scope() + "/weights"
        
        w_min, w_max = self._get_thresholds(weights_name_scope)
        
        weights_node = tf.constant(weights_data, tf.float32, name="weights")
        self._add_reference_node(weights_node)

        if quant:
            quantized_weights = tf.fake_quant_with_min_max_args(weights_node,
                                                                w_min,
                                                                w_max,
                                                                name="quantized_weights")
            return quantized_weights
        
        else:
            return weights_node


    def _cell_output(self, net, output_type=None, subscope='output'):

        output_name_scope = _get_name_scope() + "/" + subscope

        if output_type == "fixed":
            pass
            #i_min, i_max = -1, 1
        else:
            i_min, i_max = self._get_thresholds(output_name_scope)
            net = tf.fake_quant_with_min_max_args(net, i_min, i_max, name="output")

        self._add_reference_node(net)

        return net
