import uuid
import os

import tensorflow as tf

from scripts.effnetb0.effnetb0_model_float import EffNetB0ModelFloat
from scripts.effnetb0.effnetb0_model_adjutable import EffNetB0ModelAdjustable
from scripts.effnetb0.effnetb0_model_fake_quant import EffNetB0ModelFakeQuant


def create_input_node(input_shape):
    with tf.Graph().as_default():
        return tf.placeholder(tf.float32, shape=input_shape, name="input_node")


def create_float_model(input_node, weights, mix_prec=False, output_node_name="output_node"):
    with input_node.graph.as_default():
        return EffNetB0ModelFloat(input_node, weights, mix_prec, output_node_name=output_node_name)


def create_adjustable_model(input_node, weights, tea_weights, thresholds, r_alpha=[0.5,1.3], r_beta=[-0.2,0.4], tea_model='eff_b0', weight_bits=8, act_bits=8, swish_bits=8, bias_bits=8, weight_const=False, bits_trainable=True, mix_prec=False):

    with input_node.graph.as_default():
        def _clip_grad_op(op, grad):
            x = op.inputs[0]
            x_min = op.inputs[1]
            x_max = op.inputs[2]
            cond = tf.logical_or(tf.less(x, x_min), tf.greater(x, x_max))
            return_grad = tf.where(cond, tf.zeros_like(grad, name="zero_grad"), grad)
            return return_grad, tf.constant(0, name="constant_min_grad"), tf.constant(0, name="constant_max_grad")

        # Register the gradient with a unique id
        grad_name = "MyClipGrad_" + str(uuid.uuid4())
        tf.RegisterGradient(grad_name)(_clip_grad_op)

        with input_node.graph.gradient_override_map({"Round": "Identity", "ClipByValue": grad_name}):
            effnetb0_model_quantized = EffNetB0ModelAdjustable(input_node, weights, thresholds, r_alpha, r_beta, weight_bits, act_bits, swish_bits, bias_bits, weight_const, bits_trainable, mix_prec)

        with tf.name_scope("float_model"):
            if tea_model == 'eff_b0':
                effnetb0_model_float = EffNetB0ModelFloat(input_node, tea_weights, mix_prec)

    return effnetb0_model_float, effnetb0_model_quantized


def create_fakequant_model(input_node, weights, thresholds):
    with input_node.graph.as_default():
        return EffNetB0ModelFakeQuant(input_node, weights, thresholds)


def prepare_effnetb0_environment(pickle_path, output_dir, input_size=None, suffix="_weights.pickle"):
    if not isinstance(pickle_path, str):
        raise TypeError("Specified file name must be a string")

    if not isinstance(output_dir, str):
        raise TypeError("Specified name of the output directory must be a string")

    pickle_path = os.path.realpath(pickle_path)

    if not os.path.exists(pickle_path):
        raise FileNotFoundError("File '{}' not found".format(pickle_path))

    model_base_name = os.path.basename(os.path.realpath(pickle_path)).replace(suffix, "")

    model_output_dir = os.path.join(output_dir, model_base_name)
    checkpoint_folder = os.path.join(model_output_dir, "ckpt")
    best_checkpoint_folder = os.path.join(model_output_dir, "best_ckpt")

    thresholds_path = os.path.join(model_output_dir, model_base_name + "_thresholds.pickle")
    fakequant_output_path = os.path.join(model_output_dir, model_base_name + "_fakequant.pb")

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    if not os.path.exists(best_checkpoint_folder):
        os.makedirs(best_checkpoint_folder)

    if input_size is not None:
        img_size = int(input_size)
    else:
        try:
            img_size = int(model_base_name.split('_')[-1])
        except (TypeError, ValueError):
            print("Unable to retrieve the iput size from the model name. 224x224 will be used")
            img_size = 224

    input_shape = (None, img_size, img_size, 3)

    print("Model: '{}'".format(model_base_name))
    print("INPUT_SHAPE:", input_shape)

    return input_shape, checkpoint_folder, best_checkpoint_folder, thresholds_path, fakequant_output_path
