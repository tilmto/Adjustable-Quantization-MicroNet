import types
import tensorflow as tf
from scripts.effnetb0 import combined_tools as ct
from scripts import helpers as hp
import numpy as np

_strides = (1,2,1,2,1,2,1,1,1,1,1,2,1,1,1,1)


# EffNetB0 related exceptions
class EffNetB0WeightsShapeError(Exception):
    pass


class EffNetB0WeightsKeyError(Exception):
    pass


class EffNetB0BuildError(Exception):
    pass


class EffNetB0InitError(Exception):
    pass


class EffNetB0ModelFloat(object):
    """Creates EffNetB0 model based on the specified weights.

    Weights must be prepared, so that all batch normalization operations are fused with
    corresponding convolution operations.

    Properties
    ----------
    graph: tf.Graph
        A TensorFlow graph that hosts the model data
    input_node: tf.Tensor
        The input node of the EffNetB0 model
    output_node: tf.Tensor
        The output node of the EffNetB0 model
    reference_nodes: dict
        A dictionary containing all tensors which output data is necessary for
        calculating the quantization thresholds
    """

    def __init__(self, input_node, weights, mix_prec=False, output_node_name="output_node"):
        """

        Parameters
        ----------
        input_node
        weights
        output_node_name
        """
        self._input_node = input_node
        self._graph = tf.get_default_graph()

        self._weights = weights

        self.quant = not mix_prec

        self._output_node_name = output_node_name

        self._reference_nodes = dict()

        self._output_node = self._create_model()

    @property
    def graph(self):
        return self._graph

    @property
    def output_node(self):
        return self._output_node

    @property
    def input_node(self):
        return self._input_node

    @property
    def reference_nodes(self):
        return dict(self._reference_nodes)

    def _add_reference_node(self, node: tf.Tensor):
        ref_node_name = node.name.split(":")[0]
        self._reference_nodes[ref_node_name] = node

    def _get_layer_weights(self, layer_name):
        if layer_name not in self._weights:
            raise EffNetB0WeightsKeyError("Weights for the layer '{}' are not provided".format(layer_name))

        return self._weights[layer_name]

    def _cell_output(self, net, output_type="identity", subscope='output'):
        if output_type not in ["identity", "relu", "fixed"]:
            raise EffNetB0BuildError("Unsupported type of cell output was specified: " + str(output_type))

        if output_type == "relu":
            net = tf.nn.relu(net, name="Relu")
        
        net = tf.identity(net, "output")
        self._add_reference_node(net)

        return net
    
    def _swish(self,net):
        return net*tf.nn.sigmoid(net)

    def _create_weights_node(self, weights_data, quant=True, dws=False):
        net =  tf.constant(weights_data, tf.float32, name="weights")
        self._add_reference_node(net)
        return net

    def _expanded_conv(self, net, cell_scope, strides=1):

        input_node = net

        with tf.name_scope(cell_scope):
            with tf.name_scope("expand"):
                net = self._build_conv(net, self._get_layer_weights(cell_scope + "/expand"), 1)
                net = self._swish(net)
                net = self._cell_output(net, "identity", subscope='output')

            with tf.name_scope('dws'):
                net = self._build_dws(net, self._get_layer_weights(cell_scope + "/dws"), strides)
                net = self._swish(net)
            
            net_se = tf.reduce_mean(net, [1, 2], keepdims=True)
            
            with tf.name_scope('se_1'):
                net_se = self._build_conv(net_se,self._get_layer_weights(cell_scope + "/se_1"), 1, quant=self.quant)
                net_se = self._swish(net_se)
                net_se = self._cell_output(net_se, "identity", subscope='output')
            
            with tf.name_scope('se_2'):
                net_se = self._build_conv(net_se,self._get_layer_weights(cell_scope + "/se_2"), 1, quant=self.quant)
                net_se = tf.nn.sigmoid(net_se)
            
            net = net*net_se
            net = self._cell_output(net, "identity")

            with tf.name_scope("project"):
                net = self._build_conv(net, self._get_layer_weights(cell_scope + "/project"), 1)
                net = self._cell_output(net, "identity", subscope='output')

            if strides == 1 and net.shape[3].value == input_node.shape[3].value:
                with tf.name_scope("add"):
                    net = tf.add(net, input_node)

        return net
    
    def _build_dws(self, input_node, weights_and_bias, strides, quant=True):
        weights_node = self._create_weights_node(weights_and_bias["weights"],quant,dws=True)
        
        op_output = tf.nn.depthwise_conv2d(input_node, 
                                           weights_node, 
                                           (1, strides, strides, 1), 
                                           padding="SAME")

        if "bias" in weights_and_bias:
            bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output
    
    def _build_conv(self, input_node, weights_and_bias, strides, quant=True):
        weights_node = self._create_weights_node(weights_and_bias["weights"],quant)
        
        op_output = tf.nn.conv2d(input_node, 
                                 weights_node, 
                                 (1, strides, strides, 1), 
                                 padding="SAME")
        
        if "bias" in weights_and_bias:
            bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output
    
    def _build_fc(self, input_node, weights_and_bias, quant=True):
        weights_node = self._create_weights_node(weights_and_bias["weights"],quant)
        
        op_output = tf.matmul(input_node, weights_node)
        
        if "bias" in weights_and_bias:
            bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output

    def _build(self):

        with tf.name_scope("input_data"):
            net = self._cell_output(self._input_node, "fixed")

        # Conv
        with tf.name_scope("Conv"):
            net = self._build_conv(net, self._get_layer_weights("Conv"), 2, quant=self.quant)
            net = self._swish(net)
            net = self._cell_output(net, "identity", subscope='output')

        # expanded_conv_0
        with tf.name_scope('expanded_conv_0'):
            with tf.name_scope('dws'):
                net = self._build_dws(net, self._get_layer_weights("expanded_conv/dws"), _strides[0])
                net = self._swish(net)
                net = self._cell_output(net, "identity", subscope='output')
            
            net_se = tf.reduce_mean(net, [1, 2], keepdims=True)
            
            with tf.name_scope('se_1'):
                net_se = self._build_conv(net_se,self._get_layer_weights("expanded_conv/se_1"),1, quant=self.quant)
                net_se = self._swish(net_se)
                net_se = self._cell_output(net_se, "identity", subscope='output')
            
            with tf.name_scope('se_2'):
                net_se = self._build_conv(net_se,self._get_layer_weights("expanded_conv/se_2"),1, quant=self.quant)
                net_se = tf.nn.sigmoid(net_se)
            
            net = net*net_se
            net = self._cell_output(net, "identity")
            
            with tf.name_scope("project"):
                net = self._build_conv(net, self._get_layer_weights("expanded_conv/project"), 1)
                net = self._cell_output(net, "identity")

        # series of cells with the similar structure
        for i in range(1, 16):
            cell_scope = "expanded_conv_" + str(i)
            dws_stride = _strides[i]
            net = self._expanded_conv(net, cell_scope, dws_stride)
        
        # Conv_1
        with tf.name_scope("Conv_1"):
            net = self._build_conv(net, self._get_layer_weights("Conv_1"), 1)
            net = self._swish(net)
            net = self._cell_output(net, "identity", subscope='output')
        
        net = tf.reduce_mean(net, [1, 2], keepdims=True)

        # output
        with tf.name_scope("output"):
            net = self._build_conv(net, self._get_layer_weights("output"), 1, quant=self.quant)
        
        net = tf.squeeze(net,[1, 2])

        return tf.identity(net, name=self._output_node_name)

    def _create_model(self):
        try:
            output_node = self._build()
        except tf.errors.InvalidArgumentError as e:
            raise EffNetB0WeightsShapeError("Specified weights for the model are inconsistent. " + e.message)

        return output_node



if __name__ == "__main__":
    weights_path = "/home/tilmto/Ericpy/tensorflow/micronet/pretrained/efficient_b0/frozen_efficientnet_b0_weights.pickle"
    WEIGHTS = hp.load_pickle(weights_path)
    float_model = ct.create_float_model(ct.create_input_node((None,224,224,3)), WEIGHTS)

    with tf.Session(graph=float_model.graph) as sess:
        output = sess.run(float_model.output_node,feed_dict={float_model.input_node:np.ones((10,224,224,3))})
        print(output.shape)