import tensorflow as tf

from .effnetb0_model_fake_quant import EffNetB0ModelFakeQuant


def _rounding_fn(node, name):
    return tf.round(node, name)


def _get_name_scope():
    return tf.get_default_graph().get_name_scope()


def _fixed_quant_signed(input_node, min_th, max_th, bits=8, name="fixed_signed_data"):
    th_width = max_th - min_th

    with tf.name_scope(name):
        min_th_node = tf.constant(min_th, tf.float32, name="min_th")
        th_width_node = tf.constant(th_width, tf.float32, name="th_width")

        # scale
        with tf.name_scope("scale"):
            q_range = 2. ** bits - 1.

            eps = tf.constant(10. ** -7, dtype=tf.float32, name='eps')
            scale_node = tf.div(q_range, th_width_node + eps, "new_input_scale")

        with tf.name_scope("quantized_bias"):
            quant_bias = tf.multiply(min_th_node, scale_node, name="scaling")
            quant_bias = _rounding_fn(quant_bias, name="rounding")

        with tf.name_scope("discrete_input_data"):
            net = tf.multiply(input_node, scale_node, name="scaling")
            net = _rounding_fn(net, name='rounding')
            net = tf.clip_by_value(net - quant_bias, clip_value_min=0, clip_value_max=2 ** bits - 1) + quant_bias

            descrete_input_data = tf.div(net, scale_node, name="discrete_data")

    return descrete_input_data


class EffNetB0ModelAdjustable(EffNetB0ModelFakeQuant):

    def __init__(self, input_node, weights, thresholds, r_alpha=[0.5,1.3], r_beta=[-0.2,0.4], weight_bits=8, act_bits=8, swish_bits=8, bias_bits=8, weight_const=False, bits_trainable=True, mix_prec=False, output_node_name="output_node"):
        self._adjusted_thresholds = dict()
        self._ths_vars = list()
        self.r_alpha = r_alpha
        self.r_beta = r_beta
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_const = weight_const

        self.bits_trainable = bits_trainable

        self.swish_bits = swish_bits
        self.bias_bits = bias_bits

        self.swish_bits_list = []

        self.quant_info = {}
        self.quant_info['Conv'] = {'weight':4,'act':8}
        self.quant_info['Conv_1'] = {'weight':4,'act':8}
        self.quant_info['output'] = {'weight':4,'act':tf.constant(0.)}
        
        for i in range(16):
            scope = 'expanded_conv_' + str(i)

            if i==0:
                expand_dict = {'weight':tf.constant(0.),'act':tf.constant(0.)}
            else:
                expand_dict = {'weight':4,'act':8}
                
            dws_dict = {'weight':4,'act':8}
            se_1_dict = {'weight':4,'act':8}
            se_2_dict = {'weight':4,'act':8}
            project_dict = {'weight':4,'act':8}
            
            self.quant_info[scope] = {'expand':expand_dict,'dws':dws_dict,'se_1':se_1_dict,'se_2':se_2_dict,'project':project_dict}

        self.num_sf = 0

        super().__init__(input_node, weights, thresholds, mix_prec, output_node_name)

    
    def _add_thresholds(self, name, min_th, max_th):
        self._adjusted_thresholds[name] = {"min": min_th, "max": max_th}

    def _add_th_var(self, th_variable):
        self._ths_vars.append(th_variable)

    @property
    def adjusted_thresholds(self):
        return {th_name: dict(th_data) for th_name, th_data in self._adjusted_thresholds.items()}

    @property
    def variables(self):
        return list(self._ths_vars)

    @property
    def initializer(self):
        return list(var.initializer for var in self._ths_vars)

    def _adjustable_quant_signed(self, input_node, min_th, max_th, out_name, bits=8, name="adjust_signed_data"):
        self.num_sf += input_node.shape[3].value
        th_width = max_th - min_th

        with tf.name_scope(name):
            with tf.name_scope("thresholds"):
                min_th_node = tf.constant(min_th, tf.float32, name="min_th")
                th_width_node = tf.constant(th_width, tf.float32, name="th_width")

                alpha = tf.Variable(tf.constant(1.0, shape=[input_node.shape[3],]), dtype=tf.float32, name="th_width_scale")
                beta = tf.Variable(tf.constant(0.0, shape=[input_node.shape[3],]), dtype=tf.float32, name="th_shift_percent")

                alpha_constrained = tf.clip_by_value(alpha, self.r_alpha[0], self.r_alpha[1], name="th_width_scale_constrain")
                beta_constrained = tf.clip_by_value(beta, self.r_beta[0], self.r_beta[1], name="th_shift_percent_constrain")

                adjusted_th_width = tf.multiply(th_width_node, alpha_constrained, name="adjusted_width")
                shift_node = tf.multiply(th_width_node, beta_constrained, name="min_th_shift")

                adjusted_min_th = tf.add(min_th_node, shift_node, name="adjusted_min_th")

                adjusted_max_th = tf.add(adjusted_min_th, adjusted_th_width, name="adjusted_max_th")

            # scale
            with tf.name_scope("scale"):
                if self.bits_trainable:
                    bits_scale = tf.Variable(tf.constant(1.0, shape=[input_node.shape[3],]), dtype=tf.float32, name='bits_scale')
                    bits = bits_scale * tf.to_float(bits)
                    bits = _rounding_fn(bits, name='bits_rounding')
                    bits = tf.clip_by_value(bits, 2, 8, name="bits_clip")

                q_range = 2. ** bits - 1.

                eps = tf.constant(10. ** -7, shape=[input_node.shape[3],], dtype=tf.float32, name='eps')
                scale_node = tf.div(q_range, adjusted_th_width + eps, "new_input_scale")
                scale_node = tf.reshape(scale_node,[1,1,1,-1])

            with tf.name_scope("quantized_bias"):
                quant_bias = tf.multiply(adjusted_min_th, scale_node, name="scaling")
                quant_bias = _rounding_fn(quant_bias, name="rounding")

            with tf.name_scope("discrete_input_data"):
                net = tf.multiply(input_node, scale_node, name="scaling")
                net = _rounding_fn(net, name='rounding')
                net = tf.clip_by_value(net - quant_bias, clip_value_min=0., clip_value_max=tf.reshape(2.**bits-1., [1,1,1,-1])) + quant_bias

                descrete_input_data = tf.div(net, scale_node, name="discrete_data")

        self._add_thresholds(out_name, adjusted_min_th, adjusted_max_th)
        self._add_th_var(alpha)
        self._add_th_var(beta)

        return descrete_input_data, bits
    

    def _adjustable_dws_quant_signed(self, input_node, min_th, max_th, out_name, bits=8, name="adjust_signed_data"):
        self.num_sf += input_node.shape[2].value
        th_width = max_th - min_th

        with tf.name_scope(name):
            with tf.name_scope("thresholds"):
                min_th_node = tf.constant(min_th, tf.float32, name="min_th")
                th_width_node = tf.constant(th_width, tf.float32, name="th_width")

                alpha = tf.Variable(tf.constant(1.0, shape=[input_node.shape[2],]), dtype=tf.float32, name="th_width_scale")
                beta = tf.Variable(tf.constant(0.0, shape=[input_node.shape[2],]), dtype=tf.float32, name="th_shift_percent")

                alpha_constrained = tf.clip_by_value(alpha, self.r_alpha[0], self.r_alpha[1], name="th_width_scale_constrain")
                beta_constrained = tf.clip_by_value(beta, self.r_beta[0], self.r_beta[1], name="th_shift_percent_constrain")

                adjusted_th_width = tf.multiply(th_width_node, alpha_constrained, name="adjusted_width")
                shift_node = tf.multiply(th_width_node, beta_constrained, name="min_th_shift")

                adjusted_min_th = tf.add(min_th_node, shift_node, name="adjusted_min_th")

                adjusted_max_th = tf.add(adjusted_min_th, adjusted_th_width, name="adjusted_max_th")

            # scale
            with tf.name_scope("scale"):
                if self.bits_trainable:
                    bits_scale = tf.Variable(tf.constant(1.0, shape=[input_node.shape[2],]), dtype=tf.float32, name='bits_scale')
                    bits = bits_scale * tf.to_float(bits)
                    bits = _rounding_fn(bits, name='bits_rounding')
                    bits = tf.clip_by_value(bits, 2, 8, name="bits_clip")

                q_range = 2. ** bits - 1.

                eps = tf.constant(10. ** -7, shape=[input_node.shape[2],], dtype=tf.float32, name='eps')
                scale_node = tf.div(q_range, adjusted_th_width + eps, "new_input_scale")
                
                scale_node = tf.reshape(scale_node,[1,1,-1,1])
                adjusted_min_th = tf.reshape(adjusted_min_th,[1,1,-1,1])

            with tf.name_scope("quantized_bias"):
                quant_bias = tf.multiply(adjusted_min_th, scale_node, name="scaling")
                quant_bias = _rounding_fn(quant_bias, name="rounding")

            with tf.name_scope("discrete_input_data"):
                net = tf.multiply(input_node, scale_node, name="scaling")
                net = _rounding_fn(net, name='rounding')
                net = tf.clip_by_value(net - quant_bias, clip_value_min=0., clip_value_max=tf.reshape(2.**bits-1., [1,1,-1,1])) + quant_bias
                descrete_input_data = tf.div(net, scale_node, name="discrete_data")

        self._add_thresholds(out_name, adjusted_min_th, adjusted_max_th)
        self._add_th_var(alpha)
        self._add_th_var(beta)

        return descrete_input_data, bits


    def _adjustable_quant_signed_static(self, input_node, min_th, max_th, out_name, bits=8, name="adjust_signed_data"):
        self.num_sf += input_node.shape[3].value
        th_width = max_th - min_th

        with tf.name_scope(name):
            with tf.name_scope("thresholds"):
                min_th_node = tf.constant(min_th, tf.float32, name="min_th")
                th_width_node = tf.constant(th_width, tf.float32, name="th_width")

                alpha = tf.Variable(tf.constant(1.0, shape=[input_node.shape[3],]), dtype=tf.float32, name="th_width_scale")
                beta = tf.Variable(tf.constant(0.0, shape=[input_node.shape[3],]), dtype=tf.float32, name="th_shift_percent")

                alpha_constrained = tf.clip_by_value(alpha, self.r_alpha[0], self.r_alpha[1], name="th_width_scale_constrain")
                beta_constrained = tf.clip_by_value(beta, self.r_beta[0], self.r_beta[1], name="th_shift_percent_constrain")

                adjusted_th_width = tf.multiply(th_width_node, alpha_constrained, name="adjusted_width")
                shift_node = tf.multiply(th_width_node, beta_constrained, name="min_th_shift")

                adjusted_min_th = tf.add(min_th_node, shift_node, name="adjusted_min_th")

                adjusted_max_th = tf.add(adjusted_min_th, adjusted_th_width, name="adjusted_max_th")

            # scale
            with tf.name_scope("scale"):
                q_range = 2. ** bits - 1.

                eps = tf.constant(10. ** -7, shape=[input_node.shape[3],], dtype=tf.float32, name='eps')
                scale_node = tf.div(q_range, adjusted_th_width + eps, "new_input_scale")
                scale_node = tf.reshape(scale_node,[1,1,1,-1])

            with tf.name_scope("quantized_bias"):
                quant_bias = tf.multiply(adjusted_min_th, scale_node, name="scaling")
                quant_bias = _rounding_fn(quant_bias, name="rounding")

            with tf.name_scope("discrete_input_data"):
                net = tf.multiply(input_node, scale_node, name="scaling")
                net = _rounding_fn(net, name='rounding')
                net = tf.clip_by_value(net - quant_bias, clip_value_min=0., clip_value_max=tf.reshape(2.**bits-1., [1,1,1,-1])) + quant_bias

                descrete_input_data = tf.div(net, scale_node, name="discrete_data")

        self._add_thresholds(out_name, adjusted_min_th, adjusted_max_th)
        self._add_th_var(alpha)
        self._add_th_var(beta)

        return descrete_input_data, bits


    def _adjustable_quant_bias(self, input_node, out_name='bias', bits=8, name="adjust_bias"):
        self.num_sf += input_node.shape[0].value

        max_th = tf.reduce_max(input_node)
        min_th = tf.reduce_min(input_node)
        th_width = max_th - min_th

        with tf.name_scope(name):
            with tf.name_scope("thresholds"):
                min_th_node = tf.identity(min_th, name="min_th")
                th_width_node = tf.identity(th_width, name="th_width")

                alpha = tf.Variable(tf.constant(1.0), dtype=tf.float32, name="th_width_scale")
                beta = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="th_shift_percent")

                alpha_constrained = tf.clip_by_value(alpha, self.r_alpha[0], self.r_alpha[1], name="th_width_scale_constrain")
                beta_constrained = tf.clip_by_value(beta, self.r_beta[0], self.r_beta[1], name="th_shift_percent_constrain")

                adjusted_th_width = tf.multiply(th_width_node, alpha_constrained, name="adjusted_width")
                shift_node = tf.multiply(th_width_node, beta_constrained, name="min_th_shift")

                adjusted_min_th = tf.add(min_th_node, shift_node, name="adjusted_min_th")

                adjusted_max_th = tf.add(adjusted_min_th, adjusted_th_width, name="adjusted_max_th")

            # scale
            with tf.name_scope("scale"):
                q_range = 2. ** bits - 1.

                eps = tf.constant(10. ** -7, dtype=tf.float32, name='eps')
                scale_node = tf.div(q_range, adjusted_th_width + eps, "new_input_scale")

            with tf.name_scope("quantized_bias"):
                quant_bias = tf.multiply(adjusted_min_th, scale_node, name="scaling")
                quant_bias = _rounding_fn(quant_bias, name="rounding")

            with tf.name_scope("discrete_input_data"):
                net = tf.multiply(input_node, scale_node, name="scaling")
                net = _rounding_fn(net, name='rounding')
                net = tf.clip_by_value(net - quant_bias, clip_value_min=0., clip_value_max=2.**bits-1.) + quant_bias

                descrete_input_data = tf.div(net, scale_node, name="discrete_data")

        self._add_thresholds(out_name, adjusted_min_th, adjusted_max_th)
        self._add_th_var(alpha)
        self._add_th_var(beta)

        return descrete_input_data
    

    def _create_weights_node(self, weights_data, quant=True, dws=False):
        weights_name_scope = _get_name_scope() + "/weights"

        w_min, w_max = self._get_thresholds(weights_name_scope)

        if self.weight_const:
            weights_node = tf.constant(weights_data, tf.float32, name="weights")
        else:
            weights_node = tf.Variable(weights_data, tf.float32, name="weights")
            
        self._add_reference_node(weights_node)

        if quant:
            if not dws:
                quantized_weights, bits = self._adjustable_quant_signed(weights_node,
                                                                  w_min,
                                                                  w_max,
                                                                  weights_name_scope,
                                                                  bits=self.weight_bits,
                                                                  name="quantized_weights")
            else:
                quantized_weights, bits = self._adjustable_dws_quant_signed(weights_node,
                                                                  w_min,
                                                                  w_max,
                                                                  weights_name_scope,
                                                                  bits=self.weight_bits,
                                                                  name="quantized_weights")

            key = weights_name_scope.split('/')[0]

            if 'expanded_conv' in key:   
                sub_key = weights_name_scope.split('/')[1]
                self.quant_info[key][sub_key]['weight'] = bits
            else:
                self.quant_info[key]['weight'] = bits

            return quantized_weights
        
        else:
            return weights_node


    def _cell_output(self, net, output_type=None, subscope='output'):

        if output_type == "fixed":
            pass
            #net = _fixed_quant_signed(net, -1, 1, bits=self.bits, name="fixed_quantized_input")
        else:
            output_name_scope = _get_name_scope() + "/" + subscope
            i_min, i_max = self._get_thresholds(output_name_scope)

            net, bits = self._adjustable_quant_signed(net, i_min, i_max, output_name_scope, bits=self.act_bits, name="quantized_input")

            key = output_name_scope.split('/')[0]

            if 'expanded_conv' in key:
                sub_key = output_name_scope.split('/')[1]
                if sub_key == 'output':
                    self.quant_info[key]['se_2']['act'] = bits
                elif sub_key == 'add':
                    self.quant_info[key]['project']['act'] = bits
                else:
                    self.quant_info[key][sub_key]['act'] = bits
            
            else:
                self.quant_info[key]['act'] = bits

        net = tf.identity(net, name="output")

        self._add_reference_node(net)

        return net
        

    def _swish_input(self, net):
        if self.swish_bits == -1:
            return net

        output_name_scope = _get_name_scope() + "/swish_input"
        i_min, i_max = self._get_thresholds(output_name_scope)

        if self.swish_bits:
            net, bits = self._adjustable_quant_signed_static(net, i_min, i_max, output_name_scope, bits=self.swish_bits, name="swish_input")
        else:
            net, bits = self._adjustable_quant_signed(net, i_min, i_max, output_name_scope, bits=self.act_bits, name="swish_input")

        self.swish_bits_list.append(bits)
        net = tf.identity(net, name="swish_input")

        self._add_reference_node(net)

        return net


    def _build_dws(self, input_node, weights_and_bias, strides, quant=True):
        weights_node = self._create_weights_node(weights_and_bias["weights"],quant,dws=True)
        
        op_output = tf.nn.depthwise_conv2d(input_node, 
                                           weights_node, 
                                           (1, strides, strides, 1), 
                                           padding="SAME")

        if "bias" in weights_and_bias:
            if self.weight_const:
                bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            else:
                bias_node = tf.Variable(weights_and_bias["bias"], tf.float32, name="bias")     

            if self.bias_bits != 0:
                output_name_scope = _get_name_scope()
                bias_node = self._adjustable_quant_bias(bias_node, output_name_scope, self.bias_bits)

            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output
    

    def _build_conv(self, input_node, weights_and_bias, strides, quant=True):
        weights_node = self._create_weights_node(weights_and_bias["weights"],quant)
        
        op_output = tf.nn.conv2d(input_node, 
                                 weights_node, 
                                 (1, strides, strides, 1), 
                                 padding="SAME")
        
        if "bias" in weights_and_bias:
            if self.weight_const:
                bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            else:
                bias_node = tf.Variable(weights_and_bias["bias"], tf.float32, name="bias")
            
            if self.bias_bits != 0:
                output_name_scope = _get_name_scope()
                bias_node = self._adjustable_quant_bias(bias_node, output_name_scope, self.bias_bits)

            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output

