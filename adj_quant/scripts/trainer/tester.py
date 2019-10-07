import os
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import re

class Tester:

    def __init__(self,
                 train_graph,
                 labels,
                 original_output,
                 quant_model,
                 **kwargs):

        self.batch_size = kwargs["batch_size"]
        self.ckpt_path = kwargs["ckpt_path"]
        self.bits_trainable = kwargs["bits_trainable"]

        self.quant_info = quant_model.quant_info
        self.num_sf = quant_model.num_sf
        self.num_swish = quant_model.num_swish
        self.num_swish_list = quant_model.num_swish_list
        self.swish_bits = quant_model.swish_bits
        self.swish_bits_list = quant_model.swish_bits_list

        self.total_train_img = 1281167
        self.total_val_img = 50000

        self.labels = labels
        self.quantized_output = quant_model.output_node

        self._quant_argmax = tf.argmax(self.quantized_output, axis=-1)
        self._original_argmax = tf.argmax(original_output, axis=-1)
        self._quant_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self._quant_argmax,tf.cast(self.labels,tf.int64))))
        self._original_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self._original_argmax,tf.cast(self.labels,tf.int64))))

        self.loss_metric = self._compute_metric()


    def _compute_metric(self):
        '''
        model_info.json contains the params and flops of each layer in EfficientNet-B0
        quant_info is a dict containing the corresponding precision of each layer 
        '''

        model_info = json.load(open('model_info.json','r'))

        params = 0
        flops = 0

        for key in model_info.keys():
            if key == 'expanded_conv':
                new_key = 'expanded_conv_0'
            elif key == 'Logits':
                new_key = 'output'
            else:
                new_key = key

            if 'Conv' in new_key or 'output' in new_key:
                params += model_info[key][0]['total'] * tf.reduce_mean(self.quant_info[new_key]['weight']/32)

            else:
                params += (model_info[key][0]['expand']*tf.reduce_mean(self.quant_info[new_key]['expand']['weight'])
                          + model_info[key][0]['dws']*tf.reduce_mean(self.quant_info[new_key]['dws']['weight'])
                          + model_info[key][0]['se_1']*tf.reduce_mean(self.quant_info[new_key]['se_1']['weight']) 
                          + model_info[key][0]['se_2']*tf.reduce_mean(self.quant_info[new_key]['se_2']['weight'])
                          + model_info[key][0]['project']*tf.reduce_mean(self.quant_info[new_key]['project']['weight'])) / 32

            if new_key == 'Conv':
                flops += model_info[key][1]['total'] * 8 / 32
            elif new_key == 'Conv_1':
                flops += model_info[key][1]['total'] * self.soft_reduce_max_op(self.quant_info[new_key]['weight'],self.quant_info['expanded_conv_15']['project']['act']) / 32
            elif new_key == 'output':
                flops += model_info[key][1]['total'] * self.soft_reduce_max_op(self.quant_info[new_key]['weight'],self.quant_info['Conv_1']['act']) / 32
            else:
                index = int(re.findall(r'\d+',new_key)[0])
                if index:
                    pre_index = index - 1
                    pre_key = new_key.replace(str(index),str(pre_index))
                    flops_expand = model_info[key][1]['expand'] * self.soft_reduce_max_op(self.quant_info[new_key]['expand']['weight'],self.quant_info[pre_key]['project']['act'])
                    flops_dws = model_info[key][1]['dws'] * self.soft_reduce_max_op(self.quant_info[new_key]['dws']['weight'],self.quant_info[new_key]['expand']['act'])
                else:
                    flops_expand = 0
                    flops_dws = model_info[key][1]['dws'] * self.soft_reduce_max_op(self.quant_info[new_key]['dws']['weight'],self.quant_info['Conv']['act'])

                flops_se_1 = model_info[key][1]['se_1'] * self.soft_reduce_max_op(self.quant_info[new_key]['se_1']['weight'],self.quant_info[new_key]['dws']['act'])
                flops_se_2 = model_info[key][1]['se_2'] * self.soft_reduce_max_op(self.quant_info[new_key]['se_2']['weight'],self.quant_info[new_key]['se_1']['act'])
                flops_project = model_info[key][1]['project'] * self.soft_reduce_max_op(self.quant_info[new_key]['project']['weight'],self.quant_info[new_key]['se_2']['act'])
                
                flops += (flops_expand + flops_dws + flops_se_1 + flops_se_2 + flops_project) / 32

        self.params = params

        if self.swish_bits == -1:
            flops_swish = 0.
        elif self.swish_bits:
            flops_swish = 3*self.num_swish*self.swish_bits/32
        else:
            flops_swish = 0.
            for i in range(len(self.num_swish_list)):
                flops_swish += 3*self.num_swish_list[i]*tf.reduce_mean(self.swish_bits_list[i])/32

        self.flops = flops*2 + flops_swish

        metric = self.params / 6.9e6 + self.flops / 1170e6

        return metric


    def soft_reduce_max_op(self,a,b):
        '''
        Since the max operatin is not differentiable, we rewrite its backward function here for reduing Flops
        '''

        if isinstance(b,int):
            return tf.to_float(b)
            
        new_a = tf.tile(a, b.shape)
        new_b = tf.tile(b, a.shape)
        if self.bits_trainable:
            return tf.reduce_mean(tf.reduce_max([new_a, new_b],axis=0))
        else:
            return tf.reduce_mean(tf.cast(tf.reduce_max([new_a, new_b],axis=0),tf.float64))


    def validate(self, sess, use_quantized=True):
        top_1_acc_arr = []

        # Specify the checkpoint file to be evaluated
        if self.ckpt_path:
            saver = tf.train.Saver()
            saver.restore(sess,self.ckpt_path)

        for i in tqdm(range(int(self.total_val_img/self.batch_size))):
            accuracy = self._valid_step(sess, use_quantized)
            top_1_acc_arr.append(accuracy)

        if use_quantized:
            metric, params, flops = sess.run([self.loss_metric, self.params, self.flops])
            return float(np.mean(top_1_acc_arr)), metric, params, flops
        else:
            return float(np.mean(top_1_acc_arr))


    def _valid_step(self, sess, use_quantized: bool):
        if use_quantized:
            res = sess.run([self._quant_accuracy])
        else:
            res = sess.run([self._original_accuracy])

        return res
