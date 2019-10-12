import os
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import re

class Trainer:

    def __init__(self,
                 train_graph,
                 train_labels,
                 val_labels,
                 original_output,
                 quant_model,
                 is_training=None,
                 **kwargs):

        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_weight = kwargs['learning_rate_weight']
        self.learning_rate_decay = kwargs["learning_rate_decay"]
        self.reinit_adam_after_n_batches = kwargs["reinit_adam_after_n_batches"]
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.log_dir = kwargs["log_dir"]
        self.lr_fixed = kwargs["lr_fixed"]
        self.swa_freq = kwargs["swa_freq"]
        self.swa_delay = kwargs["swa_delay"]
        self.warm_up_epoch = kwargs["warm_up_epoch"]
        self.bits_trainable = kwargs["bits_trainable"]
        self.lr_bits = kwargs["lr_bits"]
        self.ckpt_path = kwargs["ckpt_path"]
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.gamma = kwargs["gamma"]
        self.metric = kwargs["metric"]
        self.iter_train = kwargs["iter_train"]
        self.iter_train_freq = kwargs["iter_train_freq"]
        self.finetune = kwargs["finetune"]

        self.quant_info = quant_model.quant_info
        self.num_sf = quant_model.num_sf
        self.num_swish = quant_model.num_swish
        self.num_swish_list = quant_model.num_swish_list
        self.swish_bits = quant_model.swish_bits
        self.swish_bits_list = quant_model.swish_bits_list
        self.bias_bits = quant_model.bias_bits

        self.total_train_img = 1281167
        self.total_val_img = 50000

        self.is_training = is_training

        self.train_labels = train_labels
        self.val_labels = val_labels
        self.quantized_output = quant_model.output_node

        self._quant_argmax = tf.argmax(self.quantized_output, axis=-1)
        self._original_argmax = tf.argmax(original_output, axis=-1)
        self._quant_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self._quant_argmax,tf.cast(self.val_labels,tf.int64))))
        self._original_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self._original_argmax,tf.cast(self.val_labels,tf.int64))))

        with train_graph.as_default():
            self._build_train_procedure(original_output, self.quantized_output, self.learning_rate)


    def _compute_metric(self):
        model_info = json.load(open('model_info.json','r'))

        params = 0
        flops = 0

        for key in model_info.keys():
            if key == 'total_bias':
                continue
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

        if self.bias_bits:
            params_bias = model_info['total_bias']*self.bias_bits/32
        else:
            params_bias = model_info['total_bias']

        self.params = params + params_bias

        if self.swish_bits == -1:
            flops_swish = 3.*self.num_swish
        elif self.swish_bits:
            flops_swish = 3.*self.num_swish*self.swish_bits/32
        else:
            flops_swish = 0.
            for i in range(len(self.num_swish_list)):
                flops_swish += 3*self.num_swish_list[i]*tf.reduce_mean(self.swish_bits_list[i])/32

        self.flops = flops*2 + flops_swish

        metric = self.params / 6.9e6 + self.flops / 1170e6

        return metric


    def soft_reduce_max_op(self,a,b):
        if isinstance(b,int):
            return tf.to_float(b)
            
        new_a = tf.tile(a, b.shape)
        new_b = tf.tile(b, a.shape)
        bp = (1-self.gamma)*new_a + self.gamma*new_b
        if self.bits_trainable:
            return tf.reduce_mean(bp + tf.stop_gradient(tf.reduce_max([new_a, new_b],axis=0) - bp))
        else:
            return tf.reduce_mean(bp + tf.stop_gradient(tf.cast(tf.reduce_max([new_a, new_b],axis=0),tf.float64) - bp))


    def _build_train_procedure(self, original_output: tf.Tensor, quantized_output: tf.Tensor, initial_lr):
        self.loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=quantized_output,labels=self.train_labels))
        self.loss_kd= 0.01*tf.reduce_mean(tf.reduce_sum(tf.sqrt((quantized_output - original_output) ** 2 + 1e-5), axis=-1))
        self.loss_metric = self._compute_metric()
        sign = tf.cond(self.loss_metric>self.metric, lambda:1.0, lambda: -1.0)
        self.loss = self.loss_kd + self.alpha*self.loss_cls + sign*self.beta*tf.cast(self.loss_metric, tf.float32)

        self.lr = tf.placeholder_with_default(initial_lr, shape=[], name="learning_rate")

        if self.bits_trainable:
            var_bits = []
            var_other = []
            for var in tf.trainable_variables():
                if 'bits_scale' in var.name:
                    var_bits.append(var)
                else:
                    var_other.append(var)
            opt = tf.train.AdamOptimizer(self.lr)
            self.train_op_1 = opt.minimize(self.loss,var_list=var_other)
            opt2 = tf.train.AdamOptimizer(self.lr_bits)
            self.train_op_2 = opt2.minimize(self.loss,var_list=var_bits)

            self.train_op = tf.group(self.train_op_1, self.train_op_2)

        else:
            opt = tf.train.AdamOptimizer(self.lr)
            self.train_op = opt.minimize(self.loss)


    def train(self, sess):

        thresholds = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        optimizer_vars = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(thresholds))

        saver = tf.train.Saver(max_to_keep=0) 
        loss_min_train = np.Infinity
        loss_min_val = np.Infinity
        acc_max_val = -np.Infinity
        filewriter = tf.summary.FileWriter(self.log_dir, sess.graph)

        sess.run([var.initializer for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        global_step_train = 0
        decay_step = 0

        if self.ckpt_path:
            # variables_to_restore = []
            # for var in tf.trainable_variables():
            #     if 'bias' not in var.name:
            #         variables_to_restore.append(var)
            # saver2 = tf.train.Saver(variables_to_restore)
            # saver2.restore(sess, self.ckpt_path)
            saver.restore(sess, self.ckpt_path)
            print('Restore model from ckpt file:',self.ckpt_path)
            self.ckpt_path = None

        if self.swa_delay is not None:
            var_names = {}

            for var in tf.trainable_variables():
                var_names[var.name] = var.name

            weight_all_epoch = []

        exit_flag = False

        for cur_epoch in range(int(np.ceil(self.epochs))):
            print('')
            print('=' * 40)
            print("Epoch: {}".format(cur_epoch + 1))
            print('=' * 40)

            for k in range(10):
                print("******** {} / 10 of the Epoch ********".format(k+1))

                train_op = None

                if not self.bits_trainable:
                    train_op = self.train_op
                elif self.finetune:
                    train_op = self.train_op_1
                else:
                    if self.warm_up_epoch is not None:
                        if cur_epoch + k*0.1 < self.warm_up_epoch:
                            train_op = self.train_op_1

                    if not train_op:
                        if self.iter_train and k % self.iter_train_freq:
                            train_op = self.train_op_1
                        else:
                            train_op = self.train_op

                global_step_train, decay_step, epoch_mean_loss = self._train_one_tenth_epoch(sess, train_op,
                                                                                   global_step_train, decay_step,
                                                                                   thresholds, saver, filewriter,
                                                                                   optimizer_vars)
                if epoch_mean_loss < loss_min_train:
                    loss_min_train = epoch_mean_loss

                print("Train loss value: {}, min is {}".format(epoch_mean_loss, loss_min_train))
                print("Check current accuracy ...")
                top_1_acc, metric, params, flops = self.validate(sess)

                saver.save(sess, os.path.join('weights', "ckpt_metric_{}".format(metric)))

                print("Top 1 acc: {}, Metric: {}, Params: {}, Flops: {}".format(top_1_acc, metric, params, flops))

                if self.swa_delay is not None:
                    if self.swa_delay <= cur_epoch + k*0.1 and k and not k % int(10*self.swa_freq):
                        print("Saving current SWA weights...")
                        weight_current = sess.run(var_names)
                        weight_all_epoch.append(weight_current)

                if cur_epoch + (k+1)/10 >= self.epochs:
                    exit_flag = True
                    break
            
            if exit_flag:
                break

        if self.swa_delay is not None:
            print("Compute and save final SWA model...")
            weight_avg = {}
            for var_name in var_names:
                value_sum = 0
                for weight_dict in weight_all_epoch:
                    value_sum += weight_dict[var_name]
                weight_avg[var_name] = value_sum / len(weight_all_epoch)

            for var in tf.trainable_variables():
                sess.run(var.assign(weight_avg[var.name]))

            saver.save(sess,os.path.join('weights','model_swa.ckpt'))

            top_1_acc, metric, params, flops = self.validate(sess)
            print("Evaluation of SWA model:")
            print("Top 1 acc: {}, Metric: {}, Params: {}, Flops: {}".format(top_1_acc, metric, params, flops))


    def _train_one_tenth_epoch(
            self,
            sess,
            train_op,
            global_step_train,
            decay_step,
            thresholds,
            saver,
            filewriter: tf.summary.FileWriter,
            optimizer_vars):

        loss_arr_train = []

        for i in tqdm(range(int(self.total_train_img/self.batch_size/10))):
            if self.lr_fixed:
                learning_rate_val = self.learning_rate
            else:
                learning_rate_val = self.learning_rate * \
                          np.exp(-global_step_train * self.learning_rate_decay) * \
                          np.abs(np.cos(np.pi * decay_step / 4 / self.reinit_adam_after_n_batches)) + 10.0 ** -7

            feed_dict = {self.lr: learning_rate_val,
                     self.is_training: True}

            _, loss_value, loss_kd, loss_metric, params, flops = sess.run([train_op, self.loss, self.loss_kd, self.loss_metric, self.params, self.flops], feed_dict)
            loss_arr_train.append(loss_value)

            print('total loss:', loss_value, 'loss kd:', loss_kd, 'loss metric:', loss_metric, 'param:', params, 'flops:', flops)

            global_step_train += 1
            decay_step += 1

            if not self.lr_fixed and global_step_train % self.reinit_adam_after_n_batches == 0:
                init_opt_vars_op = tf.variables_initializer(optimizer_vars)
                sess.run(init_opt_vars_op)
                decay_step = 0

        return global_step_train, decay_step, float(np.mean(loss_arr_train))


    def validate(self, sess, use_quantized=True):
        loss_arr_validation = []
        top_1_acc_arr = []

        if self.ckpt_path:
            saver = tf.train.Saver()
            saver.restore(sess,self.ckpt_path)

        for i in tqdm(range(int(self.total_val_img/self.batch_size))):
            loss_value, accuracy = self._valid_step(sess, use_quantized)
            loss_arr_validation.append(loss_value)
            top_1_acc_arr.append(accuracy)

        if use_quantized:
            metric, params, flops = sess.run([self.loss_metric, self.params, self.flops])
            return float(np.mean(top_1_acc_arr)), metric, params, flops
        else:
            return float(np.mean(top_1_acc_arr))


    def _valid_step(self, sess, use_quantized: bool):
        if use_quantized:
            res = sess.run([self.loss, self._quant_accuracy], {self.is_training: False})
        else:
            res = sess.run([self.loss, self._original_accuracy], {self.is_training: False})

        return res
