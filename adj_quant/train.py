import warnings
warnings.filterwarnings('ignore')

import os
import sys
import tensorflow as tf
import scripts.effnetb0.combined_tools as ct
from scripts import helpers as hp
from scripts.thresholds import eval_thresholds
from scripts.trainer.trainer import Trainer
from data import gen_data
import argparse
import json
import numpy as np

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--local',
                    help='train on local device or not',
                    action='store_true',
                    default=False)
parser.add_argument('-e',
                    '--eval_initial_thresholds',
                    help='eval initial thresholds or not',
                    action='store_true',
                    default=False)
parser.add_argument('--validation',
                    help='validation mode or not',
                    action='store_true',
                    default=False)
parser.add_argument('--export_training_graph',
                    help='export training graph',
                    action='store_true',
                    default=False)
parser.add_argument('--export_inference_graph',
                    help='export inference graph',
                    action='store_true',
                    default=False)
parser.add_argument('--mix_prec',
                    help='quant with mix precision',
                    action='store_true',
                    default=False)
parser.add_argument('--tfs',
                    help='train from scratch or not',
                    action='store_true',
                    default=False)
parser.add_argument('--allow_growth',
                    help='gpu allow growth or not',
                    action='store_true',
                    default=False)
parser.add_argument('-c','--weight_const',
                    help='gpu allow growth or not',
                    action='store_true',
                    default=False)
parser.add_argument('--ckpt_path',
                    help='ckpt path for go on training/validation',
                    default='')
parser.add_argument('--ckpt_path_thres',
                    help='ckpt path for eval thresholds',
                    default='')
parser.add_argument('-d',
                    '--dataset',
                    default='/home/cl114/common_datasets/imagenet_tfrecord',
                    help='eval initial thresholds or not',
)
parser.add_argument('--batch_size',
                    type=int,
                    default=25,
                    help='batch size',
)
parser.add_argument('-a','--alpha',
                    type=float,
                    default=7,
                    help='weight of cls loss',
)
parser.add_argument('-b','--beta',
                    type=float,
                    default=85,
                    help='weight of metric loss',
)
parser.add_argument('-g','--gamma',
                    type=float,
                    default=0.7,
                    help='relative weight of activations in FLOPS loss backpropagation',
)
parser.add_argument('-r','--range',
                    type=str,
                    default='5',
                    help='choose clip value for quant thresholds',
)
parser.add_argument('--weight_path',
                    type=str,
                    default='../pretrained/efficient_b0_autoaugment_quant/frozen_efficientnet_b0_quant_weights.pickle',
                    help='float weight path',
)
parser.add_argument('--teacher_path',
                    type=str,
                    default='../pretrained/efficient_b0_autoaugment/frozen_efficientnet_b0_weights.pickle',
                    help='weight path of teacher network',
)
parser.add_argument('--weight_bits',
                    type=int,
                    default=8,
                    help='num bits to represent weights',
)
parser.add_argument('--act_bits',
                    type=int,
                    default=8,
                    help='num bits to represent activations',
)
parser.add_argument('--swish_bits',
                    type=int,
                    default=8,
                    help='num bits of the input to swish (0 means trainable, -1 means no quantization)',
)
parser.add_argument('--swa_delay',
                    help='num epoches for swa delay',
                    type=float,
                    default=None)
parser.add_argument('--swa_freq',
                    help='frequency of saving swa weights',
                    type=float,
                    default=0.2)
parser.add_argument('--lr',
                    help='learning rate',
                    type=float,
                    default=3e-6)
parser.add_argument('--lr_decay',
                    help='learning rate decay',
                    type=float,
                    default=1e-4)
parser.add_argument('--reinit',
                    help='reinit adam after n batches',
                    type=int,
                    default=800)
parser.add_argument('--lr_bits',
                    help='learning rate for bits scale',
                    type=float,
                    default=1e-3)
parser.add_argument('--lr_weight',
                    help='learning rate for weight',
                    type=float,
                    default=3e-6)
parser.add_argument('--max_epoch',
                    type=float,
                    default=3,
                    help='total epoch num for training',
)
parser.add_argument('--lr_fixed',
                    help='use fixed learning rate or not',
                    action='store_true',
                    default=False)
parser.add_argument('--img_size',
                    type=int,
                    default=224,
                    help='train image size',
)
parser.add_argument('-t','--tea_model',
                    help='teacher model for knowledge distillation',
                    default='eff_b0')
parser.add_argument('--warm_up_epoch',
                    help='delay from train thresholds and weights only to jointly train bits scale factor',
                    type=float,
                    default=None)
parser.add_argument('--bits_trainable',
                    help='use trainable precision per channel or not',
                    action='store_true',
                    default=False)
parser.add_argument('--metric',
                    type=float,
                    default=0.17,
                    help='target metric',
)
parser.add_argument('--aug',
                    help='use data augmentation during training',
                    action='store_true',
                    default=False)
parser.add_argument('--iter_train',
                    help='iteratively train adjustable precision and adjustable thresholds/weights',
                    action='store_true',
                    default=False)
parser.add_argument('--iter_train_freq',
                    help='the frequency of train thresholds/weights/precision jointly',
                    default=2,
                    type=int)
parser.add_argument('--finetune',
                    help='fix precision and finetune',
                    action='store_true',
                    default=False)
args = parser.parse_args()

weight_path = args.weight_path  # Initial weights after Quantization Aware Training 
teacher_path = args.teacher_path  # Float32 weights as the teacher model in Knowledge Distillation

WEIGHTS = hp.load_pickle(weight_path)
TEA_WEIGHTS = hp.load_pickle(teacher_path)

if args.range == '2':
    r_alpha=[0.5,1.0]
    r_beta=[-0.2,0.4]
elif args.range == '3':
    r_alpha=[0.2,1.7]
    r_beta=[-0.4,0.7]
elif args.range == '4':
    r_alpha=[0.1,2]
    r_beta=[-0.5,1]
elif args.range == '5':
    r_alpha=[0,2]
    r_beta=[-0.6,1.2]
elif args.range == '6':
    r_alpha=[0,2.2]
    r_beta=[-0.8,1.5]
elif args.range == '7':
    r_alpha=[-100,100]
    r_beta=[-100,100]
else:
    r_alpha=[0.5,1.3]
    r_beta=[-0.2,0.4]


class MyEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# Eval initial quantization range, if no specification, the initial_thresholds.json will serve as the initial range
if args.ckpt_path_thres:
    with open('initial_thresholds.json', 'r') as f:
        THRESHOLDS = json.load(f)

    print("Eval Initial Thresholds from ", args.ckpt_path_thres)

    thr_input_node,_ = gen_data('train',args.dataset,args.batch_size) 
    _, quant_model = ct.create_adjustable_model(thr_input_node, WEIGHTS, TEA_WEIGHTS, THRESHOLDS, r_alpha, r_beta, tea_model=args.tea_model, weight_bits=args.weight_bits, act_bits=args.act_bits, swish_bits=args.swish_bits, weight_const=args.weight_const, bits_trainable=args.bits_trainable, mix_prec=args.mix_prec)

    with tf.Session(graph=quant_model.graph) as sess:
        saver_e = tf.train.Saver()
        saver_e.restore(sess, args.ckpt_path_thres)

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        THRESHOLDS = eval_thresholds(sess, quant_model.input_node,
                                     quant_model.reference_nodes, args.batch_size)

        coord.request_stop()
        coord.join(threads)

    with open('initial_thresholds.json', 'w') as f:
        json.dump(THRESHOLDS, f, cls=MyEncoder)

    tf.reset_default_graph()

elif args.eval_initial_thresholds:
    print("Eval Initial Thresholds...")

    thr_input_node,_ = gen_data('train',args.dataset,args.batch_size) 
    float_model = ct.create_float_model(thr_input_node,
                                        WEIGHTS, args.mix_prec)

    with tf.Session(graph=float_model.graph) as sess:
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        THRESHOLDS = eval_thresholds(sess, float_model.input_node,
                                     float_model.reference_nodes, args.batch_size)

        coord.request_stop()
        coord.join(threads)

    tf.reset_default_graph()

    with open('initial_thresholds.json', 'w') as f:
        json.dump(THRESHOLDS, f, cls=MyEncoder)

else:
    print("Load Initial Thresholds...")

    with open('initial_thresholds.json', 'r') as f:
        THRESHOLDS = json.load(f)

# Prepare the training/validation data
print("Prepare Data...")
is_training = tf.placeholder(tf.bool)
val_images,val_labels = gen_data('validation', args.dataset, args.batch_size, args.img_size)
train_images,train_labels = gen_data('train', args.dataset, args.batch_size, args.img_size, args.aug)

# Swith between training data and validation data during training to monitor the accuracy on validation set
input_node = tf.cond(is_training,lambda:train_images,lambda:val_images)

# Create Model with Adjustable Quantization nodes
print("Create Adjustable Model...")
float_model, quant_model = ct.create_adjustable_model(input_node, WEIGHTS, TEA_WEIGHTS, THRESHOLDS, r_alpha, r_beta, tea_model=args.tea_model, weight_bits=args.weight_bits, act_bits=args.act_bits, swish_bits=args.swish_bits, weight_const=args.weight_const, bits_trainable=args.bits_trainable, mix_prec=args.mix_prec)


if args.export_inference_graph:
	print('Exporting inference graph.')
	hp.save_pb(input_node.graph,'./inference_graph.pb')
	sys.exit(0)


train_configuration_data = {}

train_configuration_data['log_dir'] = './logs'
train_configuration_data['batch_size'] = args.batch_size
train_configuration_data['learning_rate'] = args.lr
train_configuration_data['learning_rate_decay'] = args.lr_decay
train_configuration_data['reinit_adam_after_n_batches'] = args.reinit
train_configuration_data['learning_rate_weight'] = args.lr_weight
train_configuration_data['lr_fixed'] = args.lr_fixed
train_configuration_data['lr_bits'] = args.lr_bits
train_configuration_data['epochs'] = args.max_epoch
train_configuration_data['swa_freq'] = args.swa_freq
train_configuration_data['bits_trainable'] = args.bits_trainable
train_configuration_data['alpha'] = args.alpha
train_configuration_data['beta'] = args.beta
train_configuration_data['gamma'] = args.gamma
train_configuration_data['swa_delay'] = args.swa_delay
train_configuration_data['warm_up_epoch'] = args.warm_up_epoch
train_configuration_data['ckpt_path'] = args.ckpt_path
train_configuration_data['metric'] = args.metric
train_configuration_data['iter_train'] = args.iter_train
train_configuration_data['iter_train_freq'] = args.iter_train_freq
train_configuration_data['finetune'] = args.finetune


# Create Trainer with iterative training strategy
print("Build Trainer...")
my_trainer = Trainer(input_node.graph, train_labels, val_labels, float_model.output_node,
                     quant_model,is_training=is_training, **train_configuration_data)

if args.export_training_graph:
	print('Exporting training graph.')
	hp.save_pb(input_node.graph,'./training_graph.pb')
	sys.exit(0)

print('############# Start Training Procedure #############')

config=tf.ConfigProto()
config.gpu_options.allow_growth = args.allow_growth

with tf.Session(graph=input_node.graph,config=config) as sess:
    with sess.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(quant_model.initializer)

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        if not args.validation:
            
            if args.tfs:
                # Check accuracy of the original model
                print("Check accuracy of the original model ...")
                original_top1 = my_trainer.validate(sess, False)
                print("Original top 1:", original_top1)

                # Check accuracy of the non-trained quantized model
                print("Check accuracy of the non-trained quantized model ...")
                initial_top1, initial_metric, initial_params, initial_flops = my_trainer.validate(sess)
                print("Initial top 1:", initial_top1, 'Initial metric:', initial_metric, 'Initial params:', initial_params, 'Initial flops:'. initial_flops)

            # Start training process
            print("Training thresholds of the quantized model ...")
            my_trainer.train(sess)
        
        else:
            print("Check accuracy of the quantized model ...")
            top1, metric, params, flops = my_trainer.validate(sess)
            print("top 1:", top1, 'metric:', metric, 'params:', params, 'flops:', flops)

        coord.request_stop()
        coord.join(threads)
