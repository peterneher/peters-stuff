import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
from Layers.DiceLoss import DiceLossLayer
from Layers.DiceIndex import DiceIndexLayer

class CaffeUNet_2D :
          
    
    def __init__(self, batch_size, num_classes, input_size, data_channels, label_channels, base_n_filters, num_blocks, loss_func='xent', ignore_label=None) :

        self.data_channels = data_channels
        self.label_channels = label_channels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.base_n_filters = base_n_filters
        self.num_blocks = num_blocks
        self.loss_func = loss_func

        self.do_dropout = True
        self.kernel_size = 3
        self.weight_filler = dict(type='msra')
        self.bias_filler = dict(type='constant')
        self.net_spec = None
        self.param = [dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)]
        self.ignore_label = ignore_label
        self.use_batchnorm = False

        self.create_network_structure()
        
    def save_net_spec(self, output_file):
        
        f = open(output_file, 'w')
        f.write(str(self.net_spec.to_proto()))
        f.close()

    def print_all_available_layers(self):

        layer_ist = caffe.layer_type_list()
        for el in layer_ist:
            print el

    def print_network_sizes(self, model_file):

        net = caffe.Net(model_file, caffe.TRAIN)
        for k, v in net.blobs.items():
            print k, v.data.shape

    def add_contraction_block(self, input, block_number):

        if self.use_batchnorm :
            l = self.add_batchnormscale(name='contr_' + str(block_number) + '_1', input=L.ReLU(L.Convolution(input, pad=self.pad, kernel_size=self.kernel_size, num_output=self.base_n_filters * pow(2, block_number-1), weight_filler=self.weight_filler, bias_filler=self.bias_filler, param=self.param), in_place=True))
            l = self.add_batchnormscale(name='contr_' + str(block_number) + '_2', input=L.ReLU(L.Convolution( l, pad=self.pad, kernel_size=self.kernel_size, num_output=self.base_n_filters * pow(2, block_number-1), weight_filler=self.weight_filler, bias_filler=self.bias_filler, param=self.param), in_place=True))
        else:
            l = self.add_conv(input=input, name='contr_' + str(block_number) + '_1', filter_mult=block_number-1)
            l = self.add_conv(input=l, name='contr_' + str(block_number) + '_2', filter_mult=block_number-1)

        l = L.Pooling( l, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        setattr(self.net_spec, 'pool_' + str(block_number), l)

        return l

    def add_expansion_block(self, input, block_number):

        l = L.Deconvolution(input, convolution_param=dict(num_output=self.base_n_filters * pow(2, block_number), group=self.base_n_filters * pow(2, block_number), kernel_size=2, stride=2, weight_filler=dict(type='constant', value=1), bias_term=False), param = dict(lr_mult=0, decay_mult=0))
        setattr(self.net_spec, 'upscale_' + str(block_number), l)

        l = L.Concat(l, getattr(self.net_spec, 'contr_' + str(block_number) + '_2'), axis=1)
        setattr(self.net_spec, 'concat_' + str(block_number), l)

        if self.use_batchnorm:
            l = self.add_batchnormscale(name='expand_' + str(block_number) + '_1', input=L.ReLU(L.Convolution(l, pad=self.pad, kernel_size=self.kernel_size, num_output=self.base_n_filters * pow(2, block_number-1), weight_filler=self.weight_filler, bias_filler=self.bias_filler, param=self.param), in_place=True))
            l = self.add_batchnormscale(name='expand_' + str(block_number) + '_2', input=L.ReLU(L.Convolution(l, pad=self.pad, kernel_size=self.kernel_size, num_output=self.base_n_filters * pow(2, block_number-1), weight_filler=self.weight_filler, bias_filler=self.bias_filler, param=self.param), in_place=True))
        else:
            l = self.add_conv(input=l, name='expand_' + str(block_number) + '_1', filter_mult=block_number-1)
            l = self.add_conv(input=l, name='expand_' + str(block_number) + '_2', filter_mult=block_number-1)

        return l

    def add_batchnormscale(self, input, name):

        if True : # necessary?
            batch_norm_param={'moving_average_fraction': 0.95, 'use_global_stats': True }
            param = [dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)]
            l = L.BatchNorm(input, name=name+'_bn', batch_norm_param=batch_norm_param, param=param, include={'phase': caffe.TEST}, ntop=1)
            setattr(self.net_spec, name+'_bn', l)

            batch_norm_param={'moving_average_fraction': 0.95, 'use_global_stats': False }
            l = L.BatchNorm(input, name=name+'_bn', top=name+'_bn', batch_norm_param=batch_norm_param, param=param, include={'phase': caffe.TRAIN}, ntop=0)
            setattr(self.net_spec, name+'_bn' + '_train', l)

            l = L.Scale(getattr(self.net_spec, name+'_bn'), scale_param = { 'bias_term': True } )
            setattr(self.net_spec, name, l)
        else : # here without split in use_global_stats True/False
            l = L.Scale(L.BatchNorm(input), scale_param={'bias_term': True})
            setattr(self.net_spec, name, l)

        return l

    def add_conv(self, input, name, filter_mult):
        l = L.Convolution(input, pad=self.pad, kernel_size=self.kernel_size,
                                          num_output=self.base_n_filters * pow(2, filter_mult),
                                          weight_filler=self.weight_filler)
        setattr(self.net_spec, name, l)
        return L.ReLU(l, in_place=True)

    def create_network_structure(self):

        self.pad = (self.kernel_size - 1) / 2
        self.net_spec = caffe.NetSpec()

        self.net_spec.data = L.Input(ntop=1, input_param={'shape': {'dim': [self.batch_size, self.data_channels, self.input_size, self.input_size]}})
        self.net_spec.target = L.Input(ntop=1, input_param={'shape': {'dim': [self.batch_size, self.label_channels, self.input_size, self.input_size]}}, exclude={'stage': 'deploy'})

        last_layer = self.net_spec.data
        for i in range(1,self.num_blocks+1) :
            last_layer = self.add_contraction_block(last_layer, i)

        if self.do_dropout:
            last_layer = L.Dropout(last_layer, dropout_ratio=0.4, in_place=True)

        if self.use_batchnorm:
            last_layer = self.add_batchnormscale(name='encode_1', input=L.ReLU(L.Convolution(last_layer, pad=self.pad, kernel_size=self.kernel_size, num_output=self.base_n_filters * pow(2, self.num_blocks), weight_filler=self.weight_filler), in_place=True))
            last_layer = self.add_batchnormscale(name='encode_2', input=L.ReLU(L.Convolution(last_layer, pad=self.pad, kernel_size=self.kernel_size, num_output=self.base_n_filters * pow(2, self.num_blocks), weight_filler=self.weight_filler), in_place=True))
        else:
            last_layer = self.add_conv(last_layer, name='encode_1', filter_mult=self.num_blocks)
            last_layer = self.add_conv(last_layer, name='encode_2', filter_mult=self.num_blocks)

        for i in range(1,self.num_blocks+1)[::-1] :
            last_layer = self.add_expansion_block(last_layer, i)

        self.net_spec.seg = L.Convolution(last_layer, pad=0, kernel_size=1, num_output=self.num_classes, weight_filler=self.weight_filler)

        self.net_spec.softmax = L.Softmax(self.net_spec.seg)
        self.net_spec.argmax = L.ArgMax(self.net_spec.softmax, axis=1)
        self.net_spec.silence = L.Silence(self.net_spec.argmax, ntop=0, include={'phase': caffe.TRAIN})
        self.net_spec.target_argmax = L.ArgMax(self.net_spec.target, axis=1, exclude={'stage': 'deploy'})

        if self.loss_func=='xent' :

            if self.ignore_label is None :
                self.net_spec.loss = L.SoftmaxWithLoss(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'})
                self.net_spec.accuracy = L.Accuracy(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'})
            else :
                self.net_spec.loss = L.SoftmaxWithLoss(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'}, loss_param={'ignore_label': self.ignore_label})
                self.net_spec.accuracy = L.Accuracy(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'}, accuracy_param={'ignore_label': self.ignore_label})

        elif self.loss_func=='dice' :

            if self.ignore_label is None :
                self.net_spec.loss = L.Python(self.net_spec.softmax, self.net_spec.target, loss_weight=1, python_param=dict( module='DiceLoss', layer='DiceLossLayer' ), exclude={'stage': 'deploy'} )
                self.net_spec.accuracy = L.Accuracy(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'})
            else :
                self.net_spec.loss = L.Python(self.net_spec.softmax, self.net_spec.target, loss_weight=1, python_param=dict(module='DiceLoss', layer='DiceLossLayer', param_str="{'param1': " + str(self.ignore_label) + "}"), exclude={'stage': 'deploy'})
                self.net_spec.accuracy = L.Accuracy(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'}, accuracy_param={'ignore_label': self.ignore_label})

        elif self.loss_func=='both' :

            if self.ignore_label is None :
                self.net_spec.xent_loss = L.SoftmaxWithLoss(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'}, loss_weight=10)
                self.net_spec.loss = L.Python(self.net_spec.softmax, self.net_spec.target, loss_weight=1, python_param=dict( module='DiceLoss', layer='DiceLossLayer' ), exclude={'stage': 'deploy'})
                self.net_spec.accuracy = L.Accuracy(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'})
            else :
                self.net_spec.xent_loss = L.SoftmaxWithLoss(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'}, loss_weight=10, loss_param={'ignore_label': self.ignore_label})
                self.net_spec.loss = L.Python(self.net_spec.softmax, self.net_spec.target, loss_weight=1, python_param=dict(module='DiceLoss', layer='DiceLossLayer', param_str="{'param1': " + str(self.ignore_label) + "}"), exclude={'stage': 'deploy'})
                self.net_spec.accuracy = L.Accuracy(self.net_spec.seg, self.net_spec.target_argmax, exclude={'stage': 'deploy'}, accuracy_param={'ignore_label': self.ignore_label})

        self.net_spec.dice = L.Python(self.net_spec.softmax, self.net_spec.target, loss_weight=1, python_param=dict(module='DiceIndex', layer='DiceIndexLayer'), exclude={'stage': 'deploy'})