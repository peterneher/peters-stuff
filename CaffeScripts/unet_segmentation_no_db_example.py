__author__ = "Peter F. Neher"
# Simple example for segmentation (U-Net) using Caffe

import numpy as np
import matplotlib.pyplot as plt
import caffe
from caffe import layers as L
from caffe import params as P
from scipy import misc

# randomly creates segmentation images with noise + ground truth
def get_data(batchsize, numsq=4, numclasses=3) :
    data = []
    gt_segmentation = []
    sz = 64
    for i in range(batchsize) :
        d = np.zeros((1,sz,sz))
        for i in range(np.random.randint(1,numsq+1)) :
            s = np.random.randint(5, 20)
            x = np.random.randint(0, sz-s)
            y = np.random.randint(0, sz-s)

            c = np.random.randint(1, numclasses)
            d[0,x:x+s,y:y+s] = c

        noise = np.random.normal(0, 0.1, (1,sz,sz))
        data.append(np.copy(d)+noise)
        gt_segmentation.append(np.copy(d))
    return np.array(data).astype('float32'), np.array(gt_segmentation).astype('float32')

# create actual network structure (here U-net, Ronneberger et al.) and save it to .prototxt
def create_unet_model(batch_size, num_classes, input_size, base_n_filters, output_file):

    kernel_size = 3
    pad = (kernel_size - 1) / 2
    do_dropout = True
    weight_filler = dict(type='msra')

    n = caffe.NetSpec()

    n.data = L.Input(ntop=1, input_param =  { 'shape' : { 'dim': [batch_size, 1, input_size, input_size] } })
    n.target = L.Input(ntop=1, input_param =  { 'shape' : { 'dim': [batch_size, 1, input_size, input_size] } }, exclude={'stage' : 'deploy'})

    n.contr_1_1 = L.BatchNorm(L.ReLU(L.Convolution(n.data, pad=pad, kernel_size=kernel_size, num_output=base_n_filters, weight_filler=weight_filler), in_place=True), in_place=True)
    n.contr_1_2 = L.BatchNorm(L.ReLU(L.Convolution(n.contr_1_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters, weight_filler=weight_filler), in_place=True), in_place=True)
    n.pool_1 = L.Pooling(n.contr_1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.contr_2_1 = L.BatchNorm(L.ReLU(L.Convolution(n.pool_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 2, weight_filler=weight_filler), in_place=True), in_place=True)
    n.contr_2_2 = L.BatchNorm(L.ReLU(L.Convolution(n.contr_2_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 2, weight_filler=weight_filler), in_place=True), in_place=True)
    n.pool_2 = L.Pooling(n.contr_2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.contr_3_1 = L.BatchNorm(L.ReLU(L.Convolution(n.pool_2, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 4, weight_filler=weight_filler), in_place=True), in_place=True)
    n.contr_3_2 = L.BatchNorm(L.ReLU(L.Convolution(n.contr_3_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 4, weight_filler=weight_filler), in_place=True), in_place=True)
    n.pool_3 = L.Pooling(n.contr_3_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.contr_4_1 = L.BatchNorm(L.ReLU(L.Convolution(n.pool_3, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 8, weight_filler=weight_filler), in_place=True), in_place=True)
    n.contr_4_2 = L.BatchNorm(L.ReLU(L.Convolution(n.contr_4_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 8, weight_filler=weight_filler), in_place=True), in_place=True)
    n.pool_4 = L.Pooling(n.contr_4_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    if do_dropout:
        n.pool_4 = L.Dropout(n.pool_4, dropout_ratio=0.4, in_place=True)

    n.encode_1 = L.BatchNorm(L.ReLU(L.Convolution(n.pool_4, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 16, weight_filler=weight_filler), in_place=True), in_place=True)
    n.encode_2 = L.BatchNorm(L.ReLU(L.Convolution(n.encode_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 16, weight_filler=weight_filler), in_place=True), in_place=True)
    n.upscale_1 = L.Deconvolution(n.encode_2, convolution_param=dict(num_output=base_n_filters * 16, kernel_size=2, stride=2))

    n.concat_1 = L.Concat(n.upscale_1, n.contr_4_2, axis=1)
    n.expand_1_1 = L.BatchNorm(L.ReLU(L.Convolution(n.concat_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 8, weight_filler=weight_filler), in_place=True), in_place=True)
    n.expand_1_2 = L.BatchNorm(L.ReLU(L.Convolution(n.expand_1_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 8, weight_filler=weight_filler), in_place=True), in_place=True)
    n.upscale_2 = L.Deconvolution(n.expand_1_2, convolution_param=dict(num_output=base_n_filters * 8, kernel_size=2, stride=2))

    n.concat_2 = L.Concat(n.upscale_2, n.contr_3_2, axis=1)
    n.expand_2_1 = L.BatchNorm(L.ReLU(L.Convolution(n.concat_2, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 4, weight_filler=weight_filler), in_place=True), in_place=True)
    n.expand_2_2 = L.BatchNorm(L.ReLU(L.Convolution(n.expand_2_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 4, weight_filler=weight_filler), in_place=True), in_place=True)
    n.upscale_3 = L.Deconvolution(n.expand_2_2, convolution_param=dict(num_output=base_n_filters * 4, kernel_size=2, stride=2))

    n.concat_3 = L.Concat(n.upscale_3, n.contr_2_2, axis=1)
    n.expand_3_1 = L.BatchNorm(L.ReLU(L.Convolution(n.concat_3, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 2, weight_filler=weight_filler), in_place=True), in_place=True)
    n.expand_3_2 = L.BatchNorm(L.ReLU(L.Convolution(n.expand_3_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters * 2, weight_filler=weight_filler), in_place=True), in_place=True)
    n.upscale_4 = L.Deconvolution(n.expand_3_2, convolution_param=dict(num_output=base_n_filters * 2, kernel_size=2, stride=2))

    n.concat_4 = L.Concat(n.upscale_4, n.contr_1_2, axis=1)
    n.expand_4_1 = L.BatchNorm(L.ReLU(L.Convolution(n.concat_4, pad=pad, kernel_size=kernel_size, num_output=base_n_filters, weight_filler=weight_filler), in_place=True), in_place=True)
    n.expand_4_2 = L.BatchNorm(L.ReLU(L.Convolution(n.expand_4_1, pad=pad, kernel_size=kernel_size, num_output=base_n_filters, weight_filler=weight_filler), in_place=True), in_place=True)

    n.seg = L.Convolution(n.expand_4_2, pad=0, kernel_size=1, num_output=num_classes, weight_filler=weight_filler)

    n.softmax = L.Softmax(n.seg, include={'phase':caffe.TEST})
    n.argmax = L.ArgMax(n.softmax, axis=1, include={'phase':caffe.TEST})
    n.loss = L.SoftmaxWithLoss(n.seg, n.target, include={'phase':caffe.TRAIN})
    n.accuracy = L.Accuracy(n.seg, n.target, exclude={'stage' : 'deploy'})

    if output_file is not None :
        f = open(output_file, 'w')
        f.write(str(n.to_proto()))
        f.close()

    return n

def train_network(solver_file, output_weights, batch_size, num_iterations, stop_loss_thres=0.01, use_gpu=True) :

    if use_gpu :
        caffe.set_mode_gpu()
    else :
        caffe.set_mode_cpu()

    solver = caffe.get_solver(solver_file)
    solver.net.blobs['data'].reshape(batch_size, 1, 64, 64)
    solver.net.blobs['target'].reshape(batch_size, 1, 64, 64)
    solver.net.reshape()

    for i in range(num_iterations):

        data, target = get_data(batch_size, numclasses=3)
        solver.net.blobs['data'].data[...] = data
        solver.net.blobs['target'].data[...] = target
        solver.step(1)
        output = solver.net.blobs['seg'].data[...]
        loss = solver.net.blobs['loss'].data

        if loss < stop_loss_thres:

            solver.net.save(output_weights)

            fig, sub = plt.subplots(ncols=3, figsize=(15, 5))
            sub[0].set_title('Input')
            sub[0].imshow(data[0, 0, :, :])
            sub[1].set_title('Ground Truth')
            sub[1].imshow(target[0, 0, :, :])
            sub[2].set_title('Segmentation')
            sub[2].imshow(np.argmax(output[0, :, :, :], axis=0))
            plt.show()

            break

def test_network(model_file, weights_file, image_file) :
    caffe.set_mode_gpu()
    net = caffe.Net(model_file, caffe.TEST, weights=weights_file)
    # TODO

def print_all_available_layers() :
    layer_ist = caffe.layer_type_list()
    for el in layer_ist:
        print el

def print_network_sizes(model_file) :
    net = caffe.Net(model_file, caffe.TRAIN)
    for k, v in net.blobs.items():
        print k, v.data.shape

# NETWORK CREATION (only necessary once since the network is written to the output .prototxt file)
model_file = 'unet_structure.prototxt'
create_unet_model(batch_size=16, num_classes=3, input_size=64, base_n_filters=8, output_file=model_file)

# TRAIN NETWORK
solver_file = 'unet_solver.prototxt'
output_weights = 'unet_weights.caffemodel'
train_network(solver_file=solver_file, output_weights=output_weights, batch_size=16, num_iterations=10000, stop_loss_thres=0.001, use_gpu=True)





