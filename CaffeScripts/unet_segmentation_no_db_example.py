__author__ = "Peter F. Neher"
# Example for segmentation (U-Net) using Caffe

import numpy as np
import matplotlib.pyplot as plt
import caffe
from CaffeNetworks.CaffeUNet_2D import CaffeUNet_2D

# randomly creates segmentation images with noise + ground truth
def get_data(batchsize, sz = 64, numsq=5, numclasses=3) :
    data = []
    gt_segmentation = []

    for i in range(batchsize) :
        d = np.zeros((1,sz,sz))
        l = np.zeros((numclasses,sz,sz))
        l[0,:,:] = 1
        num = np.random.randint(1,numsq+1)
        for i in range(num) :
            s = np.random.randint(5, 20)
            x = np.random.randint(0, sz-s)
            y = np.random.randint(0, sz-s)

            c = np.random.randint(1, numclasses)
            d[0,x:x+s,y:y+s] = c
            l[:,x:x+s,y:y+s] = 0
            l[c,x:x+s,y:y+s] = 1

        d -= d.mean()
        d /= d.std()
        noise = np.random.normal(0, 0.1, (1,sz,sz))
        d += noise
        data.append(d)
        gt_segmentation.append(l)

    return np.array(data).astype('float32'), np.array(gt_segmentation).astype('float32')

def train_network(solver_file, num_classes, batch_size, num_iterations, use_gpu=True) :

    if use_gpu :
        caffe.set_mode_gpu()
    else :
        caffe.set_mode_cpu()

    solver = caffe.get_solver(solver_file)
    solver.net.blobs['data'].reshape(batch_size, solver.net.blobs['data'].shape[1], solver.net.blobs['data'].shape[2], solver.net.blobs['data'].shape[3])
    solver.net.blobs['target'].reshape(batch_size, solver.net.blobs['target'].shape[1], solver.net.blobs['target'].shape[2], solver.net.blobs['target'].shape[3])
    solver.net.reshape()

    for i in range(num_iterations):

        data, target = get_data(batch_size, numclasses=num_classes)

        solver.net.blobs['data'].data[...] = data
        solver.net.blobs['target'].data[...] = target
        solver.step(1)
        output = solver.net.blobs['argmax'].data[...]

    fig, sub = plt.subplots(ncols=3, figsize=(15, 5))
    sub[0].set_title('Input')
    sub[0].imshow(data[0, 0, :, :])
    sub[1].set_title('Ground Truth')
    sub[1].imshow(np.argmax(target[0, :, :, :], axis=0))
    sub[2].set_title('Segmentation')
    sub[2].imshow(output[0, 0, :, :])
    plt.show()

def print_all_available_layers() :
    layer_ist = caffe.layer_type_list()
    for el in layer_ist:
        print el

def print_network_sizes(model_file) :
    net = caffe.Net(model_file, caffe.TRAIN)
    for k, v in net.blobs.items():
        print k, v.data.shape

num_classes = 3

# NETWORK CREATION (only necessary once since the network is written to the output .prototxt file)
net_spec_file = 'unet_structure.prototxt'

unet = CaffeUNet_2D(loss_func='xent', batch_size=32, input_size=64, base_n_filters=32, num_blocks=4, num_classes=num_classes, data_channels=1, label_channels=num_classes)
unet.save_net_spec(net_spec_file)

# TRAIN NETWORK
solver_file = 'unet_solver.prototxt'
train_network(solver_file=solver_file, num_classes=num_classes, batch_size=32, num_iterations=1000, use_gpu=True)





