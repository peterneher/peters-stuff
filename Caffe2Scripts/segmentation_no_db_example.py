__author__ = "Peter F. Neher"
# Simple example for segmentation (U-Net) using Caffe2 including training, testing, saving and loading of networks

import numpy as np
from caffe2.python import workspace, core, model_helper, brew, optimizer, utils
from caffe2.proto import caffe2_pb2
import matplotlib.pyplot as plt

# randomly creates segmentation images with noise + ground truth
def get_data(batchsize, numsq=1) :
    data = []
    gt_segmentation = []
    sz = 64
    classes = 3
    for i in range(batchsize) :
        l = np.zeros((sz,sz,classes))
        if classes>1 :
            l[:,:,0] = 1
        for i in range(np.random.randint(0,numsq+1)) :
            s = np.random.randint(5, 20)
            x = np.random.randint(0, sz-s)
            y = np.random.randint(0, sz-s)
            l[x:x+s,y:y+s,0] = 0

            c = np.random.randint(1, classes)
            l[x:x+s,y:y+s,c] = 1

        noise = np.random.normal(0, 0.004, (sz,sz,classes))
        data.append(np.copy(l)+noise)
        gt_segmentation.append(l)
    return np.array(data).astype('float32'), np.array(gt_segmentation).astype('float32')

# create actual network structure (here U-net, Ronneberger et al.)
def create_unet_model(m, device_opts, is_test) :

    base_n_filters = 16
    kernel_size = 3
    pad = (kernel_size-1)/2
    do_dropout = True
    num_classes = 3

    weight_init=("MSRAFill", {})

    with core.DeviceScope(device_opts):

        contr_1_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, 'data', 'conv_1_1', dim_in=num_classes, dim_out=base_n_filters, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_1_1'), 'contr_1_1', dim_in=base_n_filters, epsilon=1e-3, momentum=0.1, is_test=is_test)
        contr_1_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, contr_1_1, 'conv_1_2', dim_in=base_n_filters, dim_out=base_n_filters, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_1_2'), 'contr_1_2', dim_in=base_n_filters, epsilon=1e-3, momentum=0.1, is_test=is_test)
        pool1 = brew.max_pool(m, contr_1_2, 'pool1', kernel=2, stride=2)

        contr_2_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, pool1, 'conv_2_1', dim_in=base_n_filters, dim_out=base_n_filters*2, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_2_1'), 'contr_2_1', dim_in=base_n_filters*2, epsilon=1e-3, momentum=0.1, is_test=is_test)
        contr_2_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, contr_2_1, 'conv_2_2', dim_in=base_n_filters*2, dim_out=base_n_filters*2, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_2_2'), 'contr_2_2', dim_in=base_n_filters*2, epsilon=1e-3, momentum=0.1, is_test=is_test)
        pool2 = brew.max_pool(m, contr_2_2, 'pool2', kernel=2, stride=2)

        contr_3_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, pool2, 'conv_3_1', dim_in=base_n_filters*2, dim_out=base_n_filters*4, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_3_1'), 'contr_3_1', dim_in=base_n_filters*4, epsilon=1e-3, momentum=0.1, is_test=is_test)
        contr_3_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, contr_3_1, 'conv_3_2', dim_in=base_n_filters*4, dim_out=base_n_filters*4, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_3_2'), 'contr_3_2', dim_in=base_n_filters*4, epsilon=1e-3, momentum=0.1, is_test=is_test)
        pool3 = brew.max_pool(m, contr_3_2, 'pool3', kernel=2, stride=2)

        contr_4_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, pool3, 'conv_4_1', dim_in=base_n_filters*4, dim_out=base_n_filters*8, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_4_1'), 'contr_4_1', dim_in=base_n_filters*8, epsilon=1e-3, momentum=0.1, is_test=is_test)
        contr_4_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, contr_4_1, 'conv_4_2', dim_in=base_n_filters*8, dim_out=base_n_filters*8, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_4_2'), 'contr_4_2', dim_in=base_n_filters*8, epsilon=1e-3, momentum=0.1, is_test=is_test)
        pool4 = brew.max_pool(m, contr_4_2, 'pool4', kernel=2, stride=2)

        if do_dropout:
            pool4 = brew.dropout(m, pool4, 'drop', ratio=0.4)

        encode_5_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, pool4, 'conv_5_1', dim_in=base_n_filters*8, dim_out=base_n_filters*16, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_5_1'), 'encode_5_1', dim_in=base_n_filters*16, epsilon=1e-3, momentum=0.1, is_test=is_test)
        encode_5_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, encode_5_1, 'conv_5_2', dim_in=base_n_filters*16, dim_out=base_n_filters*16, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_5_2'), 'encode_5_2', dim_in=base_n_filters*16, epsilon=1e-3, momentum=0.1, is_test=is_test)
        upscale5 = brew.conv_transpose(m, encode_5_2, 'upscale5', dim_in=base_n_filters*16, dim_out=base_n_filters*16, kernel=2, stride=2, weight_init=weight_init)

        concat6 = brew.concat(m, [upscale5, contr_4_2], 'concat6')#, axis=1)
        expand_6_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, concat6, 'conv_6_1', dim_in=base_n_filters * 8*3, dim_out=base_n_filters * 8, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_6_1'), 'expand_6_1', dim_in=base_n_filters * 8, epsilon=1e-3, momentum=0.1, is_test=is_test)
        expand_6_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, expand_6_1, 'conv_6_2', dim_in=base_n_filters * 8, dim_out=base_n_filters * 8, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_6_2'), 'expand_6_2', dim_in=base_n_filters * 8, epsilon=1e-3, momentum=0.1, is_test=is_test)
        upscale6 = brew.conv_transpose(m, expand_6_2, 'upscale6', dim_in=base_n_filters * 8, dim_out=base_n_filters * 8, kernel=2, stride=2, weight_init=weight_init)

        concat7 = brew.concat(m, [upscale6, contr_3_2], 'concat7')
        expand_7_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, concat7, 'conv_7_1', dim_in=base_n_filters * 4*3, dim_out=base_n_filters * 4, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_7_1'), 'expand_7_1', dim_in=base_n_filters * 4, epsilon=1e-3, momentum=0.1, is_test=is_test)
        expand_7_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, expand_7_1, 'conv_7_2', dim_in=base_n_filters * 4, dim_out=base_n_filters * 4, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_7_2'), 'expand_7_2', dim_in=base_n_filters * 4, epsilon=1e-3, momentum=0.1, is_test=is_test)
        upscale7 = brew.conv_transpose(m, expand_7_2, 'upscale7', dim_in=base_n_filters * 4, dim_out=base_n_filters * 4, kernel=2, stride=2, weight_init=weight_init)

        concat8 = brew.concat(m, [upscale7, contr_2_2], 'concat8')
        expand_8_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, concat8, 'conv_8_1', dim_in=base_n_filters * 2*3, dim_out=base_n_filters * 2, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_8_1'), 'expand_8_1', dim_in=base_n_filters * 2, epsilon=1e-3, momentum=0.1, is_test=is_test)
        expand_8_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, expand_8_1, 'conv_8_2', dim_in=base_n_filters * 2, dim_out=base_n_filters * 2, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_8_2'), 'expand_8_2', dim_in=base_n_filters * 2, epsilon=1e-3, momentum=0.1, is_test=is_test)
        upscale8 = brew.conv_transpose(m, expand_8_2, 'upscale8', dim_in=base_n_filters * 2, dim_out=base_n_filters * 2, kernel=2, stride=2, weight_init=weight_init)

        concat9 = brew.concat(m, [upscale8, contr_1_2], 'concat9')
        expand_9_1 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, concat9, 'conv_9_1', dim_in=base_n_filters * 3, dim_out=base_n_filters, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_9_1'), 'expand_9_1', dim_in=base_n_filters, epsilon=1e-3, momentum=0.1, is_test=is_test)
        expand_9_2 = brew.spatial_bn(m, brew.relu(m, brew.conv(m, expand_9_1, 'conv_9_2', dim_in=base_n_filters, dim_out=base_n_filters, kernel=kernel_size, pad=pad, weight_init=weight_init), 'nonl_9_2'), 'expand_9_2', dim_in=base_n_filters, epsilon=1e-3, momentum=0.1, is_test=is_test)

        output_segmentation = brew.conv(m, expand_9_2, 'output_segmentation', dim_in=base_n_filters, dim_out=num_classes, kernel=1, pad=0, stride=1, weight_init=weight_init)
        m.net.AddExternalOutput(output_segmentation)

        output_sigmoid = m.Sigmoid(output_segmentation, 'output_sigmoid')
        m.net.AddExternalOutput(output_sigmoid)

        return output_segmentation

# create actual network structure
def create_minimal_model(m, device_opts, is_test):

    base_n_filters = 16
    kernel_size = 3
    pad = (kernel_size - 1) / 2
    do_dropout = True
    num_output_channels = 3

    weight_init = ("MSRAFill", {})

    with core.DeviceScope(device_opts):
        contr_1_1 = brew.conv(m, 'data', 'contr_1_1', dim_in=1, dim_out=base_n_filters, kernel=kernel_size, pad=pad, weight_init=weight_init)
        pool1 = brew.max_pool(m, contr_1_1, 'pool1', kernel=2, stride=2)

        contr_2_1 = brew.conv(m, pool1, 'contr_2_1', dim_in=base_n_filters, dim_out=base_n_filters * 2, kernel=kernel_size, pad=pad, weight_init=weight_init)
        pool2 = brew.max_pool(m, contr_2_1, 'pool2', kernel=2, stride=2)

        contr_3_1 = brew.conv(m, pool2, 'contr_3_1', dim_in=base_n_filters * 2, dim_out=base_n_filters * 4, kernel=kernel_size, pad=pad, weight_init=weight_init)

        expand_7_1 = brew.conv(m, contr_3_1, 'expand_7_1', dim_in=base_n_filters * 4, dim_out=base_n_filters * 2, kernel=kernel_size, pad=pad, weight_init=weight_init)
        upscale7 = brew.conv_transpose(m, expand_7_1, 'upscale7', dim_in=base_n_filters * 2, dim_out=base_n_filters * 2, kernel=2, stride=2, weight_init=weight_init)

        expand_8_1 = brew.conv(m, upscale7, 'expand_8_1', dim_in=base_n_filters * 2, dim_out=base_n_filters * 2, kernel=kernel_size, pad=pad, weight_init=weight_init)
        upscale8 = brew.conv_transpose(m, expand_8_1, 'upscale8', dim_in=base_n_filters * 2, dim_out=base_n_filters * 2, kernel=2, stride=2, weight_init=weight_init)

        expand_9_1 = brew.conv(m, upscale8, 'expand_9_1', dim_in=base_n_filters * 3, dim_out=base_n_filters, kernel=kernel_size, pad=pad, weight_init=weight_init)

        output_segmentation = brew.conv(m, expand_9_1, 'output_segmentation', dim_in=base_n_filters, dim_out=num_output_channels, kernel=1, pad=0, stride=1, weight_init=weight_init)
        m.net.AddExternalOutput(output_segmentation)

        output_sigmoid = m.Sigmoid(output_segmentation, 'output_sigmoid')
        m.net.AddExternalOutput(output_sigmoid)

        return output_segmentation

# add loss and optimizer
def add_training_operators(output_segmentation, model, device_opts) :

    with core.DeviceScope(device_opts):
        loss = model.SigmoidCrossEntropyWithLogits([output_segmentation, "gt_segmentation"], 'loss')
        avg_loss = model.AveragedLoss(loss, "avg_loss")
        model.AddGradientOperators([loss])
        opt = optimizer.build_adam(model, base_learning_rate=0.01)

def train(INIT_NET, PREDICT_NET, epochs, batch_size, device_opts) :

    data, gt_segmentation = get_data(batch_size)
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.FeedBlob("gt_segmentation", gt_segmentation, device_option=device_opts)

    train_model= model_helper.ModelHelper(name="train_net", arg_scope = {"order": "NHWC"})
    output_segmentation = create_unet_model(train_model, device_opts=device_opts, is_test=0)
    add_training_operators(output_segmentation, train_model, device_opts=device_opts)
    with core.DeviceScope(device_opts):
        brew.add_weight_decay(train_model, 0.001)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    print '\ntraining for', epochs, 'epochs'
    for j in range(0, epochs):
        data, gt_segmentation = get_data(batch_size, 4)

        workspace.FeedBlob("data", data, device_option=device_opts)
        workspace.FeedBlob("gt_segmentation", gt_segmentation, device_option=device_opts)

        workspace.RunNet(train_model.net, 1)   # run for 10 times
        print str(j) + ': ' + str(workspace.FetchBlob("avg_loss"))

    print 'training done'
    test_model= model_helper.ModelHelper(name="test_net", arg_scope = {"order": "NHWC"}, init_params=False)
    create_unet_model(test_model, device_opts=device_opts, is_test=1)
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    print '\nsaving test model'
    save_net(INIT_NET, PREDICT_NET, test_model)

def save_net(INIT_NET, PREDICT_NET, model) :

    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    init_net = caffe2_pb2.NetDef()
    for param in model.params:
        #print param
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill", [], [param],arg=[ utils.MakeArgument("shape", shape),utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
    init_net.op.extend([core.CreateOperator("ConstantFill", [], ["data"], shape=get_data(1)[0][0,:,:,:].shape)])
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())

def load_net(INIT_NET, PREDICT_NET, device_opts):

    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'r') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())

    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'r') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

INIT_NET = '/home/neher/Projects/caffe2_networks/init_net_seg.pb'
PREDICT_NET = '/home/neher/Projects/caffe2_networks/predict_net_seg.pb'
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0) # change to 'core.DeviceOption(caffe2_pb2.CPU, 0)' for CPU processing

train(INIT_NET, PREDICT_NET, epochs=100, batch_size=100, device_opts=device_opts)

print '\n********************************************'
print 'loading test model'
# Problems loading batch-norm layer (SpatialBN) here!
load_net(INIT_NET, PREDICT_NET, device_opts=device_opts)

while True :
    data, gt_segmentation = get_data(1, 4)
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.RunNet('test_net', 1)
    out1 = workspace.FetchBlob("output_sigmoid")

    fig, sub = plt.subplots(ncols=3, figsize=(15, 5))

    sub[0].set_title('Input')
    sub[0].imshow(data[0,:,:,:])

    sub[1].set_title('Ground Truth')
    sub[1].imshow(gt_segmentation[0,:,:,:])

    sub[2].set_title('Segmentation')
    sub[2].imshow(out1[0, :, :, :])

    plt.show()
