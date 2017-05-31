__author__ = "Peter F. Neher"
# Minimal example for classification using Caffe2 including training, testing, saving and loading of networks

import numpy as np
from caffe2.python import workspace, core, model_helper, brew, optimizer, utils
from caffe2.proto import caffe2_pb2

# randomly creates 30x30 patches of ones or zeros with label 1 and 0 respectively
def get_data(batchsize) :
    data = []
    label = []
    for i in range(batchsize) :
        r = np.random.randint(0, 2)
        if r==0 :
            d = np.zeros((1,30,30))
            l = 0
        else :
            d = np.ones((1,30,30))
            l = 1
        data.append(d)
        label.append(l)
    return np.array(data).astype('float32'), np.array(label).astype('int32')

# create actual network structure (from input to output (here softmax))
def create_model(m, device_opts) :
    with core.DeviceScope(device_opts):

        conv1 = brew.conv(m, 'data', 'conv1', dim_in=1, dim_out=20, kernel=5)
        pool1 = brew.max_pool(m, conv1, 'pool1', kernel=2, stride=2)
        conv2 = brew.conv(m, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
        pool2 = brew.max_pool(m, conv2, 'pool2', kernel=2, stride=2)
        fc3 = brew.fc(m, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
        fc3 = brew.relu(m, fc3, fc3)
        pred = brew.fc(m, fc3, 'pred', 500, 2)
        softmax = brew.softmax(m, pred, 'softmax')
        m.net.AddExternalOutput(softmax)
        return softmax

# add loss and optimizer
def add_training_operators(softmax, model, device_opts) :

    with core.DeviceScope(device_opts):
        xent = model.LabelCrossEntropy([softmax, "label"], 'xent')
        loss = model.AveragedLoss(xent, "loss")
        brew.accuracy(model, [softmax, "label"], "accuracy")

        model.AddGradientOperators([loss])
        opt = optimizer.build_sgd(model, base_learning_rate=0.01, policy="step", stepsize=1, gamma=0.999)  # , momentum=0.9
        #opt = optimizer.build_adam(model, base_learning_rate=0.001)

def train(INIT_NET, PREDICT_NET, epochs, batch_size, device_opts) :

    data, label = get_data(batch_size)
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.FeedBlob("label", label, device_option=device_opts)

    train_model= model_helper.ModelHelper(name="train_net")
    softmax = create_model(train_model, device_opts=device_opts)
    add_training_operators(softmax, train_model, device_opts=device_opts)
    with core.DeviceScope(device_opts):
        brew.add_weight_decay(train_model, 0.001)  # any effect???

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    print '\ntraining for', epochs, 'epochs'

    for j in range(0, epochs):
        data, label = get_data(batch_size)

        workspace.FeedBlob("data", data, device_option=device_opts)
        workspace.FeedBlob("label", label, device_option=device_opts)

        workspace.RunNet(train_model.net, 10)   # run for 10 times
        print str(j) + ': ' + str(workspace.FetchBlob("loss")) + ' - ' + str(workspace.FetchBlob("accuracy"))

    print 'training done'

    print '\nrunning test model'

    test_model= model_helper.ModelHelper(name="test_net", init_params=False)
    create_model(test_model, device_opts=device_opts)
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    data = np.zeros((1,1,30,30)).astype('float32')
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.RunNet(test_model.net, 1)
    print "\nInput: zeros"
    print "Output:", workspace.FetchBlob("softmax")
    print "Output class:", np.argmax(workspace.FetchBlob("softmax"))

    data = np.ones((1,1,30,30)).astype('float32')
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.RunNet(test_model.net, 1)
    print "\nInput: ones"
    print "Output:", workspace.FetchBlob("softmax")
    print "Output class:", np.argmax(workspace.FetchBlob("softmax"))

    print '\nsaving test model'

    save_net(INIT_NET, PREDICT_NET, test_model)

def save_net(INIT_NET, PREDICT_NET, model) :

    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    init_net = caffe2_pb2.NetDef()
    for param in model.params:
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill", [], [param],arg=[ utils.MakeArgument("shape", shape),utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
    init_net.op.extend([core.CreateOperator("ConstantFill", [], ["data"], shape=(1,30,30))])
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

INIT_NET = '/path/to/init_net.pb'
PREDICT_NET = '/path/to/predict_net.pb'
device_opts = core.DeviceOption(caffe2_pb2.CPU, 0) # change to 'core.DeviceOption(caffe2_pb2.CUDA, 0)' for GPU processing

train(INIT_NET, PREDICT_NET, epochs=3, batch_size=100, device_opts=device_opts)

print '\n********************************************'
print 'loading test model'
load_net(INIT_NET, PREDICT_NET, device_opts=device_opts)

data = np.ones((1,1,30,30)).astype('float32')
workspace.FeedBlob("data", data, device_option=device_opts)
workspace.RunNet('test_net', 1)

print "\nInput: ones"
print "Output:", workspace.FetchBlob("softmax")
print "Output class:", np.argmax(workspace.FetchBlob("softmax"))
