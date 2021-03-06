__author__ = "Peter F. Neher"

import caffe
import numpy as np
import theano
import theano.tensor as T

class DiceLossLayer(caffe.Layer):
    """
    Compute loss based on dice coefficient.
    """
    f = None
    g = None
    ignore_label = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")

        try :
            params = eval(self.param_str)
            if "param1" in params :
                self.ignore_label = int(params["param1"])
        except :
            pass

        if len(bottom[0].data.shape)==4 :
            self.prediction = T.fmatrix()
            self.ground_truth = T.fmatrix()
        elif len(bottom[0].data.shape)==5 :
            self.prediction = T.ftensor3()
            self.ground_truth = T.ftensor3()
        else:
            raise Exception('DiceIndexLayer only supports 2D or 3D data at the moment.')

        intersection = T.sum(self.prediction * self.ground_truth)
        denominator = T.sum(self.prediction) + T.sum(self.ground_truth)
        dice = 1 - 2 * intersection / (denominator + 0.00001)

        self.f = theano.function([self.prediction, self.ground_truth], dice)

        grad = T.grad(dice, wrt=self.prediction)
        self.g = theano.function([self.prediction, self.ground_truth], grad)



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            print bottom[0].data.shape
            print bottom[1].data.shape
            raise Exception("the dimension of inputs should match")

        top[0].reshape(bottom[0].data.shape[1])

    def forward(self, bottom, top):

        dice = np.zeros(bottom[0].data.shape[1])
        for batch_idx in range(0,bottom[0].data.shape[0]):

            sample = bottom[0].data[batch_idx]
            gt = bottom[1].data[batch_idx]

            for c_idx in range(0,bottom[0].data.shape[1]):
                dice[c_idx] += self.f(sample[c_idx], gt[c_idx])

        top[0].data[...]=dice/bottom[0].data.shape[0]

    def backward(self, top, propagate_down, bottom):

        for batch_idx in range(0,bottom[0].data.shape[0]):

            for c_idx in range(0, bottom[0].data.shape[1]):

                bottom[0].diff[batch_idx,c_idx] = self.g(bottom[0].data[batch_idx, c_idx], bottom[1].data[batch_idx, c_idx])


        if self.ignore_label is not None :
            bottom[0].diff[:, self.ignore_label] *= np.logical_not( bottom[1].data[:,self.ignore_label] )
