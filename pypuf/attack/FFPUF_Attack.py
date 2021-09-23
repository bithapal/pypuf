"""
Attack code for Feed-Forward Arbiter PUF and homogeneous XOR Feed-Forward Arbiter PUF.
"""
import os
import sys
from datetime import datetime
from typing import List

import numpy as np

import pypuf.io
import pypuf.simulation.delay
from pypuf.batch import StudyBase

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import time

#import MLPAttack2021
import mlp2021
from mlp2021 import *
from mlp2021 import MLPAttack2021

class FFPUFAttack2021(MLPAttack2021):

    earlystop = MLPAttack2021.EarlyStopCallback()
    Model = MLPAttack2021.Model()


    def __init__(self, n:int, k:int, loops:List[Tuple[int, int]], N:int, seed: int, net: List[int], epochs: int, lr: float, bs: int,
                 early_stop: float, patience: int = None, activation_hl: str = 'tanh')-> None:
    '''    
        :param n: n is the number of stages, the total stages - the number of loops
        :type n: ``int``
        :param k: k is the number of components, used for homogeneous FFXOR PUF and heterogeneous FFXOR PUF
        for FFPUF, it is always 1.
        :type k: ``int``
        :param loops: list of loops for FFXOR PUF: (i,j) where i is the arbiter position and j is the loop position
        :type loops: ``List of tuples``
        :param N: N is the number of CRPs
        :type N: ``int``
        :param seed: random seed for model initlization
        :type seed: ``int``
        :param net: Hidden-layer sizes for the multilayer perceptron. Note that the layers are all *dense*, i.e. fully
        connected.
        :type net: ``List[int]``
        :param epochs: Maximum number of epochs performed.
        :type epochs: ``int``
        :param lr: Learning rate of the Adam optimizer used for optimization.
        :type lr: ``float``
        :param bs: Number of training examples that are processed together. Larger block size benefits from higher
        confidence of gradient direction and better computational performance, smaller block size benefits from
        earlier feedback of the weight adoption on following training steps.
        :type bs: ``int``
        :param early_stop: Training will stop when validation loss is below this threshold.
        :type early_stop: ``float``
        :param patience: Training will stop when validation loss did not improve for the given number of epochs.
        Counter is not reset after validation improved in one epoch.
        :type patience: ``Optional[int]``
        :param activation_hl: Activation function used on the hidden layers.
        :type activation_hl: ``str``
        '''
        super().__init__()
            self.n = n
            self.k = k
            self.loops = loops
            self.N = N
            self.net = net
            self.epochs = epochs
            self.lr = lr
            self.bs = bs
            self.seed = seed
            self.early_stop = early_stop
            self.patience = patience or epochs
            self.activation_hl = activation_hl
            self._history = None

    @staticmethod
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.binary_crossentropy(.5 - .5 * y_true, .5 - .5 * y_pred)

    @staticmethod
    def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.metrics.binary_accuracy(.5 - .5 * y_true, .5 - .5 * y_pred)

    @property
    def history(self) -> Optional[dict]:
        """
        After :meth:`fit` was called, returns a dictionary that contains information about the training process.
        The dictionary contains lists of length corresponding to the number of executed epochs:
        - ``loss`` the training loss,
        - ``val_loss`` the validation loss,
        - ``accuracy`` the training accuracy, and
        - ``val_accuracy`` the validation accuracy.
        """
        return self._history


    def transform(self, C, loops):
        loops_array = np.array(loops)
        ending_loops1 = np.array(loops_array[:, 1])
        print('ending loops: ', ending_loops1)
        C = np.cumprod(np.fliplr(C), axis=1)
        return C

    def fit(self) -> Model:

     
         puf = pypuf.simulation.delay.XORFeedForwardArbiterPUF(n=n, k=k, ff=loops, seed=seed_sim + 0,
                                                                  noisiness=noisiness)
         features = pypuf.io.random_inputs(n=n, N=N, seed=seed)
         labels = puf.eval(features)
         features = self.transform(features, loops)
         labels = .5 - .5 * labels

         #build network
         model = tf.keras.Sequential()
         model.add(tf.keras.layers.Dense(self.net[0], activation=self.activation_hl,
                                            input_dim=len(features[0, :]), kernel_initializer='random_normal'))
         for layer in self.net[1:]:
            model.add(tf.keras.layers.Dense(layer, activation=self.activation_hl))

         model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
         opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
         model.compile(optimizer=opt, loss=self.loss, metrics=[self.accuracy])

        # fit
        self._history = model.fit(
            features, labels,
            epochs=self.epochs,
            batch_size=self.bs,
            callbacks=earlystop(self.early_stop, self.patience),
            shuffle=True,
            validation_split=0.01,
        ).history

        #create pypuf model
        self._model = self.Model(model, challenge_length=len(features[0, :]))
        return self._model
     




    if __name__ == '__main__':
        PUFDLStudy.cli(sys.argv)