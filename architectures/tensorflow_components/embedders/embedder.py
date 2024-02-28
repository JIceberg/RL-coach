#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Union, Tuple
import copy
import sys

import numpy as np
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import BatchnormActivationDropout, convert_layer, Dense, FuncLayer
from rl_coach.base_parameters import EmbedderScheme, NetworkComponentParameters

from rl_coach.core_types import InputEmbedding
from rl_coach.utils import force_list

def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

class InputEmbedder(object):
    """
    An input embedder is the first part of the network, which takes the input from the state and produces a vector
    embedding by passing it through a neural network. The embedder will mostly be input type dependent, and there
    can be multiple embedders in a single network
    """
    def __init__(self, input_size: List[int], activation_function=tf.nn.relu,
                 scheme: EmbedderScheme=None, batchnorm: bool=False, dropout_rate: float=0.0,
                 name: str= "embedder", input_rescaling=1.0, input_offset=0.0, input_clipping=None, dense_layer=Dense,
                 is_training=False):
        self.name = name
        self.input_size = input_size
        self.activation_function = activation_function
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.input = None
        self.output = None
        self.scheme = scheme
        self.return_type = InputEmbedding
        self.layers_params = []
        self.layers = []
        self.input_rescaling = input_rescaling
        self.input_offset = input_offset
        self.input_clipping = input_clipping
        self.dense_layer = dense_layer
        self.thresh_upper = {}
        self.stats = {}
        self.means = {}
        self.thresh_layers = {}
        self.batch_layers = {}
        self.thresh_mask = {}
        self.dummy_input = {}
        self.thresholding_flag = False
        self.existingAggregate = {}
        if self.dense_layer is None:
            self.dense_layer = Dense
        self.is_training = is_training

        # layers order is conv -> batchnorm -> activation -> dropout
        if isinstance(self.scheme, EmbedderScheme):
            self.layers_params = copy.copy(self.schemes[self.scheme])
            self.layers_params = [convert_layer(l) for l in self.layers_params]
        else:
            # if scheme is specified directly, convert to TF layer if it's not a callable object
            # NOTE: if layer object is callable, it must return a TF tensor when invoked
            self.layers_params = []
            for l in copy.copy(self.scheme):
                layer_param = convert_layer(l)
                self.layers_params.append(layer_param)

        # we allow adding batchnorm, dropout or activation functions after each layer.
        # The motivation is to simplify the transition between a network with batchnorm and a network without
        # batchnorm to a single flag (the same applies to activation function and dropout)
        if self.batchnorm or self.activation_function or self.dropout_rate > 0:
            for layer_idx in reversed(range(len(self.layers_params))):
                self.layers_params.insert(layer_idx+1,
                                          BatchnormActivationDropout(batchnorm=self.batchnorm,
                                                                     activation_function=self.activation_function,
                                                                     dropout_rate=self.dropout_rate))

    def __call__(self, prev_input_placeholder: tf.placeholder=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wrapper for building the module graph including scoping and loss creation
        :param prev_input_placeholder: the input to the graph
        :return: the input placeholder and the output of the last layer
        """
        with tf.variable_scope(self.get_name()):
            if prev_input_placeholder is None:
                self.input = tf.placeholder("float", shape=[None] + self.input_size, name=self.get_name())
            else:
                self.input = prev_input_placeholder
            self._build_module()

        return self.input, self.output
    
    def get_layers(self):
        return self.layers
    
    def get_layer_params(self):
        return self.layers_params

    def get_aggregate(self, key):
        return self.existingAggregate[key]

    def set_aggregate(self, newValue, key):
        self.existingAggregate[key] = newValue

    def get_output(self):
        return self.output
    
    def get_thresh_layers(self):
        return self.thresh_layers

    def finalize_stats(self):
        for key in self.stats.keys():
            existing_agg = self.existingAggregate[key]
            self.stats[key] = finalize(existing_agg)

    def set_thresholds(self, prob_val=0.00025):
        for key in self.stats.keys():
            if isinstance(self.stats[key], tuple):
                thresh_cantelli = (1-prob_val)*self.stats[key][2]/prob_val
                self.thresh_upper[key] = np.sqrt(thresh_cantelli)
                self.means[key] = self.stats[key][0]
    
    def forward_thresholding(self, x, key):
        out = np.copy(x)
        if self.thresholding_flag:
            out[np.isnan(x)] = 0
            thresh_mask = np.logical_or(out - self.means[key] > self.thresh_upper[key], out < 0)
            out[thresh_mask] = 0
        return out

    def dummy_func(self, x):
        return x

    def _build_module(self) -> None:
        """
        Builds the graph of the module
        This method is called early on from __call__. It is expected to store the graph
        in self.output.
        :return: None
        """
        # NOTE: for image inputs, we expect the data format to be of type uint8, so to be memory efficient. we chose not
        #  to implement the rescaling as an input filters.observation.observation_filter, as this would have caused the
        #  input to the network to be float, which is 4x more expensive in memory.
        #  thus causing each saved transition in the memory to also be 4x more pricier.

        input_layer = self.input / self.input_rescaling
        input_layer -= self.input_offset
        # clip input using te given range
        if self.input_clipping is not None:
            input_layer = tf.clip_by_value(input_layer, self.input_clipping[0], self.input_clipping[1])

        self.layers.append(input_layer)

        curr_layer_num = -1

        for idx, layer_params in enumerate(self.layers_params):
            self.layers.extend(force_list(
                layer_params(input_layer=self.layers[-1], name='{}_{}'.format(layer_params.__class__.__name__, idx),
                            is_training=self.is_training)
            ))
            if layer_params.__class__.__name__ == 'BatchnormActivationDropout':
                curr_layer_num += 1
                curr_layer_name = f"conv{curr_layer_num}"
                self.batch_layers[curr_layer_name] = self.layers[-1]
                thresh_shape = self.layers[-1].get_shape().as_list()[1:]
                thresh_shape = [1] + thresh_shape
                self.existingAggregate[curr_layer_name] = (
                    0,
                    np.zeros(shape=thresh_shape, dtype=np.float32),
                    np.zeros(shape=thresh_shape, dtype=np.float32)
                )
                self.stats[curr_layer_name] = (
                    0,
                    np.zeros(shape=thresh_shape, dtype=np.float32),
                    np.zeros(shape=thresh_shape, dtype=np.float32)
                )
                self.thresh_upper[curr_layer_name] = None
                self.means[curr_layer_name] = None

                # now, create a dummy layer that is a placeholder that serves as a
                # storage for the results of the forward thresholding
                dummy_layer = FuncLayer(func=self.dummy_func)
                
                # create the dummy input
                dummy_input = tf.placeholder(dtype=tf.float32, shape=self.layers[-1].get_shape())
                self.dummy_input[curr_layer_name] = dummy_input

                # add the dummy layer to the layer list
                self.layers.extend(force_list(dummy_layer(inputs=dummy_input)))
                self.thresh_layers[curr_layer_name] = self.layers[-1]

        self.output = tf.contrib.layers.flatten(self.layers[-1])

    @property
    def input_size(self) -> List[int]:
        return self._input_size

    @input_size.setter
    def input_size(self, value: Union[int, List[int]]):
        if isinstance(value, np.ndarray) or isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, int):
            value = [value]
        if not isinstance(value, list):
            raise ValueError((
                'input_size expected to be a list, found {value} which has type {type}'
            ).format(value=value, type=type(value)))
        self._input_size = value

    @property
    def schemes(self):
        raise NotImplementedError("Inheriting embedder must define schemes matching its allowed default "
                                  "configurations.")

    def get_name(self) -> str:
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return self.name

    def __str__(self):
        result = ['Input size = {}'.format(self._input_size)]
        if self.input_rescaling != 1.0 or self.input_offset != 0.0:
            result.append('Input Normalization (scale = {}, offset = {})'.format(self.input_rescaling, self.input_offset))
        result.extend([str(l) for l in self.layers_params])
        if not self.layers_params:
            result.append('No layers')

        return '\n'.join(result)
