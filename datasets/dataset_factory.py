# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# from AutoML_refactor.datasets import customized
# from AutoML_refactor.datasets import imagenet
# from AutoML_refactor.datasets import mnist
from AutoML_refactor.datasets import cifar10
from AutoML_refactor.datasets import flowers
from AutoML_refactor.datasets import preprocessing


datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    # 'customized': customized,
    # 'imagenet': imagenet,
    # 'mnist': mnist,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)


def get_data(hyp, is_training):
    if is_training:
        dataset = get_dataset(hyp.dataset_name, hyp.train_split_name, hyp.DATA_PATH)
    else:
        dataset = get_dataset(hyp.dataset_name, hyp.test_split_name, hyp.DATA_PATH)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=4,
                                                              common_queue_capacity=20 * hyp.BATCH_SIZE,
                                                              common_queue_min=10 * hyp.BATCH_SIZE)
    [image, label] = provider.get(['image', 'label'])
    image_train = preprocessing.preprocess_image(image, hyp.IMAGE_SIZE, hyp.IMAGE_SIZE, is_training)
    images_train, labels_train = tf.train.batch(
                                                [image_train, label],
                                                batch_size=hyp.BATCH_SIZE,
                                                num_threads=4,
                                                capacity=200)
    labels_train = slim.one_hot_encoding(labels_train, dataset.num_classes)
    batch_queue = slim.prefetch_queue.prefetch_queue([images_train, labels_train], capacity=20)
    images, labels = batch_queue.dequeue()
    return images, labels
