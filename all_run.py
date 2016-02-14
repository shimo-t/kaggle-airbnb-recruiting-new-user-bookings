#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import Dataset
import predict


dataset_name = 'b88369b'

dataset = Dataset(name=dataset_name)
dataset.generate()
dataset.save_pkl()

dataset = Dataset(name=dataset_name).load_pkl()
predict.predict(dataset)
