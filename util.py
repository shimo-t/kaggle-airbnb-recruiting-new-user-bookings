#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def logit(p):
    return np.log(p / (1. - p))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
