#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


def get_logger(name):
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./logs/log.txt')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
