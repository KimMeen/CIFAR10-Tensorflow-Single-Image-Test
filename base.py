# -*- coding: utf-8 -*-
"""
@author: Ming JIN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
