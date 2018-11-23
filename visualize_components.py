#!/usr/bin/env python
##
##  High Performance Computing for Science and Engineering (HPCSE) 2018
##  TDLL: Tiny Deep Learning Library - solution code for exercises 6 and 7.
##
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@gmail.com).
##
##  Visualize principal components from the various autoencoders.

import os, pathlib, sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

p = pathlib.Path('./')
for comp in list(p.glob('component_*.raw')):
    D = np.fromfile(comp.name, dtype=np.float32)
    D.resize([28, 28])
    plt.title("%s" % comp)
    plt.imshow(D)
    plt.show()
