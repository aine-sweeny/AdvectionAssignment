# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:33:50 2026

@author: he820946
"""

import subprocess

subprocess.call(["python","advection.py"])
subprocess.call(["python","growingerrorsCourant.py"])
subprocess.call(["python","Convergence.py"])
