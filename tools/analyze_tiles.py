#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:18:02 2024

@author: boyuantian
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))
                   
def pearson_correlation(v1, v2):
    cov_matrix = np.cov(v1, v2)
    return cov_matrix[0, 1] / (np.std(v1) * np.std(v2))

def nw_similarity(v1, v2):
    v1 = [str(val) for val in v1]
    v2 = [str(val) for val in v2]
    alignments = pairwise2.align.globalms(v1, v2, 2, -1, -0.5, -0.1, gap_char=['-'])
    best_alignment = alignments[0]
    alignment_score = best_alignment[2]
    # print (alignment_score / max(len(v1), len(v2)))
    return alignment_score / max(len(v1), len(v2))

def overlap_ratio(v1, v2):
    intersection = set(v1) & set(v2)
    ratio = len(intersection) / min(len(v1), len(v2))
    print (ratio)
    return ratio
    

path = "/home/boyuantian/Desktop/3DGS/MonoGS/results/replica_office0/2024-06-28-14-12-22/log_tile"

def plot_tile(iter_idx):
    filename = "frame_4_cam_0_iter_%s.log" %iter_idx
    # filename = "frame_4_cam_0_iter_0.log"

    f = open(os.path.join(path, filename))
    lines = f.readlines()
    f.close()
    
    frame1 = []
    for idx, line in enumerate(lines):
        if ("#" in line):
            continue
        line = line.strip()[:-1].split(", ")[1:]
        # print (line)
        frame1.append([int(val) for val in line])
    
    
    
    filename = "frame_4_cam_0_iter_%s.log" %(iter_idx+1)
    f = open(os.path.join(path, filename))
    lines = f.readlines()
    f.close()
    
    frame2 = []
    for idx, line in enumerate(lines):
        if ("#" in line):
            continue
        line = line.strip()[:-1].split(", ")[1:]
        frame2.append([int(val) for val in line])
    
    # frame1 = np.array(frame1)
    # frame2 = np.array(frame2)
    
    img = np.zeros((43, 75))
    for tile_idx in range(3225):
        x = int(tile_idx % 75)
        y = int(tile_idx / 75)
        # img[y, x] = nw_similarity(frame1[tile_idx], frame2[tile_idx])
        img[y, x] = overlap_ratio(frame1[tile_idx], frame2[tile_idx])
    plt.imshow(img)
    plt.colorbar(shrink=0.5)
    plt.savefig('results/%s.png' %iter_idx)
    plt.close()
    
    return img
    
for idx in range(1):
    print ('====== iter', idx)
    img = plot_tile(idx)

