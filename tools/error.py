#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:48:05 2024

@author: boyuantian
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



l1_rgb_raw = torch.load("../l1_rgb.pt")
l1_depth_raw = torch.load("../l1_depth.pt")
alpha = torch.load("../alpha.pt")


error_map = np.ones((43, 75))

tile_error = []
for idx_row, row in enumerate(range(0, 680, 16)):
    for idx_col, col in enumerate(range(0, 1200, 16)):
        tile_color = l1_rgb_raw[0, row:row+16, col:col+16]
        tile_depth = l1_depth_raw[0, row:row+16, col:col+16]
        error = alpha * tile_color.mean() + (1 - alpha) * tile_depth.mean()
        # print ('row, col, tile error: ', row, col, alpha * tile_color.mean() + (1 - alpha) * tile_depth.mean())
        error_map[idx_row, idx_col] = error
        tile_error.append(float(error))

# tile_error_raw = tile_error
# tile_error = sorted(tile_error, reverse=True)
# threshold_idx = int(len(tile_error) * 0.1)
# threshold_val = tile_error[threshold_idx]

# indices = [idx for idx, val in enumerate(tile_error_raw) if val >= threshold_val]
# print (indices)

error_flat = error_map.flatten()
error_flat = sorted(error_flat, reverse=True)
threshold_idx = int(len(error_flat) * 0.01)
threshold_val = error_flat[threshold_idx]
mask = error_map <= threshold_val
error_map_filtered = np.where(mask, 0, error_map)
error_map_filtered = np.where(error_map_filtered>0, 1, 0)
XX = torch.tensor(error_map_filtered.flatten()).int()
plt.imshow(error_map_filtered)

test_render = l1_rgb_raw.cpu().detach().numpy().transpose(1, 2, 0)

# plt.imshow(test_render)

# cmap = plt.get_cmap("jet")
# error_map = cmap(error_map)
            
# plt.imshow(error_map)


# plt.imshow(error_map)
plt.colorbar(shrink=0.5)
# plt.savefig('results/%s.png' %iter_idx)
# plt.close()
    