#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:01:18 2018

@author: blue
"""

import imageio
import os
import numpy as np

def find_joints_box(im,perc_w =0.1, perc_h=0.1):
    shapy = im.shape
    left = []
    right = []
    top = []
    bottom = []

    for i in range(shapy[0]):
        for j in range(shapy[1]):
            if im[i,j] != 0:
                left.append(j)
                break
        for j in range(shapy[1]-1,-1,-1):
            if im[i,j] != 0:
                right.append(j)
                break

    for j in range(shapy[1]):
        for i in range(shapy[0]):
            if im[i,j] != 0:
                top.append(i)
                break
        for i in range(shapy[0]-1,-1,-1):
            if im[i,j] != 0:
                bottom.append(i)
                break
    
    if len(bottom) != 0:
        f_left = np.min(left)
        f_right = np.max(right)
        f_top = np.min(top)
        f_bottom = np.max(bottom)

        width = np.abs(f_right -f_left)
        height = np.abs(f_bottom -f_top)

        x = int(np.floor(f_left + width/2))
        y = int(np.floor(f_top +height/2))
        
    else:
        x = -1
        y = -1
        width,height = 0,0

    return x,y, max(width, height)

def joints(image, n_joints):
    coords = np.zeros((n_joints,2))
    for i in range(1, n_joints+1):
        n_im = np.where(image==i,1,0)
        x,y,size = find_joints_box(n_im)
        coords[i-1] = x,y
    
    return coords, size

def rad_original_file(which="train"):
    if which=="train":
        file ='./list/train_cluster_list.txt'
    elif which=="eval":
        file = "./list/eval_cluster_list.txt"
    else:
        file ="./list/test_cluster_list.txt"
        
    data_dict = {}
    train_table = []
    input_file = open(file, 'r')
    print('READING TRAIN DATA')
    for line in input_file:
        line = line.strip()
        line = line.split(' ')
        name = line[0]
        gt_name = line[1]

        data_dict[name] = {'gt_name' : gt_name}
        train_table.append(name)
            #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
    input_file.close()
    return data_dict, train_table

def define_joints_file(which="train"):
    d_dict, table = rad_original_file(which)
    if which=="train":
        first_path = "./train/"
        file_name = "./train/train_head.txt"
        
    elif which=="eval":
        first_path = "./eval/"
        file_name = "./eval/eval_head.txt"
    else:
        first_path = "./test/"
        file_name = "./test/test_head.txt"
    
    with open(file_name, 'w') as f:
        count = 0
        for i in table:
            gt_name = d_dict[i]['gt_name']
            namy = os.path.join(first_path, gt_name)
            im = imageio.imread(namy)
            coord,s = joints(im, 24)
            d_dict[i]['joints'] = coord
            str1 = ' '.join(str(e) for e in list(coord.reshape(-1)))
            f.write(i + " " + str1 + "\n")
            count += 1
            if count % 2500 == 0: 
                print(count)
                print(d_dict[i])
        f.close()
		
		
if __name__ == '__main__':
	define_joints_file()
	define_joints_file("eval")
	define_joints_file("test")