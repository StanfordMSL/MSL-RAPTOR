#!/usr/bin/env python3
# system
import sys
import os
import glob
import pdb
# math
import math
import numpy as np
# magic
from string import digits

def load_mesh(mesh_path, is_save=False, is_normalized=False, is_flipped=False):
    with open(mesh_path, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for l in lines:
        l = l.strip()
        words = l.split(' ')
        if words[0] == 'v':
            vertices.append([float(words[1]), float(words[2]), float(words[3])])
        if words[0] == 'f':
            face_words = [x.split('/')[0] for x in words]
            faces.append([int(face_words[1])-1, int(face_words[2])-1, int(face_words[3])-1])


    vertices = np.array(vertices, dtype=np.float64)
    # flip mesh to unity rendering
    if is_flipped:
        vertices[:, 2] = -vertices[:, 2] 
    faces = np.array(faces, dtype=np.int32)
    
    if is_normalized:
        maxs = np.amax(vertices, axis=0)
        mins = np.amin(vertices, axis=0)
        diffs = maxs - mins
        assert diffs.shape[0] == 3
        vertices = vertices/np.linalg.norm(diffs)
    
    if is_save:
        np.savetxt(mesh_path.replace('.obj', '_vertices.txt'), X = vertices)

    return vertices, faces


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        mesh_path = "/mounted_folder/obj_models/real_train/"
        obs_paths = glob.glob(mesh_path + '*.obj')
        start_num = 2
        for i, mesh_path in enumerate(obs_paths):
            vertices, faces  = load_mesh(mesh_path)
            spans = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            name = mesh_path.split("/")[-1].split(".")[0]
            class_str = name.split('_')[0].rstrip(digits)
            # print("dims for {}: ".format(name, spans))
            print("---\nid: {}\nns: '{}'\nclass_str: '{}'".format(i + start_num, name, class_str))
            print("bound_box_l: {}\nbound_box_w: {}\nbound_box_h: {}".format(*spans))
            print("b_enforce_0: []")
        print("\n\nDONE!!!")
        
    except:
        import traceback
        traceback.print_exc()
