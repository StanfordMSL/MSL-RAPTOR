#!/usr/bin/env python3
# system
import sys
import os
import glob
import pdb
# math
import math
import numpy as np
# plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def mug_dims_to_verts(D, H, l, w, h, o, name=None):
    """
    This if for a standard, non-tapered mug
    The cup's origin is at the center of the axis-aligned 3D bouning box, with Y directed up and X directed in handle direction
    """
    origin = np.array([(D + l)/2, H/2, D/2])
    # verts assuming origin is is at bottom corner of 3D bb s.t. everything is positive
    cup_verts = np.asarray([[0,   0,       0    ], [D,    0,       0    ], [  0,     0,       D    ], [ D,      0,       D    ],\
                            [0,   H,       0    ], [D,    H,       0    ], [  0,     H,       D    ], [ D,      H,       D    ],\
                            [D,   o,   D/2 - w/2], [D,    o,   D/2 + w/2], [D + l,   o,   D/2 - w/2], [D + l,   o,   D/2 + w/2],\
                            [D, o + h, D/2 - w/2], [D,  o + h, D/2 + w/2], [D + l, o + h, D/2 - w/2], [D + l, o + h, D/2 + w/2]]) - origin
    # cup_verts = np.asarray([[-D/2, 0,  D/2], [-D/2, 0,  -D/2], [D/2,      0,  D/2], [D/2,      0,  -D/2],\
    #                         [-D/2, H,  D/2], [-D/2, H,  -D/2], [D/2,      H,  D/2], [D/2,      H,  -D/2],\
    #                         [D/2,  o,  w/2], [D/2,  o,  -w/2], [D/2 + l,  o,  w/2], [D/2 + l,  o,  -w/2],\
    #                         [D/2, o+h, w/2], [D/2, o+h, -w/2], [D/2 + l, o+h, w/2], [D/2 + l, o+h, -w/2]])

    if name is not None:
        print("{} dims =\n{}".format(name, np.asarray(cup_verts)))
    return cup_verts
           

def mug_tapered_dims_to_verts(Dt, Db, H, lt, lb, w, ob1, ob2, ot, name=None):
    """
    This if for a tapered mug 
    The cup's origin is at the center of the axis-aligned 3D bouning box, with Y directed up and X directed in handle direction
    Assumes top dims are bigger 
    """
    origin = np.array([(Dt + lt)/2, H/2, Dt/2])
    # verts assuming origin is is at bottom corner of 3D bb s.t. everything is positive
    cup_verts = np.asarray([[Dt/2 - Db/2, 0, Dt/2 - Db/2], [Dt/2 - Db/2, 0, Dt/2 + Db/2], [Dt/2 + Db/2, 0, Dt/2 - Db/2], [Dt/2 + Db/2, 0, Dt/2 + Db/2],\
                            [0, H, 0], [Dt, H, 0], [0, H, Dt], [Dt, H, Dt],\
                            [Dt/2 + Db/2, ob1, Dt/2 - w/2], [Dt/2 + Db/2, ob1, Dt/2 + w/2],\
                            [Dt/2 + Db/2 + lb, ob1 + ob2, Dt/2 - w/2], [Dt/2 + Db/2 + lb, ob1 + ob2, Dt/2 + w/2],
                            [Dt + lt, H - ot, Dt/2 - w/2], [Dt + lt, H - ot, Dt/2 + w/2],\
                            [Dt, H - ot, Dt/2 - w/2], [Dt, H - ot, Dt/2 + w/2]]) - origin

    if name is not None:
        print("{} dims =\n{}".format(name, np.asarray(cup_verts)))
    return cup_verts


def plot_object_verts(verts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(1000*verts[:,0], 1000*verts[:,1], 1000*verts[:,2], 'b.')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.show(block=False)

def save_objs_verts_as_txt(verts, name, path):
    np.savetxt(path + name, X=verts)
    # np.savetxt(sp, s, fmt='%g') 


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        if False:
            np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
            mesh_path = "/Users/benjamin/Documents/Cours/Stanford/msl/pose_estimation/nocs_dataset/obj_models/real_train/"
            obs_paths = glob.glob(mesh_path + '*.obj')
            start_num = 2
            for i, mesh_path in enumerate(obs_paths):
                vertices, faces  = load_mesh(mesh_path)
                spans = np.max(vertices, axis=0) - np.min(vertices, axis=0)
                name = mesh_path.split("/")[-1].split(".")[0]
                class_str = name.split('_')[0].rstrip(digits)
                # print("dims for {}: ".format(name, spans))
                print("---\nid: {}\nns: '{}'\nclass_str: '{}'".format(i + start_num, name, class_str))
                print("bound_box_l: {}\nbound_box_h: {}\nbound_box_w: {}".format(*spans))
                print("b_enforce_0: []")
        else:
            objs = {}
            objs["mug_anastasia_norm"]       = mug_dims_to_verts(D=0.09140, H=0.09173, l=0.03210, h=0.05816, w=0.01353, o=0.02460, name="mug_anastasia_norm")
            objs["mug_brown_starbucks_norm"] = mug_dims_to_verts(D=0.08599, H=0.10509, l=0.02830, h=0.07339, w=0.01394, o=0.01649, name="mug_brown_starbucks_norm")
            objs["mug_daniel_norm"]          = mug_dims_to_verts(D=0.07354, H=0.05665, l=0.03313, h=0.05665, w=0.01089, o=0.02797, name="mug_daniel_norm")
            objs["mug_vignesh_norm"]         = mug_dims_to_verts(D=0.08126, H=0.10097, l=0.03192, h=0.06865, w=0.01823, o=0.01752, name="mug_vignesh_norm")
            objs["mug_white_green_norm"]     = mug_dims_to_verts(D=0.10265, H=0.08295, l=0.03731, h=0.05508, w=0.01917, o=0.02352, name="mug_white_green_norm")
            objs["mug2_scene3_norm"]         = mug_tapered_dims_to_verts(Dt=0.11442, Db=0.0687, H=0.08295, lt=0.02803, lb=0.0390, w=0.015, ob1=0.01728, ob2=0.02403, ot=0.00954, name="mug2_scene3_norm")
            print("WARNING!!!! MADE UP VALUE FOR WIDTH OF TAPERED MUG HANDLE (mug2_scene3_norm)")

            save_path = '/mounted_folder/generated_vertices_for_raptor/'
            if not os.path.exists( save_path ):
                os.makedirs( save_path )

            for key in objs:
                save_objs_verts_as_txt(verts=objs[key], name=key, path=save_path)

            plot_object_verts(objs["mug2_scene3_norm"])
            plt.show()
        print("\n\nDONE!!!")
        
    except:
        import traceback
        traceback.print_exc()
