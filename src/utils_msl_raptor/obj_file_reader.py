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
    
    """
    # if name == "mug_vignesh_norm":
    # D *= 0.95
    w *= .1
    # l *= 0.95
    hc = o + h/2
    h *= 0.5
    o = hc - h/2  # compensates for the change in size of h
    origin = np.array([(D + l)/2, H/2, D/2])  
    # The cup's origin is at the center of the axis-aligned 3D bouning box, with Y directed up and X directed in handle direction
    num_radial_points = 6
    da = 2*np.pi / num_radial_points
    pnt_offset = np.asarray([D/2,   0,      D/2   ])
    cup_verts = []
    connected_inds = []
    for i, ang in enumerate(np.linspace(0, 2*np.pi - da, num_radial_points)):
        R_deltay = np.array([[ np.cos(ang), 0.             , np.sin(ang) ],
                                [ 0.             , 1.             , 0               ],
                                [-np.sin(ang), 0.             , np.cos(ang) ]])
        rotated_point = R_deltay @ np.asarray([0,   0, D/2]) + pnt_offset
        cup_verts.append(list(rotated_point))
        cup_verts.append(list(rotated_point + np.array([0, H, 0])))
        if i == 0:
            connected_inds.append([2*i, 2*i+1])
        else:
            connected_inds.extend([[2*i, 2*i+1], [2*i, 2*i - 2], [2*i + 1, 2*i-1]])
    connected_inds.extend([ [0, 2*(num_radial_points-1)], [1, 2*(num_radial_points - 1) + 1]]) # connect first and last

    cup_verts.extend([[D,   o,   D/2 - w/2], [D,    o,   D/2 + w/2], [D + l,   o,   D/2 - w/2], 
                     [D + l,   o,   D/2 + w/2], [D, o + h, D/2 - w/2], [D,  o + h, D/2 + w/2], 
                     [D + l, o + h, D/2 - w/2], [D + l, o + h, D/2 + w/2]])
    cup_verts = np.asarray(cup_verts) - origin
   
                            
    # turn the cup verts from NOCS frame to MSL-RAPTOR frame (Z up)
    cup_verts = np.concatenate((cup_verts[:,0:1], cup_verts[:,2:3], cup_verts[:,1:2]), axis=1)

    handle_pnt0 = num_radial_points*2
    handle_conenctions = np.array([[0, 1], [2, 3], [4, 5],  [6, 7], \
                                    [0, 2], [1, 3], [4, 6],  [5, 7] , \
                                    [0, 4], [1, 5], [2, 6],  [3, 7] ]) + handle_pnt0


    connected_inds.extend(list(handle_conenctions))


    if name is not None:
        print("{} dims =\n{}".format(name, np.asarray(cup_verts)))
    return (cup_verts, connected_inds)
           

def mug_tapered_dims_to_verts(Dt, Db, H, lt, lb, w, ob1, ob2, ot, name=None):
    """
    This if for a tapered mug 
    Assumes top dims are bigger 
    """
    origin = np.array([(Dt + lt)/2, H/2, Dt/2])
    num_radial_points = 6
    da = 2*np.pi / num_radial_points
    pnt_offset = np.asarray([Dt/2,   0,      Dt/2   ])
    cup_verts = []
    connected_inds = []
    for i, ang in enumerate(np.linspace(0, 2*np.pi - da, num_radial_points)):
        R_deltay = np.array([[ np.cos(ang), 0.             , np.sin(ang) ],
                                [ 0.             , 1.             , 0               ],
                                [-np.sin(ang), 0.             , np.cos(ang) ]])
        rotated_point = R_deltay @ np.asarray([0,   0, Db/2]) + pnt_offset
        cup_verts.append(list(rotated_point))
        rotated_point = R_deltay @ np.asarray([0,   H, Dt/2]) + pnt_offset
        cup_verts.append(list(rotated_point))
        if i == 0:
            connected_inds.append([2*i, 2*i+1])
        else:
            connected_inds.extend([[2*i, 2*i+1], [2*i, 2*i - 2], [2*i + 1, 2*i-1]])
    connected_inds.extend([ [0, 2*(num_radial_points-1)], [1, 2*(num_radial_points - 1) + 1]]) # connect first and last

    cup_verts.extend([[Dt/2 + Db/2, ob1, Dt/2 - w/2], [Dt/2 + Db/2, ob1, Dt/2 + w/2],\
                     [Dt/2 + Db/2 + lb, ob1 + ob2, Dt/2 - w/2], [Dt/2 + Db/2 + lb, ob1 + ob2, Dt/2 + w/2],
                     [Dt + lt, H - ot, Dt/2 - w/2], [Dt + lt, H - ot, Dt/2 + w/2],\
                     [Dt, H - ot, Dt/2 - w/2], [Dt, H - ot, Dt/2 + w/2]])
    cup_verts = np.asarray(cup_verts) - origin
   
                            
    # turn the cup verts from NOCS frame to MSL-RAPTOR frame (Z up)
    cup_verts = np.concatenate((cup_verts[:,0:1], cup_verts[:,2:3], cup_verts[:,1:2]), axis=1)

    handle_pnt0 = num_radial_points*2
    handle_conenctions = np.array([[0, 1], [2, 3], [4, 5],  [6, 7], \
                                   [0, 2], [1, 3], [2, 4],  [3, 5], \
                                   [4, 6], [5, 7] ]) + handle_pnt0

    connected_inds.extend(list(handle_conenctions))


    # # The cup's origin is at the center of the axis-aligned 3D bouning box, with Y directed up and X directed in handle direction
    # cup_verts = np.asarray([[Dt/2 - Db/2, 0, Dt/2 - Db/2], [Dt/2 - Db/2, 0, Dt/2 + Db/2], [Dt/2 + Db/2, 0, Dt/2 - Db/2], [Dt/2 + Db/2, 0, Dt/2 + Db/2],\
    #                         [0, H, 0], [0, H, Dt], [Dt, H, 0], [Dt, H, Dt],\
    #                         [Dt/2 + Db/2, ob1, Dt/2 - w/2], [Dt/2 + Db/2, ob1, Dt/2 + w/2],\
    #                         [Dt/2 + Db/2 + lb, ob1 + ob2, Dt/2 - w/2], [Dt/2 + Db/2 + lb, ob1 + ob2, Dt/2 + w/2],
    #                         [Dt + lt, H - ot, Dt/2 - w/2], [Dt + lt, H - ot, Dt/2 + w/2],\
    #                         [Dt, H - ot, Dt/2 - w/2], [Dt, H - ot, Dt/2 + w/2]]) - origin

    # # turn the cup verts from NOCS frame to MSL-RAPTOR frame (Z up)
    # cup_verts = np.concatenate((cup_verts[:,0:1], cup_verts[:,2:3], cup_verts[:,1:2]), axis=1)
                            
    # connected_inds = [[0, 1], [0, 2], [1, 3],  [2, 3], \
    #                   [4, 5], [4, 6], [5, 7],  [6, 7], \
    #                   [0, 4], [1, 5], [2, 6],  [3, 7], \
    #                   [8, 9], [10, 11], [12, 13],  [6, 15], \
    #                   [8, 10], [9, 11], [10, 12],  [11, 13], \
    #                   [12, 14], [13, 15] ]


    if name is not None:
        print("{} dims =\n{}".format(name, np.asarray(cup_verts)))
    return (cup_verts, connected_inds)


def bowl_dims_to_verts(Dt, Dm, Db, Ht, Hb, name=None):
    """
    This if for a tapered mug 
    Assumes top dims are bigger 
    """
    # Dt *= 0.95
    # Dm *= 0.95
    # Db *= 1.1
    # if name == "bowl_brown_ikea_norm":
    #     Ht *= 0.5
    origin = np.array([Dt/2, (Ht + Hb)/2, Dt/2])

    num_radial_points = 20
    da = 2*np.pi / num_radial_points
    pnt_offset = np.asarray([Dt/2,   0,      Dt/2   ])
    bowl_verts = []
    connected_inds = []
    for i, ang in enumerate(np.linspace(0, 2*np.pi - da, num_radial_points)):
        R_deltay = np.array([[ np.cos(ang), 0.             , np.sin(ang) ],
                             [ 0.         , 1.             , 0           ],
                             [-np.sin(ang), 0.             , np.cos(ang) ]])
        rotated_point = R_deltay @ np.asarray([0,   Hb + Ht, Dt/2]) + pnt_offset
        bowl_verts.append(list(rotated_point))
        rotated_point = R_deltay @ np.asarray([0,   Hb, Dm/2]) + pnt_offset
        bowl_verts.append(list(rotated_point))
        rotated_point = R_deltay @ np.asarray([0,   0, Db/2]) + pnt_offset
        bowl_verts.append(list(rotated_point))
        if i == 0:
            connected_inds.extend([[3*i, 3*i+1], [3*i+1, 3*i+2]])
        else:
            connected_inds.extend([[3*i, 3*i+1], [3*i+1, 3*i+2], [3*i, 3*(i-1)], [3*i + 1, 3*(i-1)+1], [3*i + 2, 3*(i-1) + 2]])
    connected_inds.extend([ [0, 3*(num_radial_points-1)], [1, 3*(num_radial_points - 1) + 1], [2, 3*(num_radial_points - 1) + 2]]) # connect first and last

    bowl_verts = np.asarray(bowl_verts) - origin

    # turn the cup verts from NOCS frame to MSL-RAPTOR frame (Z up)
    bowl_verts = np.concatenate((bowl_verts[:,0:1], bowl_verts[:,2:3], bowl_verts[:,1:2]), axis=1)


    if name is not None:
        print("{} dims =\n{}".format(name, np.asarray(bowl_verts)))
    return (bowl_verts, connected_inds)


def laptop_dims_to_verts(W, lb, hb, lt, ht, angr, name=None):
    """
    This if for a laptop with its lid opened to a fixed angle
    angr is in radians
    """
    lb *= 0.90
    lt *= 0.95


    a = lt * np.sin(angr)
    b = lt * np.cos(angr)
    aa = ht * np.sin(np.pi/2 - angr)
    bb = ht * np.cos(np.pi/2 - angr)
    origin = np.array([(lb + b)/2, ( a  + ht* np.sin(np.pi/2 - angr) )/2, W/2])  
    # The cup's origin is at the center of the axis-aligned 3D bouning box, with Y directed up and X directed in handle direction
    laptop_verts = np.asarray([[b, 0, 0 ], [b + lb,  0,  0 ], [b,  0, W  ], [ b + lb, 0, W ], \
                               [b + bb, hb,  0 ], [b + lb,  hb,  0 ], [b + bb,  hb, W  ], [ b + lb, hb, W ], \
                               [0, a, 0], [0, a, W], [bb, a + aa, 0], [bb, a + aa, W]]) - origin
                            
    # turn the cup verts from NOCS frame to MSL-RAPTOR frame (Z up)
    laptop_verts = np.concatenate((laptop_verts[:,0:1], laptop_verts[:,2:3], laptop_verts[:,1:2]), axis=1)
    
    connected_inds = [[0, 1], [0, 2], [1, 3],  [2, 3], \
                      [4, 5], [4, 6], [5, 7],  [6, 7], \
                      [0, 8], [2, 9], [4, 10],  [6, 11], \
                      [8, 9], [9, 11], [11, 10],  [8, 10], \
                      [0, 4], [2, 6], [1, 5],  [3, 7] ]

    if name is not None:
        print("{} dims =\n{}".format(name, np.asarray(laptop_verts)))
    return (laptop_verts, connected_inds)


def plot_object_verts(verts, connected_inds=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(1000*verts[:,0], 1000*verts[:,1], 1000*verts[:,2], 'b.')
    verts_mm = 1000*verts
    ax.scatter(verts_mm[:,0], verts_mm[:,1], verts_mm[:,2], 'b.')
    if connected_inds is not None:
        for pair in connected_inds:
            pnt1 = verts_mm[pair[0]]
            pnt2 = verts_mm[pair[1]]
            ax.plot(xs=[pnt1[0], pnt2[0]], ys=[pnt1[1], pnt2[1]], zs=[pnt1[2], pnt2[2]], color='b', linestyle='-')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.show(block=False)

def save_objs_verts_as_txt(verts, name, path, connected_inds=None):
    np.savetxt(path + name, X=verts)
    if connected_inds is not None:
        np.savetxt(path + name + "_joined_inds", X=connected_inds)


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
            b_save = True
            b_plot = False
            objs = {}

            objs["mug_anastasia_norm"]       = mug_dims_to_verts(D=0.09140, H=0.09173, l=0.03210, h=0.05816, w=0.01353, o=0.02460, name="mug_anastasia_norm")
            objs["mug_brown_starbucks_norm"] = mug_dims_to_verts(D=0.08599, H=0.10509, l=0.02830, h=0.07339, w=0.01394, o=0.01649, name="mug_brown_starbucks_norm")
            objs["mug_daniel_norm"]          = mug_dims_to_verts(D=0.07354, H=0.05665, l=0.03313, h=0.05665, w=0.01089, o=0.02797, name="mug_daniel_norm")
            objs["mug_vignesh_norm"]         = mug_dims_to_verts(D=0.08126, H=0.10097, l=0.03192, h=0.06865, w=0.01823, o=0.01752, name="mug_vignesh_norm")
            objs["mug_white_green_norm"]     = mug_dims_to_verts(D=0.10265, H=0.08295, l=0.03731, h=0.05508, w=0.01917, o=0.02352, name="mug_white_green_norm")
            objs["mug2_scene3_norm"]         = mug_tapered_dims_to_verts(Dt=0.11442, Db=0.0687, H=0.08295, lt=0.02803, lb=0.0390, w=0.0165, ob1=0.01728, ob2=0.02403, ot=0.00954, name="mug2_scene3_norm")

            objs["laptop_air_xin_norm"]   = laptop_dims_to_verts(W=0.27497, lb=0.20273, hb=0.01275, lt=0.19536, ht=0.01073, angr=0.987935358216449, name="laptop_air_xin_norm")
            objs["laptop_alienware_norm"] = laptop_dims_to_verts(W=0.33020, lb=0.25560, hb=0.02397, lt=0.28086, ht=0.02253, angr=0.879851927765118, name="laptop_alienware_norm")
            objs["laptop_mac_pro_norm"]   = laptop_dims_to_verts(W=0.31531, lb=0.23383, hb=0.01076, lt=0.26085, ht=0.01022, angr=0.734357435546022, name="laptop_mac_pro_norm")
            objs["laptop_air_0_norm"]     = laptop_dims_to_verts(W=0.32962, lb=0.22963, hb=0.01492, lt=0.22134, ht=0.01038, angr=(np.pi-2.510641396715293), name="laptop_air_0_norm")
            objs["laptop_air_1_norm"]     = laptop_dims_to_verts(W=0.32710, lb=0.23781, hb=0.01500, lt=0.22038, ht=0.01000, angr=(np.pi-2.062630110006899), name="laptop_air_1_norm")
            objs["laptop_dell_norm"]      = laptop_dims_to_verts(W=0.30858, lb=0.19788, hb=0.01493, lt=0.18519, ht=0.01200, angr=(np.pi-2.229134520647158), name="laptop_dell_norm")

            objs["bowl_blue_ikea_norm"]          = bowl_dims_to_verts(Dt=0.16539, Dm=0.13123, Db=0.04040, Ht=0.03964, Hb=0.03821, name="bowl_blue_ikea_norm")
            objs["bowl_brown_ikea_norm"]         = bowl_dims_to_verts(Dt=0.16452, Dm=0.12303, Db=0.06431, Ht=0.035, Hb=0.023, name="bowl_brown_ikea_norm")
            # objs["bowl_brown_ikea_norm"]         = bowl_dims_to_verts(Dt=0.16452, Dm=0.12303, Db=0.06431, Ht=0.04507, Hb=0.02916, name="bowl_brown_ikea_norm")
            objs["bowl_chinese_blue_norm"]       = bowl_dims_to_verts(Dt=0.17023, Dm=0.13963, Db=0.07944, Ht=0.05247, Hb=0.03653, name="bowl_chinese_blue_norm")
            objs["bowl_blue_white_chinese_norm"] = bowl_dims_to_verts(Dt=0.15672, Dm=0.12155, Db=0.05886, Ht=0.04064, Hb=0.02452, name="bowl_blue_white_chinese_norm")
            objs["bowl_shengjun_norm"]           = bowl_dims_to_verts(Dt=0.14231, Dm=0.13025, Db=0.06516, Ht=0.05296, Hb=0.02353, name="bowl_shengjun_norm")
            objs["bowl_white_small_norm"]        = bowl_dims_to_verts(Dt=0.14231, Dm=0.12155, Db=0.05886, Ht=0.04064, Hb=0.02452, name="bowl_white_small_norm")
            if b_save:
                # save_path = '/mounted_folder/generated_vertices_for_raptor/'
                save_path = '/root/msl_raptor_ws/src/msl_raptor/params/generated_vertices_for_raptor/'
                if not os.path.exists( save_path ):
                    os.makedirs( save_path )

                for key in objs:
                    save_objs_verts_as_txt(verts=objs[key][0], name=key, path=save_path, connected_inds=objs[key][1])
            
            if b_plot:
                # plot_object_verts(objs["mug_anastasia_norm"][0], connected_inds=objs["mug_anastasia_norm"][1])
                # plot_object_verts(objs["mug_brown_starbucks_norm"][0], connected_inds=objs["mug_brown_starbucks_norm"][1])
                # plot_object_verts(objs["mug_daniel_norm"][0], connected_inds=objs["mug_daniel_norm"][1])
                plot_object_verts(objs["mug2_scene3_norm"][0], connected_inds=objs["mug2_scene3_norm"][1])
                # plot_object_verts(objs["laptop_air_xin_norm"][0], connected_inds=objs["laptop_air_xin_norm"][1])
                # plot_object_verts(objs["laptop_alienware_norm"][0], connected_inds=objs["laptop_alienware_norm"][1])
                # plot_object_verts(objs["laptop_mac_pro_norm"][0], connected_inds=objs["laptop_mac_pro_norm"][1])
                # plot_object_verts(objs["laptop_air_0_norm"][0], connected_inds=objs["laptop_air_0_norm"][1])
                # plot_object_verts(objs["laptop_air_1_norm"][0], connected_inds=objs["laptop_air_1_norm"][1])
                # plot_object_verts(objs["laptop_dell_norm"][0], connected_inds=objs["laptop_dell_norm"][1])
                # plot_object_verts(objs["bowl_blue_ikea_norm"][0], connected_inds=objs["bowl_blue_ikea_norm"][1])
                # plot_object_verts(objs["bowl_brown_ikea_norm"][0], connected_inds=objs["bowl_brown_ikea_norm"][1])
                # plot_object_verts(objs["bowl_chinese_blue_norm"][0], connected_inds=objs["bowl_chinese_blue_norm"][1])
                # plot_object_verts(objs["bowl_blue_white_chinese_norm"][0], connected_inds=objs["bowl_blue_white_chinese_norm"][1])
                # plot_object_verts(objs["bowl_shengjun_norm"][0], connected_inds=objs["bowl_shengjun_norm"][1])
                # plot_object_verts(objs["bowl_white_small_norm"][0], connected_inds=objs["bowl_white_small_norm"][1])
                plt.show()
        print("\n\nDONE!!!")
        
    except:
        import traceback
        traceback.print_exc()
