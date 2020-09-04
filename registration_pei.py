# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:12 2019

@author: User
"""

from global_registration import *
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import statistics
import math
from pathlib import Path
np.seterr(divide='ignore', invalid='ignore')
from open3d import io,visualization,registration,utility,geometry

read_point_cloud = io.read_point_cloud
read_triangle_mesh = io.read_triangle_mesh
write_point_cloud = io.write_point_cloud
registration_icp = registration.registration_icp
compute_fpfh_feature = registration.compute_fpfh_feature
TransformationEstimationPointToPlane = registration.TransformationEstimationPointToPlane
TransformationEstimationPointToPoint = registration.TransformationEstimationPointToPoint
Vector3dVector = utility.Vector3dVector
draw_geometries = visualization.draw_geometries
write_triangle_mesh = io.write_triangle_mesh
paint_uniform_color = geometry.PointCloud.paint_uniform_color
voxel_down_sample = geometry.PointCloud.voxel_down_sample
estimate_normals = geometry.PointCloud.estimate_normals
KDTreeSearchParamHybrid = geometry.KDTreeSearchParamHybrid
PointCloud = geometry.PointCloud

def cal_distance(target,xyz_load):
    target_pts = np.asarray(target.points)
    target.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=5, max_nn=30))
    target_center = target.get_center() #center point of point cloud
    target.orient_normals_towards_camera_location(camera_location=np.asarray(target_center)) #make normals pointing to center point

    target_norm = np.asarray(target.normals)
    all_dist=[]
    min_ind_list = []
    for point in xyz_load:
        #find the closest triangle vertex to the point
        distance = np.sqrt(np.sum((target_pts-point)**2,axis=1))
        # Pei np.linalg.norm(vec_a-vec_b)
        min_distance = min(distance)

        #find the point in target point cloud that has the smallest distance with source point
        distance = list(distance)
        min_ind = distance.index(min_distance)
        min_ind_list.append(min_ind)
        targetpoint = target_pts[min_ind]
        #Pei min_ind=np.where(distance==np.min(distance))
        #define the positive or negative sign of the distance
        vec_source_target = point - targetpoint #get the vector from source point to target point
        tar_point_norm = target_norm[min_ind] #target point normal
        dot_product = np.dot(vec_source_target,tar_point_norm)
        sign = np.sign(dot_product)
        final_dist = min_distance*sign

        all_dist.append(final_dist)
    return all_dist,min_ind_list
def registration_once(sourcepcd,targetpcd,voxel_size):
    source = sourcepcd
    target = targetpcd
    print("source:",source)
    print("target:",target)
    #Pei move out the source operations
    print("registration step1")
    source.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=5, max_nn=30))
    source_center = source.get_center() #center point of point cloud
    source.orient_normals_towards_camera_location(camera_location=np.asarray(source_center)) #make normals pointing to center point
    target.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=5, max_nn=30))
    target_center = target.get_center() #center point of point cloud
    target.orient_normals_towards_camera_location(camera_location=np.asarray(target_center)) #make normals pointing to center point
    print("registration step1a")
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
    prepare_dataset(voxel_size,source,target)
    print(source_down, target_down, source_fpfh, target_fpfh)
    print("registration step1b")
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("registration step1c")
    result_icp = refine_registration(source, target,result_ransac,source_fpfh, target_fpfh, voxel_size)
    print("registration step1d")
    threshold = 0.25
    trans_init = result_icp.transformation #initial transformation matrix is from global registration
    #Apply point-to-plane ICP
    #Pei remove commement velow
    reg_p2p = registration_icp(source, target, threshold, trans_init,
                                           TransformationEstimationPointToPlane(),
                                           o3d.registration.ICPConvergenceCriteria(max_iteration = 100000))
    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, reg_p2p.transformation)
    return evaluation,reg_p2p,source
def registration_distance(sourcepcd,targetpcd):
    voxel_size = 1
    evaluation,reg_p2p,source = registration_once(sourcepcd,targetpcd,voxel_size)
    print("evaluation",evaluation)
    #if result is better than 0.8, then stop
    num_while=0  
    while evaluation.fitness < 0.8 and num_while<100: #<
        num_while += 1
        #draw_registration_result(source, targetpcd, reg_p2p.transformation)
        print("Registration fitness is too low in file "+str(sourcepcd))
        evaluation,reg_p2p,source = registration_once(sourcepcd,targetpcd,voxel_size)
        print("evaluation:",evaluation)
        print("registration step3")  
        #if evaluation.fitness>0.8: 
        #    break
    #draw_registration_result(source, target, reg_p2p.transformation)
    print("num of while loop:", num_while)
    #mat_tran=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    source.transform(reg_p2p.transformation)
    #source.transform(mat_tran)
    newsource_load = np.asarray(source.points)
    pcd = PointCloud()
    pcd.points = Vector3dVector(newsource_load)
    newsource = pcd
    xyz_load = np.asarray(newsource.points)
    target = targetpcd
    #all_dist = [] # all the distance for each point to the original mesh
    all_dist,min_ind_list = cal_distance(target,np.asarray(sourcepcd.points))
    return all_dist

def registration_distance_curv(sourcepcd,targetpcd):
    voxel_size = 1
    evaluation,reg_p2p,source = registration_once(sourcepcd,targetpcd,voxel_size)
    print("evaluation",evaluation)
    #if result is better than 0.8, then stop
    num_while=0  
    while evaluation.fitness < 0.8 and num_while<100: #<
        num_while += 1
        #draw_registration_result(source, targetpcd, reg_p2p.transformation)
        print("Registration fitness is too low in file "+str(sourcepcd))
        evaluation,reg_p2p,source = registration_once(sourcepcd,targetpcd,voxel_size)
        print("evaluation:",evaluation)
        print("registration step3")  
        #if evaluation.fitness>0.8: 
        #    break
    #draw_registration_result(source, target, reg_p2p.transformation)
    print("num of while loop:", num_while)
    #mat_tran=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    source.transform(reg_p2p.transformation)
    #source.transform(mat_tran)
    newsource_load = np.asarray(source.points)
    pcd = PointCloud()
    pcd.points = Vector3dVector(newsource_load)
    newsource = pcd
    xyz_load = np.asarray(newsource.points)
    target = targetpcd
    #all_dist = [] # all the distance for each point to the original mesh
    all_dist,min_ind_list = cal_distance(target,np.asarray(sourcepcd.points))
    return all_dist,min_ind_list


def cal_dist_features(def_pts, cor_dist, patchnum):
    data = []
    for a in cor_dist: #Pei for each point in defected object
        cor = a[:3] #coordinates of the corresponding point
        alldist = np.sum((def_pts-cor)**2,axis=1) #distance of this point to all the points in the point cloud
        ndx = alldist.argsort() #sort the distance in a decreasing way
        patch = [] #create an empty array with 100 values
        for x in range(int(str(patchnum))):
            patch.append(np.array(cor_dist[ndx[x]])) #assign the 30 closest points around the coresponding point to the patch
        patch = np.asarray(patch)
        distinpatch = patch[:,3]
        dist_median = np.asarray(np.median(distinpatch)) # calculate the median distance in the patch
        newa = np.concatenate((a,dist_median),axis = None)#combine the median distance data to the patch
        dist_mean = np.asarray(distinpatch.mean())
        newa = np.concatenate((newa,dist_mean),axis = None)#combine the mean distance data to the patch
        #Pei check again the maximum distance
        dist_abs = list(abs(np.asarray(distinpatch)))
        dist_abs_max = max(dist_abs)
        dist_abs_max_ind = dist_abs.index(dist_abs_max)
        dist_max = np.asarray(distinpatch[dist_abs_max_ind])
        newa = np.concatenate((newa,dist_max),axis = None)#combine the max distance data to the patch
        # Pei cal gradient, begin-------------------
        data.append(newa)
    return data

def cal_dist_curv_features(def_pts, cor_dist, patchnum, source_curv, target_curv,min_ind_list):
    data = []
    for i in range(len(cor_dist)): #Pei for each point in defected object
        a = cor_dist[i]
        cor = a[:3] #coordinates of the corresponding point
        alldist = np.sum((def_pts-cor)**2,axis=1) #distance of this point to all the points in the point cloud
        ndx = alldist.argsort() #sort the distance in a decreasing way
        patch = [] #create an empty array with 100 values
        for x in range(int(str(patchnum))):
            patch.append(np.array(cor_dist[ndx[x]])) #assign the 30 closest points around the coresponding point to the patch
        patch = np.asarray(patch)
        distinpatch = patch[:,3]
        dist_median = np.asarray(np.median(distinpatch)) # calculate the median distance in the patch
        newa = np.concatenate((a,dist_median),axis = None)#combine the median distance data to the patch
        dist_mean = np.asarray(distinpatch.mean())
        newa = np.concatenate((newa,dist_mean),axis = None)#combine the mean distance data to the patch
        #Pei check again the maximum distance
        dist_abs = list(abs(np.asarray(distinpatch)))
        dist_abs_max = max(dist_abs)
        dist_abs_max_ind = dist_abs.index(dist_abs_max)
        dist_max = np.asarray(distinpatch[dist_abs_max_ind])
        newa = np.concatenate((newa,dist_max),axis = None)#combine the max distance data to the patch
        diff_curv = np.asarray(target_curv[min_ind_list[i]]-source_curv[i])
        newa = np.concatenate((newa,diff_curv),axis = None)#combine the difference between curvatures to the patch
        newa = np.concatenate((newa,target_curv[min_ind_list[i]]),axis = None)
        newa = np.concatenate((newa,source_curv[i]),axis = None)
        data.append(newa)
    return data


# prepare the CSV format data, features
def datapreparing(defpcd,perfpcd,dist_p2pl_array,defect):
    # get the mean distance in each patch for defect1
    def_pts = np.asarray(defpcd.points)
    perf_pts = np.asarray(perfpcd.points)
    dist_p2pl_array1 = np.array(dist_p2pl_array)
    
    p2pldist = []
    # Pei how to better combine [a,b]+[c]=[a,b,c] ? row vector -> column vector
    for d in dist_p2pl_array1:
        p2pldist.append([d]) #to make the distance array applicable to the concatenate function
    cor_dist = np.concatenate((def_pts,p2pldist),1) #combine coordinates with corresponding point to plane distance

    patchnum = 20 # nn
    data = cal_dist_features(def_pts, cor_dist, patchnum)
    #define labels for each row with the non_defect information
    defect_list = list(defect) 
    
    datawithlabel = [] #add labels according to the distance to the dataset
    for d in range(len(data)):
        row = data[d]
        if d in defect_list:
            newrow = np.append(row,1)
        else:
            newrow = np.append(row,0)
        datawithlabel.append(newrow)
    datawithlabel = np.asarray(datawithlabel)
    
    return datawithlabel

def datapreparing_curv(defpcd,perfpcd,dist_p2pl_array,min_ind_list,source_curv,target_curv,defect_list,patchnum):
    # get the mean distance in each patch for defect1
    def_pts = np.asarray(defpcd.points)
    #def_pts = defpcd
    perf_pts = np.asarray(perfpcd.points)
    dist_p2pl_array1 = np.array(dist_p2pl_array)
    
    p2pldist = []
    # Pei how to better combine [a,b]+[c]=[a,b,c] ? row vector -> column vector
    for d in dist_p2pl_array1:
        p2pldist.append([d]) #to make the distance array applicable to the concatenate function
    cor_dist = np.concatenate((def_pts,p2pldist),1) #combine coordinates with corresponding point to plane distance

    patchnum = patchnum # nn
    data = cal_dist_curv_features(def_pts, cor_dist, patchnum,source_curv, target_curv,min_ind_list) 
    
    datawithlabel = [] #add labels according to the distance to the dataset
    for d in range(len(data)):
        row = data[d]
        if d in defect_list:
            newrow = np.append(row,1)
        else:
            newrow = np.append(row,0)
        datawithlabel.append(newrow)
    datawithlabel = np.asarray(datawithlabel)
    
    return datawithlabel

