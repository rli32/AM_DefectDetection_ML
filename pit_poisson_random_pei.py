# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:47:24 2019

@author: User
"""
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
from open3d import *
import trimesh
from open3d import io,visualization,registration,utility,geometry

read_point_cloud = io.read_point_cloud
read_triangle_mesh = io.read_triangle_mesh
write_point_cloud = io.write_point_cloud
registration_icp = registration.registration_icp
compute_fpfh_feature = registration.compute_fpfh_feature
TransformationEstimationPointToPlane = registration.TransformationEstimationPointToPlane
TransformationEstimationPointToPoint = registration.TransformationEstimationPointToPoint
Vector3dVector = utility.Vector3dVector
Vector3iVector = utility.Vector3iVector
draw_geometries = visualization.draw_geometries
write_triangle_mesh = io.write_triangle_mesh
paint_uniform_color = geometry.PointCloud.paint_uniform_color
voxel_down_sample = geometry.PointCloud.voxel_down_sample
estimate_normals = geometry.PointCloud.estimate_normals
KDTreeSearchParamHybrid = geometry.KDTreeSearchParamHybrid
PointCloud = geometry.PointCloud
TriangleMesh = geometry.TriangleMesh
sample_points_poisson_disk = geometry.TriangleMesh.sample_points_poisson_disk
create_sphere = geometry.TriangleMesh.create_sphere
sample_points_uniformly = geometry.TriangleMesh.sample_points_uniformly

def cal_area_triangles(vertex,triangles):
    areasum1 = 0
    for triangle in triangles:
        #coordinates of the three points on the triangle
        A = vertex[triangle[0]]
        B = vertex[triangle[1]]
        C = vertex[triangle[2]]
        AB, AC = A-B, A-C
        #area of the triangle
        area = np.linalg.norm(np.cross(AB,AC))/2.0
        areasum1 = areasum1 + area
    return areasum1
def tran_matrix(center_i):
    c_x = (center_i)[0]
    c_y = (center_i)[1]
    c_z = (center_i)[2]
    #create mesh sphere
    my_transform = np.asarray([[1.0, 0.0, 0.0, c_x],
                                [0.0, 1.0, 0.0, c_y],
                                [0.0, 0.0, 1.0, c_z],
                                [0.0, 0.0, 0.0, 1.0]])
    return my_transform
def try_union_once(polymesh_name,center_choice,r1,r2,num,case_num):
    center = [None] * num
    r = [None] * num
    for i in range(num):
        r[i] = random.uniform(r1,r2)
        if i<1:
            center[i] = random.choice(center_choice)
        else:
            remove_index = []
            for j in range(len(center_choice)):
                p = center_choice[j]
                if np.linalg.norm(np.asarray(p)-np.asarray(center[i-1])) <= r[i]:
                    remove_index.append(j)

            center_choice = [i for j, i in enumerate(center_choice) if j not in remove_index]
            center[i] = random.choice(center_choice)
        my_transform=tran_matrix(center[i])
        # define the coordinates of the center and radius of the sphere
        mesh_sphere = create_sphere(radius = r[i])
        mesh_sphere.transform(my_transform)
        write_triangle_mesh('mesh_sphere'+str(case_num)+'.ply',mesh_sphere)

        #find the intersection surface between the mesh_sphere and mesh with library trimesh
        mesh_sphere1 = trimesh.load_mesh('mesh_sphere'+str(case_num)+'.ply','ply')
        os.remove('mesh_sphere'+str(case_num)+'.ply')

        if i<1:
            mesh_polyhedron = trimesh.load_mesh(polymesh_name)
            union = trimesh.boolean.union([mesh_sphere1,mesh_polyhedron],engine = 'scad')
            union.export("union"+str(case_num)+".ply","ply")
        else:
            union_previous = trimesh.load_mesh("union"+str(case_num)+".ply","ply")
            union = trimesh.boolean.union([mesh_sphere1,union_previous],engine = 'scad')
            union.export("union"+str(case_num)+".ply","ply")

        union_mesh = read_triangle_mesh("union"+str(case_num)+".ply")

        #decide the points number in the point cloud depends on the area size
    triangles = np.asarray(union_mesh.triangles)
    vertex = np.asarray(union_mesh.vertices)
    areasum1 = cal_area_triangles(vertex,triangles)
    return areasum1,center,r
def try_union_once_curv(polymesh_name,center_choice,r1,r2,num,case_num):
    center = [None] * num
    r = [None] * num
    for i in range(num):
        r[i] = random.uniform(r1,r2)
        if i<1:
            center[i] = random.choice(center_choice)
        else:
            remove_index = []
            for j in range(len(center_choice)):
                p = center_choice[j]
                if np.linalg.norm(np.asarray(p)-np.asarray(center[i-1])) <= r[i]:
                    remove_index.append(j)

            center_choice = [i for j, i in enumerate(center_choice) if j not in remove_index]
            center[i] = random.choice(center_choice)
        my_transform=tran_matrix(center[i])
        # define the coordinates of the center and radius of the sphere
        mesh_sphere = create_sphere(radius = r[i])
        mesh_sphere.transform(my_transform)
        write_triangle_mesh('mesh_sphere'+str(case_num)+'.ply',mesh_sphere)

        #find the intersection surface between the mesh_sphere and mesh with library trimesh
        mesh_sphere1 = trimesh.load_mesh('mesh_sphere'+str(case_num)+'.ply','ply')
        os.remove('mesh_sphere'+str(case_num)+'.ply')

        if i<1:
            mesh_polyhedron = trimesh.load_mesh(polymesh_name)
            union = trimesh.boolean.union([mesh_sphere1,mesh_polyhedron],engine = 'scad')
            union.export("union"+str(case_num)+".ply","ply")
        else:
            union_previous = trimesh.load_mesh("union"+str(case_num)+".ply","ply")
            union = trimesh.boolean.union([mesh_sphere1,union_previous],engine = 'scad')
            union.export("union"+str(case_num)+".ply","ply")
        #print("1")
        union_mesh = read_triangle_mesh("union"+str(case_num)+".ply")
        #print("2")

        #decide the points number in the point cloud depends on the area size)

    triangles = np.asarray(union_mesh.triangles)
    vertex = np.asarray(union_mesh.vertices)
    print("3")
    areasum1 = cal_area_triangles(vertex,triangles)


    return areasum1,center,r,union
def get_union_done(polymesh_name,center_choice,r1,r2,num,case_num):
    print("generate defect, step1")
    areasum1,center,r= try_union_once(polymesh_name,center_choice,r1,r2,num,case_num)
    print("area sum:",areasum1)
    polymesh = read_triangle_mesh(polymesh_name)
    vertex = np.asarray(polymesh.vertices)
    triangles = np.asarray(polymesh.triangles)
    polymesh_area = cal_area_triangles(vertex,triangles)
    print("polymesh_area:", polymesh_area)
    while areasum1<polymesh_area*0.8: #when points in a defect point cloud is fewer than 3500, the main object is missing
        try:
            areasum1,center,r=try_union_once(polymesh_name,center_choice,r1,r2,num,case_num)
        except:
            print("Areasum Exception happens in file poly_pit"+str(case_num))
    return areasum1,center,r
def get_union_done_curv(polymesh_name,center_choice,r1,r2,num,case_num):
    print("generate defect, step1")
    areasum1,center,r,union= try_union_once_curv(polymesh_name,center_choice,r1,r2,num,case_num)
    print("area sum:",areasum1)
    polymesh = read_triangle_mesh(polymesh_name)
    vertex = np.asarray(polymesh.vertices)
    triangles = np.asarray(polymesh.triangles)
    polymesh_area = cal_area_triangles(vertex,triangles)
    print("polymesh_area:", polymesh_area)
    while areasum1<polymesh_area*0.8: #when points in a defect point cloud is fewer than 3500, the main object is missing
        try:
            areasum1,center,r,union=try_union_once_curv(polymesh_name,center_choice,r1,r2,num,case_num)
        except:
            print("Areasum Exception happens in file poly_pit"+str(case_num))
    return areasum1,center,r,union

def get_defect_index(union_pts,num,center,r):
    #defect = [] #non-defect points set
    defect_index = [] #non-defect points index set
    total_defect = [None]*num
    for ind in range(len(union_pts)):
        n = np.asarray(union_pts[ind])
        for m in range(num):
            #defect_index = [] #non-defect points index set
            d_n_cm = np.linalg.norm(n-np.asarray(center[m]))
            if d_n_cm<=r[m]:
                #defect.append(n)
                defect_index.append(ind)
            total_defect[m] = defect_index
    return total_defect
def add_random_noise(union_pts):
    pcd3 = PointCloud() #poisson sampling defect point cloud
    pcd3.points = Vector3dVector(union_pts)

    # Add random noise from normal distribution to the point cloud
    # compute normals of the the whole points
    estimate_normals(pcd3,
        search_param=KDTreeSearchParamHybrid(radius=1, max_nn=30))
    normals_whole = np.asarray(pcd3.normals)
    print("step3")
    # transform point to another place with normal vector
    points = pcd3.points
    random_points = [0]*len(points)
    for j in range(len(points)):
        a = np.random.normal(0,0.05,10000)
        b = random.choice(a)
        random_points[j] = points[j] + b*normals_whole[j]
    return random_points
#This function will randomely generate a polyhedron point cloud with 8 vertices
# and a point cloud with pit defect on it 
def generate_defect(polymesh_name, polypcd,path_sample_pcd_save, pcd_pts,r1,r2,num,case_num):
    ## Start to make sphere defect
    polymesh = read_triangle_mesh(polymesh_name)
    center_choice = np.asarray(polypcd.points).tolist()
    print(len(center_choice))
    areasum1,center,r=get_union_done(polymesh_name,center_choice,r1,r2,num,case_num)
    print("get union done!")
    union_mesh = read_triangle_mesh("union"+str(case_num)+".ply")
    os.remove("union"+str(case_num)+".ply")

    pts_density = pcd_pts/areasum1    #point cloud density used
    number_of_points = int(areasum1*pts_density)
    union_pcd = sample_points_poisson_disk(union_mesh, number_of_points=number_of_points, init_factor=5)
    union_pts = union_pcd.points
    # Define the positions of the points, if inside sphere then they are defect
    total_defect = get_defect_index(union_pts,num,center,r)  # [None]*num
    print("defect IDs recorded!")
    # Add random noise from normal distribution to the point cloud
    random_points = add_random_noise(union_pts)
    print("noise added!")
    pcd4 = PointCloud() # randomized point cloud 
    pcd4.points = Vector3dVector(random_points)
    
    write_point_cloud(path_sample_pcd_save,pcd4)
    return center,r,total_defect

def generate_defect_mesh(polymesh_name, polypcd,path_sample_pcd_save, pcd_pts,r1,r2,num,case_num):
    ## Start to make sphere defect
    polymesh = read_triangle_mesh(polymesh_name)
    center_choice = np.asarray(polypcd.points).tolist()
    print(len(center_choice))
    areasum1,center,r,union=get_union_done_curv(polymesh_name,center_choice,r1,r2,num,case_num)
    print("get union done!")
    # Add random noise from normal distribution to the point cloud
    #random_points = add_random_noise(union_pts)
    #print("noise added!")
    #pcd4 = PointCloud() # randomized point cloud 
    #pcd4.points = Vector3dVector(random_points)
    union.export(path_sample_pcd_save,"ply")
    os.remove("union"+str(case_num)+".ply")
    #write_point_cloud(path_sample_pcd_save,pcd4)
    return center,r

