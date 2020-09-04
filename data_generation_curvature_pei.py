

import numpy as np
from open3d import *
import pandas
from pit_poisson_random_pei import * # generate_defect
from registration_pei import *
import sys
import os
import trimesh

#from open3d.io import read_point_cloud,read_triangle_mesh,write_point_cloud,write_triangle_mesh
#from open3d.registration import registration_icp,compute_fpfh_feature,TransformationEstimationPointToPlane,\
#                         TransformationEstimationPointToPoint
#from utility import Vector3dVector,Vector3iVector
#from open3d.visualization import draw_geometries
#from open3d.geometry.PointCloud import paint_uniform_color,voxel_down_sample,estimate_normals
#from open3d.geometry import *
#from open3d.geometry.TriangleMesh import sample_points_poisson_disk,create_sphere,sample_points_uniformly
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

#define lists for defect sphere centers, radius and defect point index
sphere_center = []
sphere_r = []
total_defect = []

# point cloud density list
pcd_pts = int(sys.argv[2]) #[1000]
case_dir=str(pcd_pts)
print("point cloud density:", pcd_pts)
r1 = 1.5
r2 = 2
defect_num =10
case_num = sys.argv[1] # 1
print("case_num:", case_num)

import time
time1=time.time()

#generate point cloud with sphere defects in defined radius range
pcd_polymesh = 'barrel.ply' #'Plane.ply'
path_pcd_barrel=os.path.join(case_dir,'pcd_barrel.ply') #'pcd_plane.ply')
print(path_pcd_barrel)
polypcd = read_point_cloud(path_pcd_barrel)
print("polypcd:",polypcd)
defname = 'union'+str(case_num)
path_sample = os.path.join(case_dir,'radius'+str(r1)+'-'+str(r2))
if os.path.exists(path_sample)==False:
  os.mkdir(path_sample)
path_sample_pcd_save=os.path.join(path_sample,defname+'.ply')

j,exit_step = 1, 0
while j>0: # and exit_step<30:
    try:
        center,r = generate_defect_mesh(pcd_polymesh,polypcd,path_sample_pcd_save,pcd_pts, r1,r2,defect_num,case_num)
        #os.remove("mesh_sphere"+str(case_num)+".ply") #remove temperary sphere file generted in try_union_once()
        #os.remove("union"+str(case_num)+".ply") #remove temperary union file generted in try_union_once()
        j -= 1
    except:
        print("Exception happens in file union"+str(case_num))
        exit_step += 1
center,r = generate_defect_mesh(pcd_polymesh,polypcd,path_sample_pcd_save,pcd_pts, r1,r2,defect_num,case_num)
union_mesh = read_triangle_mesh(path_sample_pcd_save)
union_pts = np.asarray(union_mesh.vertices)
union_triangles = np.asarray(union_mesh.triangles)



print("Time1:", time.time()-time1)
time2=time.time()

#make the defects center, radius, index list to dataframe
sphere_center = pandas.DataFrame(center)
sphere_r = pandas.DataFrame(r)


print("Time2:", time.time()-time2)
time3=time.time()

import os.path
path=str(pcd_pts)
directory=['info_defect','radius'+str(r1)+'-'+str(r2)]
for i in directory:
  path=os.path.join(path,i)
  if os.path.exists(path)==False:
    os.mkdir(path)
path_sc=os.path.join(path,'sphere_center_curv'+str(case_num)+'.csv')
sphere_center.to_csv(path_sc)
path_sr=os.path.join(path,'sphere_r_curv'+str(case_num)+'.csv')
sphere_r.to_csv(path_sr)

#ptpl_dist_total = []
source,target, targetmesh, ptpl_distance = [],[],[],[]

source_mesh_tri = trimesh.load_mesh(path_sample_pcd_save,'ply')
source_vertices = source_mesh_tri.vertices
source_faces = source_mesh_tri.faces
new_sour_vertices, new_sour_faces = trimesh.remesh.subdivide_to_size(source_vertices, source_faces,max_edge = 0.4)
trimesh.repair.fix_inversion(source_mesh_tri)
trimesh.repair.fix_winding(source_mesh_tri)
trimesh.repair.fill_holes(source_mesh_tri)
source_curv = trimesh.curvature.discrete_gaussian_curvature_measure(source_mesh_tri, new_sour_vertices,2)

target_mesh_tri = trimesh.load_mesh(pcd_polymesh,'ply')
target_vertices = target_mesh_tri.vertices
target_faces = target_mesh_tri.faces
new_tar_vertices, new_tar_faces = trimesh.remesh.subdivide_to_size(target_vertices, target_faces,max_edge = 0.4)
trimesh.repair.fix_inversion(target_mesh_tri)
trimesh.repair.fix_winding(target_mesh_tri)
trimesh.repair.fill_holes(target_mesh_tri)
target_curv = trimesh.curvature.discrete_gaussian_curvature_measure(target_mesh_tri, new_tar_vertices,2)

source = open3d.geometry.PointCloud()
source.points = open3d.utility.Vector3dVector(np.asarray(source_vertices))
target = open3d.geometry.PointCloud()
target.points = open3d.utility.Vector3dVector(np.asarray(target_vertices))

print("source",source)
print("target",target)
print("registration_distance start!")
ptpl_distance,min_ind_list = registration_distance_curv(source,target)
#rd = registration_distance(source,target)
#ptpl_distance, source, target, source_curv, target_curv = rd[0], rd[1], rd[2], rd[3], rd[4]

print("Time3:", time.time()-time3)
time4=time.time()

# Define the positions of the points, if inside sphere then they are defect
total_defect = get_defect_index(source,defect_num,center,r)  # [None]*num
print("defect IDs recorded!")

sphere_defect = pandas.DataFrame(total_defect)
path_sd=os.path.join(path,'sphere_defect_curv'+str(case_num)+'.csv')
sphere_defect.to_csv(path_sd)

defect_index = np.asarray(sphere_defect)

patch = []
if defect_num == 1:
    defect_index = defect[1:]
    #patch = datapreparing(source, target,ptpl_distance,defect_index,source_curv,target_curv)
    patch = datapreparing_curv(source, target,ptpl_distance,min_ind_list,source_curv,target_curv,defect_index)

else:
    defect_index = defect[1:][0]
    #patch = datapreparing(source, target,ptpl_distance,defect_index,source_curv,target_curv)
    patch = datapreparing_curv(source, target,ptpl_distance,min_ind_list,source_curv,target_curv,defect_index)

data = pandas.DataFrame(data = patch)
direct_prep = [case_dir,'radius'+str(r1)+'-'+str(r2)]
path_prep_temp='preprocess_data'
if os.path.exists(path_prep_temp)==False:
  os.mkdir(path_prep_temp)

for i in direct_prep:
  path_prep_temp=os.path.join(path_prep_temp,i)
  if os.path.exists(path_prep_temp)==False:
    os.mkdir(path_prep_temp)

path_csv_save = os.path.join(path_prep_temp,'data_curv_'+str(case_num)+'.csv')

data.to_csv(path_csv_save)

print("Time4:", time.time()-time4)


