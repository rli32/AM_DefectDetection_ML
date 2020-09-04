
# coding: utf-8

# In[1]:


import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


F_result = pd.read_excel(open('Result.xlsx', 'rb'),sheet_name = 'F')
F_result = pd.DataFrame(F_result)


# In[3]:


G_result = pd.read_excel(open('parameter_tuning.xlsx', 'rb'),sheet_name = 'G')
G_result = pd.DataFrame(G_result)


# In[4]:


#F_result


# In[5]:


Z1_array = F_result.loc[0:3,('100000pts','90000pts','80000pts','70000pts')]
Z1_array = np.asarray(Z1_array)
Z1 = []
for i in Z1_array:
    for j in i:
        Z1.append(j)


# In[6]:


Z2_array = F_result.loc[8:11,('100000pts','90000pts','80000pts','70000pts')]
Z2_array = np.asarray(Z2_array)
Z2 = []
for i in Z2_array:
    for j in i:
        Z2.append(j)


# In[7]:


Z3_array = F_result.loc[16:19,('100000pts','90000pts','80000pts','70000pts')]
Z3_array = np.asarray(Z3_array)
Z3 = []
for i in Z3_array:
    for j in i:
        Z3.append(j)


# In[8]:


Z4_array = F_result.loc[24:27,('100000pts','90000pts','80000pts','70000pts')]
Z4_array = np.asarray(Z4_array)
Z4 = []
for i in Z4_array:
    for j in i:
        Z4.append(j)


# In[9]:


Z5_array = F_result.loc[32:35,('100000pts','90000pts','80000pts','70000pts')]
Z5_array = np.asarray(Z5_array)
Z5 = []
for i in Z5_array:
    for j in i:
        Z5.append(j)


# In[10]:


#G_result


# In[11]:


Z6_array = G_result.loc[0:4,('G_ave_1000pts','G_ave_1200pts','G_ave_1400pts','G_ave_1600pts','G_ave_1800pts')]
Z6_array = np.asarray(Z6_array)
Z6 = []
for i in Z6_array:
    for j in i:
        Z6.append(j)


# In[12]:


Z7_array = G_result.loc[9:13,('G_ave_1000pts','G_ave_1200pts','G_ave_1400pts','G_ave_1600pts','G_ave_1800pts')]
Z7_array = np.asarray(Z7_array)
Z7 = []
for i in Z7_array:
    for j in i:
        Z7.append(j)


# In[13]:


Z8_array = G_result.loc[18:22,('G_ave_1000pts','G_ave_1200pts','G_ave_1400pts','G_ave_1600pts','G_ave_1800pts')]
Z8_array = np.asarray(Z8_array)
Z8 = []
for i in Z8_array:
    for j in i:
        Z8.append(j)


# In[14]:


Z9_array = G_result.loc[27:31,('G_ave_1000pts','G_ave_1200pts','G_ave_1400pts','G_ave_1600pts','G_ave_1800pts')]
Z9_array = np.asarray(Z9_array)
Z9 = []
for i in Z9_array:
    for j in i:
        Z9.append(j)


# In[15]:


Z10_array = G_result.loc[36:40,('G_ave_1000pts','G_ave_1200pts','G_ave_1400pts','G_ave_1600pts','G_ave_1800pts')]
Z10_array = np.asarray(Z10_array)
Z10 = []
for i in Z10_array:
    for j in i:
        Z10.append(j)


# In[16]:


Y1 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y2 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y3 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y4 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y5 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y6 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y7 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y8 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y9 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]
Y10 = [100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000,100000,90000,80000,70000]


# In[17]:


X1 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X2 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X3 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X4 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X5 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X6 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X7 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X8 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X9 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]
X10 = [1.25,1.25,1.25,1.25,
      1.75,1.75,1.75,1.75,
      2.25,2.25,2.25,2.25,
      2.75,2.75,2.75,2.75]

order_num = 3

# In[18]:


# some 3-dim points
#X1 = [0.0025,0.005,0.0075,0.01,0.0125,0.0025,0.005,0.0075,0.01,0.0125,0.0025,0.005,0.0075,0.01,0.0125,
#     0.0025,0.005,0.0075,0.01,0.0125,0.0025,0.005,0.0075,0.01,0.0125]

data1 = np.c_[X1,Y1,Z1]

mn = np.min(data1, axis=0)
mx = np.max(data1, axis=0)
X1,Y1 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X1.flatten()
YY = Y1.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data1[:,0], data1[:,1], np.ones(data1.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data1[:,2])    # coefficients
    
    # evaluate it on grid
    Z1 = C[0]*X1 + C[1]*Y1 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data1.shape[0]), data1[:,:2], np.prod(data1[:,:2], axis=1), data1[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data1[:,2])
    
    # evaluate it on a grid
    Z1 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X1.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data1.shape[0]), data1[:,:2], np.prod(data1[:,:2], axis=1), 
              data1[:,:2]**2,data1[:,:2]**3,data1[:,0]**2*data1[:,1],data1[:,1]**2*data1[:,0]]
    C1,_,_,_ = scipy.linalg.lstsq(A, data1[:,2])
    
    # evaluate it on a grid
    Z1 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C1).reshape(X1.shape)

# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X1, Y1, Z1,label='F-measure of Bagging',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data1[:,0], data1[:,1], data1[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('F-score', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "F-measure of Bagging", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F-bagging_2order.png",dpi=300)
plt.show()


# In[19]:


data2 = np.c_[X2,Y2,Z2]

mn = np.min(data2, axis=0)
mx = np.max(data2, axis=0)
X2,Y2 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X2.flatten()
YY = Y2.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data2[:,0], data2[:,1], np.ones(data2.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data2[:,2])    # coefficients
    
    # evaluate it on grid
    Z2 = C[0]*X2 + C[1]*Y2 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data2.shape[0]), data2[:,:2], np.prod(data2[:,:2], axis=1), data2[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data2[:,2])
    
    # evaluate it on a grid
    Z2 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X2.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data2.shape[0]), data2[:,:2], np.prod(data2[:,:2], axis=1), 
              data2[:,:2]**2,data2[:,:2]**3,data2[:,0]**2*data2[:,1],data2[:,1]**2*data2[:,0]]
    C2,_,_,_ = scipy.linalg.lstsq(A, data2[:,2])
    
    # evaluate it on a grid
    Z2 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C2).reshape(X1.shape)

# plot points and fitted surface
fig = plt.figure()
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X2, Y2, Z2,label='F-measure of Gradient Boosting',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data2[:,0], data2[:,1], data2[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('F-score', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "F-measure of Gradient Boosting", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F-gradient boosting_2order.png",dpi=300)
plt.show()


# In[20]:

data3 = np.c_[X3,Y3,Z3]

mn = np.min(data3, axis=0)
mx = np.max(data3, axis=0)
X3,Y3 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X3.flatten()
YY = Y3.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data3[:,0], data3[:,1], np.ones(data3.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data3[:,2])    # coefficients
    
    # evaluate it on grid
    Z3 = C[0]*X3 + C[1]*Y3 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data3.shape[0]), data3[:,:2], np.prod(data3[:,:2], axis=1), data3[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data3[:,2])
    
    # evaluate it on a grid
    Z3 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X3.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data3.shape[0]), data3[:,:2], np.prod(data3[:,:2], axis=1), 
              data3[:,:2]**2,data3[:,:2]**3,data3[:,0]**2*data3[:,1],data3[:,1]**2*data3[:,0]]
    C3,_,_,_ = scipy.linalg.lstsq(A, data3[:,2])
    
    # evaluate it on a grid
    Z3 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C3).reshape(X1.shape)


# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X3, Y3, Z3,label='F-measure of Random Forest',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data3[:,0], data3[:,1], data3[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('F-score', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "F-measure of Random Forest", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F-random forest_2order.png",dpi=300)
plt.show()


# In[21]:


data4 = np.c_[X4,Y4,Z4]

mn = np.min(data4, axis=0)
mx = np.max(data4, axis=0)
X4,Y4 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X4.flatten()
YY = Y4.flatten()

order = order_num   # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data4[:,0], data4[:,1], np.ones(data4.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data4[:,2])    # coefficients
    
    # evaluate it on grid
    Z4 = C[0]*X4 + C[1]*Y4 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data4.shape[0]), data4[:,:2], np.prod(data4[:,:2], axis=1), data4[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data4[:,2])
    
    # evaluate it on a grid
    Z4 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X4.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data4.shape[0]), data4[:,:2], np.prod(data4[:,:2], axis=1), 
              data4[:,:2]**2,data4[:,:2]**3,data4[:,0]**2*data4[:,1],data4[:,1]**2*data4[:,0]]
    C4,_,_,_ = scipy.linalg.lstsq(A, data4[:,2])
    
    # evaluate it on a grid
    Z4 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C4).reshape(X4.shape)


# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X4, Y4, Z4,label='F-measure of SVM',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data4[:,0], data4[:,1], data4[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('F-score', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "F-measure of SVM", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F-SVM_2order.png",dpi=300)
plt.show()


# In[22]:

data5 = np.c_[X5,Y5,Z5]

mn = np.min(data5, axis=0)
mx = np.max(data5, axis=0)
X5,Y5 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X5.flatten()
YY = Y5.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data5[:,0], data5[:,1], np.ones(data5.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data5[:,2])    # coefficients
    
    # evaluate it on grid
    Z5 = C[0]*X5 + C[1]*Y5 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data5.shape[0]), data5[:,:2], np.prod(data5[:,:2], axis=1), data5[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data5[:,2])
    
    # evaluate it on a grid
    Z5 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X5.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data5.shape[0]), data5[:,:2], np.prod(data5[:,:2], axis=1), 
              data5[:,:2]**2,data5[:,:2]**3,data5[:,0]**2*data5[:,1],data5[:,1]**2*data5[:,0]]
    C5,_,_,_ = scipy.linalg.lstsq(A, data5[:,2])
    
    # evaluate it on a grid
    Z5 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C5).reshape(X5.shape)


# plot points and fitted surface
fig = plt.figure()
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X5, Y5, Z5,label='F-measure of K-nearest Neighbors',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data5[:,0], data5[:,1], data5[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('F-score', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "F-measure of K-nearest Neighbors", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F-K-nearest neighbors_order.png",dpi=300)
plt.show()


# In[23]:

data6 = np.c_[X6,Y6,Z6]

mn = np.min(data6, axis=0)
mx = np.max(data6, axis=0)
X6,Y6 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X6.flatten()
YY = Y6.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data6[:,0], data6[:,1], np.ones(data6.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data6[:,2])    # coefficients
    
    # evaluate it on grid
    Z6 = C[0]*X6 + C[1]*Y6 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data6.shape[0]), data6[:,:2], np.prod(data6[:,:2], axis=1), data6[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data6[:,2])
    
    # evaluate it on a grid
    Z6 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X6.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data6.shape[0]), data6[:,:2], np.prod(data6[:,:2], axis=1), 
              data6[:,:2]**2,data6[:,:2]**3,data6[:,0]**2*data6[:,1],data6[:,1]**2*data6[:,0]]
    C6,_,_,_ = scipy.linalg.lstsq(A, data6[:,2])
    
    # evaluate it on a grid
    Z6 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C6).reshape(X6.shape)

# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X6, Y6, Z6,label='G-mean of Bagging',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data6[:,0], data6[:,1], data6[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('G-mean', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "G-mean of Bagging", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G-bagging_order.png",dpi=300)
plt.show()


# In[24]:

data7 = np.c_[X7,Y7,Z7]

mn = np.min(data7, axis=0)
mx = np.max(data7, axis=0)
X7,Y7 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X7.flatten()
YY = Y7.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data7[:,0], data7[:,1], np.ones(data7.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data7[:,2])    # coefficients
    
    # evaluate it on grid
    Z7 = C[0]*X7 + C[1]*Y7 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data7.shape[0]), data7[:,:2], np.prod(data7[:,:2], axis=1), data7[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data7[:,2])
    
    # evaluate it on a grid
    Z7 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X7.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data7.shape[0]), data7[:,:2], np.prod(data7[:,:2], axis=1), 
              data7[:,:2]**2,data7[:,:2]**3,data7[:,0]**2*data7[:,1],data7[:,1]**2*data7[:,0]]
    C7,_,_,_ = scipy.linalg.lstsq(A, data7[:,2])
    
    # evaluate it on a grid
    Z7 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C7).reshape(X7.shape)


# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X7, Y7, Z7,label='G-mean of Gradient Boosting',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data7[:,0], data7[:,1], data7[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('G-mean', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "G-mean of Gradient Boosting", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G-gradient boosting_order.png",dpi=300)
plt.show()


# In[25]:


data8 = np.c_[X8,Y8,Z8]

mn = np.min(data8, axis=0)
mx = np.max(data8, axis=0)
X8,Y8 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X8.flatten()
YY = Y8.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data8[:,0], data8[:,1], np.ones(data8.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data8[:,2])    # coefficients
    
    # evaluate it on grid
    Z8 = C[0]*X8 + C[1]*Y8 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data8.shape[0]), data8[:,:2], np.prod(data8[:,:2], axis=1), data8[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data8[:,2])
    
    # evaluate it on a grid
    Z8 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X8.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data8.shape[0]), data8[:,:2], np.prod(data8[:,:2], axis=1), 
              data8[:,:2]**2,data8[:,:2]**3,data8[:,0]**2*data8[:,1],data8[:,1]**2*data8[:,0]]
    C8,_,_,_ = scipy.linalg.lstsq(A, data8[:,2])
    
    # evaluate it on a grid
    Z8 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C8).reshape(X8.shape)


# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X8, Y8, Z8,label='G-mean of Random Forest',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data8[:,0], data8[:,1], data8[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('G-mean', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "G-mean of Random Forest", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G-random forest_order.png",dpi=300)
plt.show()


# In[26]:



data9 = np.c_[X9,Y9,Z9]

mn = np.min(data9, axis=0)
mx = np.max(data9, axis=0)
X9,Y9 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X9.flatten()
YY = Y9.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data9[:,0], data9[:,1], np.ones(data9.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data9[:,2])    # coefficients
    
    # evaluate it on grid
    Z9 = C[0]*X9 + C[1]*Y9 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data9.shape[0]), data9[:,:2], np.prod(data9[:,:2], axis=1), data9[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data9[:,2])
    
    # evaluate it on a grid
    Z9 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X9.shape)
    
elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data9.shape[0]), data9[:,:2], np.prod(data9[:,:2], axis=1), 
              data9[:,:2]**2,data9[:,:2]**3,data9[:,0]**2*data9[:,1],data9[:,1]**2*data9[:,0]]
    C9,_,_,_ = scipy.linalg.lstsq(A, data9[:,2])
    
    # evaluate it on a grid
    Z9 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C9).reshape(X9.shape)

# plot points and fitted surface
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X9, Y9, Z9,label='G-mean of SVM',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data9[:,0], data9[:,1], data9[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('G-mean', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "G-mean of SVM", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G-SVM_2order.png",dpi=300)
plt.show()


# In[27]:

data10 = np.c_[X10,Y10,Z10]

mn = np.min(data10, axis=0)
mx = np.max(data10, axis=0)
X10,Y10 = np.meshgrid(np.linspace(mn[0], mx[0], 400), np.linspace(mn[1], mx[1], 400))
XX = X10.flatten()
YY = Y10.flatten()

order = order_num    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data10[:,0], data10[:,1], np.ones(data10.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data10[:,2])    # coefficients
    
    # evaluate it on grid
    Z10 = C[0]*X10 + C[1]*Y10 + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data10.shape[0]), data10[:,:2], np.prod(data10[:,:2], axis=1), data10[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data10[:,2])
    
    # evaluate it on a grid
    Z10 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X10.shape)

elif order == 3:
    # best-fit 3rd curve
    A = np.c_[np.ones(data10.shape[0]), data10[:,:2], np.prod(data10[:,:2], axis=1), 
              data10[:,:2]**2,data10[:,:2]**3,data10[:,0]**2*data10[:,1],data10[:,1]**2*data10[:,0]]
    C10,_,_,_ = scipy.linalg.lstsq(A, data10[:,2])
    
    # evaluate it on a grid
    Z10 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2,XX**3, YY**3, XX**2*YY, YY**2*XX], C10).reshape(X10.shape)


# plot points and fitted surface
fig = plt.figure()
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X10, Y10, Z10,label='G-mean of K-nearest Neighbors',cmap='viridis',rstride=1, cstride=1, alpha=0.8)

x_scale=4
y_scale=4
z_scale=4

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
ax.scatter(data10[:,0], data10[:,1], data10[:,2], c='r', s=50)
#plt.xlabel('Defect Size',fontsize=15)
#plt.ylabel('Number of Patch Points',fontsize=15)
ax.set_zlabel('G-mean', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.text2D(0.25, 1, "G-mean of K-nearest Neighbors", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G-K-nearest Neighbors_order.png",dpi=300)
plt.show()


# In[28]:


import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.scatter(data1[:,0], data1[:,1], data1[:,2], label='Bagging', c='r', s=50)
ax.plot_surface(X1, Y1, Z1,color='red',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data2[:,0], data2[:,1], data2[:,2], label='Gradient Boosting',c='orange', s=50)
ax.plot_surface(X2, Y2, Z2,color='orange',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data3[:,0], data3[:,1], data3[:,2],label='Random Forest', c='y', s=50)
ax.plot_surface(X3, Y3, Z3,color='yellow',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data4[:,0], data4[:,1], data4[:,2],label='SVM', c='g', s=50)
ax.plot_surface(X4, Y4, Z4,color='green',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data5[:,0], data5[:,1], data5[:,2], label='K-nearest Neighbors',c='b', s=50)
ax.plot_surface(X5, Y5, Z5,color='blue',rstride=1, cstride=1, alpha=0.3)
ax.legend()
ax.set_zlabel('F-score',fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (mm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.tick_params(axis='x', which='major', pad=0)
ax.text2D(0.35, 1, "F-measure", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F_with_scatters.png",dpi=300)
plt.show()


# In[29]:


import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
#ax.scatter(data1[:,0], data1[:,1], data1[:,2], label='Bagging', c='r', s=50)
#ax.bar3d(data1[:,0], data1[:,1], data1[:,2], 0.2,0.2,0.2,label='Bagging', color='r')
p1=ax.plot_surface(X1, Y1, Z1,color='red',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data2[:,0], data2[:,1], data2[:,2], label='Gradient Boosting',color='orange', s=50)
#ax.bar3d(data2[:,0], data2[:,1], data2[:,2], 0.2,0.2,0.2,label='Gradient Boosting',color='orange')
p2=ax.plot_surface(X2, Y2, Z2,color='orange',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data3[:,0], data3[:,1], data3[:,2],label='Random Forest', c='y', s=50)
#ax.bar3d(data3[:,0], data3[:,1], data3[:,2], 0.2,0.2,0.2,label='Random Forest', color='y')
p3=ax.plot_surface(X3, Y3, Z3,color='yellow',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data4[:,0], data4[:,1], data4[:,2],label='SVM', c='g', s=50)
#ax.bar3d(data4[:,0], data4[:,1], data4[:,2], 0.2,0.2,0.2,label='SVM', color='g')
p4=ax.plot_surface(X4, Y4, Z4,color='green',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data5[:,0], data5[:,1], data5[:,2], label='K-nearest Neighbors',c='b', s=50)
#ax.bar3d(data5[:,0], data5[:,1], data5[:,2], 0.2,0.2,0.2,label='K-nearest Neighbors',color='b')
p5=ax.plot_surface(X5, Y5, Z5,color='blue',rstride=1, cstride=1, alpha=0.3)


prox1 = plt.Rectangle((0, 0), 1, 1, fc="r",alpha=0.3)
prox2 = plt.Rectangle((0, 0), 1, 1, fc="orange",alpha=0.3)
prox3 = plt.Rectangle((0, 0), 1, 1, fc="y",alpha=0.3)
prox4 = plt.Rectangle((0, 0), 1, 1, fc="g",alpha=0.3)
prox5 = plt.Rectangle((0, 0), 1, 1, fc="b",alpha=0.3)
ax.legend((prox1, prox2,prox3, prox4,prox5), ('Bagging', 'Gradient Boosting','Random Forest','SVM','K-nearest Neighbors'))
ax.set_zlabel('F-score',fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (mm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.tick_params(axis='x', which='major', pad=0)
ax.text2D(0.35, 1, "F-measure", fontsize=26,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("F_without_scatters.png",dpi=300)
plt.show()


# In[30]:


#X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 100), np.linspace(mn[1], mx[1], 100))
Z1new = np.zeros_like(X1)
Z2new = np.zeros_like(X1)
Z3new = np.zeros_like(X1)
Z4new = np.zeros_like(X1)
Z5new = np.zeros_like(X1)
for i in range(400):
    for j in range(400):
        if Z1[i][j] > Z2[i][j] and Z1[i][j] > Z3[i][j] and Z1[i][j] > Z4[i][j] and Z1[i][j] > Z5[i][j]:
            Z1new[i][j] = Z1[i][j]
        else:
            Z1new[i][j] = np.nan

for i in range(400):
    for j in range(400):            
        if Z2[i][j] > Z1[i][j] and Z2[i][j] > Z3[i][j] and Z2[i][j] > Z4[i][j] and Z2[i][j] > Z5[i][j]:
            Z2new[i][j] = Z2[i][j]
        else:
            Z2new[i][j] = np.nan
            
for i in range(400):
    for j in range(400):
        if Z3[i][j] > Z2[i][j] and Z3[i][j] > Z1[i][j] and Z3[i][j] > Z4[i][j] and Z3[i][j] > Z5[i][j]:
            Z3new[i][j] = Z3[i][j]
        else:
            Z3new[i][j] = np.nan
            
for i in range(400):
    for j in range(400):
        if Z4[i][j] > Z2[i][j] and Z4[i][j] > Z3[i][j] and Z4[i][j] > Z1[i][j] and Z4[i][j] > Z5[i][j]:
            Z4new[i][j] = Z4[i][j]
        else:
            Z4new[i][j] = np.nan
            
for i in range(400):
    for j in range(400):
        if Z5[i][j] > Z2[i][j] and Z5[i][j] > Z3[i][j] and Z5[i][j] > Z4[i][j] and Z5[i][j] > Z1[i][j]:
            Z5new[i][j] = Z5[i][j]
        else:
            Z5new[i][j] = np.nan


# In[31]:


import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(12,12))
#fig.subplots_adjust(left=0.2)
#fig.subplots_adjust(right=1)
#fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(top=1)
ax = fig.gca(projection='3d')
ax.set_proj_type('ortho')
p1=ax.plot_surface(X1, Y1, Z1new,color='red',rstride=1, cstride=1, alpha=0.3)
p2=ax.plot_surface(X2, Y2, Z2new,color='orange',rstride=1, cstride=1, alpha=0.3)
p3=ax.plot_surface(X3, Y3, Z3new,color='yellow',rstride=1, cstride=1, alpha=0.3)
p4=ax.plot_surface(X4, Y4, Z4new,color='green',rstride=1, cstride=1, alpha=0.3)
p5=ax.plot_surface(X5, Y5, Z5new,color='blue',rstride=1, cstride=1, alpha=0.3)


prox1 = plt.Rectangle((0, 0), 1, 1, fc="r",alpha=0.3)
prox2 = plt.Rectangle((0, 0), 1, 1, fc="orange",alpha=0.3)
prox3 = plt.Rectangle((0, 0), 1, 1, fc="y",alpha=0.3)
prox4 = plt.Rectangle((0, 0), 1, 1, fc="g",alpha=0.3)
prox5 = plt.Rectangle((0, 0), 1, 1, fc="b",alpha=0.3)
ax.legend((prox1, prox2,prox3, prox4,prox5), ('Bagging', 'Gradient Boosting','Random Forest','SVM','K-nearest Neighbors'),
          loc='best', bbox_to_anchor=(0.57, 0.1, 0.5, 0.5))
#zLabel = ax.set_zlabel('F-measure',fontsize=15,linespacing=10.2,labelpad=30)
#ax.set_zlabel('F-score',fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (mm)', fontname = "Calibri",fontsize=22, labelpad = 15)
ax.set_ylabel('Point Cloud Density', fontname = "Calibri",fontsize=22, labelpad = 35)
ax.zaxis.set_tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.tick_params(axis='x', which='major', pad=0)
ax.text2D(0.45, 0.85, "F-measure", fontsize=26,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
#plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
ax.view_init(90, 90)
plt.savefig("F-best.png",dpi=300)
plt.show()


# In[32]:


import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.scatter(data6[:,0], data6[:,1], data6[:,2], label='Bagging', c='r', s=50)
ax.plot_surface(X6, Y6, Z6,color='red',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data7[:,0], data7[:,1], data7[:,2], label='Gradient Boosting',c='orange', s=50)
ax.plot_surface(X7, Y7, Z7,color='orange',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data8[:,0], data8[:,1], data8[:,2],label='Random Forest', c='y', s=50)
ax.plot_surface(X8, Y8, Z8,color='yellow',rstride=1, cstride=1, alpha=0.3)
ax.scatter(data9[:,0], data9[:,1], data9[:,2],label='SVM', c='g', s=50)
ax.plot_surface(X9, Y9, Z9,color='green',rstride=1, cstride=1, alpha=0.5)
ax.scatter(data10[:,0], data10[:,1], data10[:,2], label='K-nearest Neighbors',c='b', s=50)
ax.plot_surface(X10, Y10, Z10,color='blue',rstride=1, cstride=1, alpha=0.3)
ax.legend()
ax.set_zlabel('G-mean',fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.tick_params(axis='x', which='major', pad=0)
ax.text2D(0.35, 1, "G-mean", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G_with_scatters.png",dpi=300)
plt.show()


# In[33]:


###### import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
#ax.scatter(data1[:,0], data1[:,1], data1[:,2], label='Bagging', c='r', s=50)
#ax.bar3d(data1[:,0], data1[:,1], data1[:,2], 0.2,0.2,0.2,label='Bagging', color='r')
p1=ax.plot_surface(X6, Y6, Z6,color='red',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data2[:,0], data2[:,1], data2[:,2], label='Gradient Boosting',color='orange', s=50)
#ax.bar3d(data2[:,0], data2[:,1], data2[:,2], 0.2,0.2,0.2,label='Gradient Boosting',color='orange')
p2=ax.plot_surface(X7, Y7, Z7,color='orange',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data3[:,0], data3[:,1], data3[:,2],label='Random Forest', c='y', s=50)
#ax.bar3d(data3[:,0], data3[:,1], data3[:,2], 0.2,0.2,0.2,label='Random Forest', color='y')
p3=ax.plot_surface(X8, Y8, Z8,color='yellow',rstride=1, cstride=1, alpha=0.3)
#ax.scatter(data4[:,0], data4[:,1], data4[:,2],label='SVM', c='g', s=50)
#ax.bar3d(data4[:,0], data4[:,1], data4[:,2], 0.2,0.2,0.2,label='SVM', color='g')
p4=ax.plot_surface(X9, Y9, Z9,color='green',rstride=1, cstride=1, alpha=0.5)
#ax.scatter(data5[:,0], data5[:,1], data5[:,2], label='K-nearest Neighbors',c='b', s=50)
#ax.bar3d(data5[:,0], data5[:,1], data5[:,2], 0.2,0.2,0.2,label='K-nearest Neighbors',color='b')
p5=ax.plot_surface(X10, Y10, Z10,color='blue',rstride=1, cstride=1, alpha=0.3)
prox1 = plt.Rectangle((0, 0), 1, 1, fc="r",alpha=0.3)
prox2 = plt.Rectangle((0, 0), 1, 1, fc="orange",alpha=0.3)
prox3 = plt.Rectangle((0, 0), 1, 1, fc="y",alpha=0.3)
prox4 = plt.Rectangle((0, 0), 1, 1, fc="g",alpha=0.3)
prox5 = plt.Rectangle((0, 0), 1, 1, fc="b",alpha=0.3)
ax.legend((prox1, prox2,prox3, prox4,prox5), ('Bagging', 'Gradient Boosting','Random Forest','SVM','K-nearest Neighbors'))
ax.set_zlabel('G-mean',fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.tick_params(axis='x', which='major', pad=0)
ax.text2D(0.35, 1, "G-mean", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.savefig("G_without_scatters.png",dpi=300)
plt.show()


# In[34]:


#X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 100), np.linspace(mn[1], mx[1], 100))
Z6new = np.zeros_like(X6)
Z7new = np.zeros_like(X7)
Z8new = np.zeros_like(X8)
Z9new = np.zeros_like(X9)
Z10new = np.zeros_like(X1)
for i in range(400):
    for j in range(400):
        if Z6[i][j] > Z7[i][j] and Z6[i][j] > Z8[i][j] and Z6[i][j] > Z9[i][j] and Z6[i][j] > Z10[i][j]:
            Z6new[i][j] = Z6[i][j]
        else:
            Z6new[i][j] = np.nan

for i in range(400):
    for j in range(400):
        if Z7[i][j] > Z6[i][j] and Z7[i][j] > Z8[i][j] and Z7[i][j] > Z9[i][j] and Z7[i][j] > Z10[i][j]:
            Z7new[i][j] = Z7[i][j]
        else:
            Z7new[i][j] = np.nan
            
for i in range(400):
    for j in range(400):
        if Z8[i][j] > Z7[i][j] and Z8[i][j] > Z6[i][j] and Z8[i][j] > Z9[i][j] and Z8[i][j] > Z10[i][j]:
            Z8new[i][j] = Z8[i][j]
        else:
            Z8new[i][j] = np.nan
            
for i in range(400):
    for j in range(400):
        if Z9[i][j] > Z7[i][j] and Z9[i][j] > Z8[i][j] and Z9[i][j] > Z6[i][j] and Z9[i][j] > Z10[i][j]:
            Z9new[i][j] = Z9[i][j]
        else:
            Z9new[i][j] = np.nan
            
for i in range(400):
    for j in range(400):
        if Z10[i][j] > Z7[i][j] and Z10[i][j] > Z8[i][j] and Z10[i][j] > Z9[i][j] and Z10[i][j] > Z6[i][j]:
            Z10new[i][j] = Z10[i][j]
        else:
            Z10new[i][j] = np.nan


# In[35]:


import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(12,12))
#fig.subplots_adjust(left=0.2)
#fig.subplots_adjust(right=1)
#fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(top=1)
ax = fig.gca(projection='3d')
ax.set_proj_type('ortho')
p6=ax.plot_surface(X1, Y1, Z6new,color='red',rstride=1, cstride=1, alpha=0.3)
p7=ax.plot_surface(X2, Y2, Z7new,color='orange',rstride=1, cstride=1, alpha=0.3)
p8=ax.plot_surface(X3, Y3, Z8new,color='yellow',rstride=1, cstride=1, alpha=0.3)
p9=ax.plot_surface(X4, Y4, Z9new,color='green',rstride=1, cstride=1, alpha=0.3)
p10=ax.plot_surface(X5, Y5, Z10new,color='blue',rstride=1, cstride=1, alpha=0.3)


prox6 = plt.Rectangle((0, 0), 1, 1, fc="r",alpha=0.3)
prox7 = plt.Rectangle((0, 0), 1, 1, fc="orange",alpha=0.3)
prox8 = plt.Rectangle((0, 0), 1, 1, fc="y",alpha=0.3)
prox9 = plt.Rectangle((0, 0), 1, 1, fc="g",alpha=0.3)
prox10 = plt.Rectangle((0, 0), 1, 1, fc="b",alpha=0.3)
ax.legend((prox6, prox7,prox8, prox9,prox10), ('Bagging', 'Gradient Boosting','Random Forest','SVM','K-nearest Neighbors'),
          loc='best', bbox_to_anchor=(0.57, 0.1, 0.5, 0.5))
#zLabel = ax.set_zlabel('G-mean',fontsize=15,linespacing=10.2,labelpad=30)
#ax.set_zlabel('G-mean',fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_xlabel('Hemispherical Defect Radius (cm)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.set_ylabel('Point Cloud Density (/cm**2)', fontname = "Calibri",fontsize=20, labelpad = 10)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.tick_params(axis='x', which='major', pad=0)
for tick in ax.get_xticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_yticklabels():
    tick.set_fontname("Calibri")
for tick in ax.get_zticklabels():
    tick.set_fontname("Calibri")
ax.tick_params(axis='x', which='major', pad=0)
ax.text2D(0.45, 0.85, "G-mean", fontsize=20,transform=ax.transAxes)
#ax.set_xticks([0.0375, 0.0625, 0.0875,0.1125,0.1375])
#ax.set_yticks([1000, 1200, 1400,1600,1800])
ax.set_zticks([0.0,0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0])
ax.axis('tight')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
ax.view_init(90, 90)
plt.savefig("G-best1.png",dpi=300)
plt.show()

