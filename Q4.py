import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
import csv
from pyntcloud import PyntCloud
import scipy.spatial as spt
from math import *


fujian=pd.read_excel('./附件.xlsx',sheet_name=1,header=None)
# print(fujian)
D = np.array(fujian.loc[1:251,1:201])
# print(D)
# print(D[126][101])
XX=np.array((fujian.loc[0,1:201]))
# print(X)
Y = np.array((fujian.loc[1:251,0]))
# print(Y)
P=[]
for i in range(201):
    for j in range(251):
        P.append([XX[i],Y[j],-D[j][i]])
P=np.array(P)
P[:,:2]=P[:,:2]*1852
# print(P)
X1=[]
Y1=[]
Z1=[]
for i in range(len(P)):
    X1.append(P[i][0])
for i in range(len(P)):
    Y1.append(P[i][1])
for i in range(len(P)):
    Z1.append(P[i][2])

x_grid = np.linspace(min(X1), max(X1), 1000)
y_grid = np.linspace(min(Y1), max(Y1), 1000)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# 使用griddata进行三维插值
Z_grid = griddata((X1, Y1), Z1, (X_grid, Y_grid), method='cubic')

# 绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Interpolation')

# 显示图像
plt.show()

def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


class Kmeans():
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛
    """

    def __init__(self, k=4, max_iterations=500, varepsilon=0.001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        self.Pcen = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
            self.Pcen[i]=np.array([centroid[0]*4*1852,centroid[1]*5*1852,centroid[2]*177.2-197.2])
            print("归一化随机坐标：{}，相应的原坐标：{}".format(centroid,self.Pcen[i]))
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
        for i in range(self.k):
            self.Pcen[i]=np.array([centroids[i][0]*4*1852,centroids[i][1]*5*1852,centroids[i][2]*177.2-197.2])
        return self.get_cluster_labels(clusters, X)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X=min_max_scaler.fit_transform(P)
num, dim = X.shape
clf = Kmeans(k=5) #划分为5块
y_pred = clf.predict(X)
print("聚类中心坐标：{}".format(clf.Pcen)) #打印出聚类中心坐标
#获取各类所有的点：
P_cla=[[],[],[],[],[]] #分别对应5类
for i in range(len(P)):
    P_cla[int(y_pred[i])].append(P[i])

#KDtree

tree = spt.cKDTree(data=P)
#每一类坡面的法向量
normal_vector=[]
#每一类坡面的坡角（弧度制）
alpha_list=np.zeros(5)
for point in clf.Pcen:
    distances, indexs = tree.query(point, k=4)
    normal_vector.append(np.cross(P[indexs[1]]-P[indexs[2]],P[indexs[1]]-P[indexs[3]]))
for i in range(5):
    alpha_list[i]=acos(fabs(normal_vector[i][2])/np.linalg.norm(normal_vector[i]))
normal_vector=np.array(normal_vector)
# aver_vector=normal_vector.sum(axis=0)/5
# aver_alpha=acos(fabs(aver_vector[2])/np.linalg.norm(aver_vector))
print("法向量分别为：{}".format(normal_vector))
print("坡角（弧度制）分别为：{}".format(alpha_list))
print("坡角（角度制）分别为：{}".format(alpha_list*180/pi))
# print(aver_vector)
# print(degrees(aver_alpha))

#将每一块近似为正方形，计算每一块中距中心点最大的距离
distance=[[],[],[],[],[]]
for i in range(5):
    for j in range(len(P_cla[i])):
        distance[i].append(sqrt(np.sum((np.array(P_cla[i][j])-clf.Pcen[i])[:2]**2)))
    distance[i]=max(distance[i])
print(distance)
#计算每一块近似长度
length=np.array(distance)*sqrt(2)
print("每一块近似长度分别为：{}".format(length))

#计算每一块的测线距离
theta=radians(120)
D_list=-clf.Pcen[:,2]
x1_list=D_list*tan(theta/2)-length/2*np.cos(np.full(5,theta/2)+alpha_list)/(cos(theta/2)*np.cos(alpha_list))
# print(x1_list)
C1_list=sin(theta/2)*np.cos(alpha_list)/np.cos(np.full(5,theta/2)-alpha_list)
C2_list=sin(theta/2)*(1/np.cos(np.full(5,theta/2)+alpha_list)+1/np.cos(np.full(5,theta/2)-alpha_list))
# print(C1_list)
# print(C2_list)
x_list=[[],[],[],[],[]]
d_list=[[],[],[],[],[]]
for i in range(5):
    x_k=x1_list[i]
    while (D_list[i]-tan(alpha_list[i])*x_k)*C1_list[i]<length[i]/2-x_k:
        x_list[i].append(x_k)
        d_k = 0.9 * C2_list[i] * (D_list[i]-tan(alpha_list[i])*x_k) / (1 + 0.9 * C2_list[i] * tan(alpha_list[i]))
        d_list[i].append(d_k)
        x_k = x_k + d_k
    del d_list[i][-1]
for i in range(5):
    print("第{}块共{}条测线，测线距离为：{}，测线坐标为：{}".format(i+1,len(x_list[i]),d_list[i],x_list[i]))

#计算测线总长度
totallength=0
for i in range(5):
    totallength+=len(x_list[i])*length[i]
print('测线总长度为{}米'.format(totallength))

normal_vector=np.array(normal_vector)
#计算所需夹角v
v_list=np.arctan(np.abs(normal_vector[:,1]/normal_vector[:,0]))
# print(v_list)
#构建旋转矩阵
v_rotate=[0,0,0,0,0]
#测线坐标
xy_list=[[],[],[],[],[]]
#测线坐标系转换坐标
xyztrans_list=[[],[],[],[],[]]
for i in range(5):
    v_rotate[i]=np.array([[cos(v_list[i]),sin(v_list[i])],[-sin(v_list[i]),cos(v_list[i])]])
    x_list[i]=np.array(x_list[i])
    for j in range(len(x_list[i])):
        xy_list[i].append([x_list[i][j],0])
    xy_list[i]=np.array(xy_list[i])
    for k in range(len(xy_list[i])):
        xyztrans_list[i].append(np.dot(v_rotate[i],xy_list[i][k])+clf.Pcen[i][:2])
    xyztrans_list[i]=np.array(xyztrans_list[i])
    xyztrans_list[i]=np.insert(xyztrans_list[i],2,np.zeros(len(xyztrans_list[i])),axis=1)

Q=[]
for i in range(201):
    for j in range(251):
        Q.append([XX[i],Y[j],0])
Q=np.array(Q)
Q[:,:2]=Q[:,:2]*1852
tree2 = spt.cKDTree(data=Q)
xyz_minlist=[[],[],[],[],[]]
for i in range(5):
    for Point in xyztrans_list[i]:
        distances, indexs = tree.query(Point, k=2)
        xyz_minlist[i].append(Q[indexs[1]])
    xyz_minlist[i]=np.array(xyz_minlist[i])
D_minlist=[[],[],[],[],[]]
for i in range(5):
    for k in range(len(xyztrans_list[i])):
        D_minlist[i].append(P[np.argwhere(P[:,:2]==xyz_minlist[i][k][:2])[0][0]][2])
    D_minlist[i]=np.array(D_minlist[i])
y_list=[]
for i in range(5):
    d_list[i]=np.array(d_list[i])
    y_list.append(np.ones(len(d_list[i]))-d_list[i]/(C2_list[i]*np.abs(D_minlist[i][1:])))


#计算覆盖率超过20%的测线长度
totalge20=0
for i in range(5):
    for j in range(len(y_list[i])):
        if y_list[i][j]>0.2:
            totalge20+=length[i]

print('超过20%的测线长度：{}'.format(totalge20))



#绘制聚类分析板块图像
color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
ax = plt.subplot(111, projection='3d')
f = open('means-output.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)

for p in range(0,num):
    y=y_pred[p]
    csv_writer.writerow([y])
    ax.scatter((P[p, 0]), (P[p, 1]), (P[p, 2]), c=color[(int(y))])
f.close()

plt.show()
