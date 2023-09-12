from math import *
import numpy as np
import pandas as pd
#设海域中心点深度 D ，测量船距海域中心点处的距离 x（海里），坡面倾角 a ,开角 t ,测线方向夹角 b
#此处角为角度制
def g(D,a,t,x,b):
    """函数g： 计算测线方向夹角为 b ，测量船延测线方向距海域中心点处的距离为 x（海里）时的海水深度 h（米）
       计算此时的开角为 t 的光束覆盖宽度 W"""
    alpha = a * pi / 180
    theta = t * pi / 180
    beta = b * pi / 180
    X = 1852 * x
    h=D+X*tan(alpha)*cos(beta)
    v=asin(fabs(tan(beta)*tan(alpha)/sqrt(1+tan(beta)**2+tan(alpha)**2*tan(beta)**2)))
    W=h*sin(theta)*cos(v)/(cos(theta/2)**2*cos(v)**2-sin(theta/2)**2*sin(v)**2)
    return W
#定义data2来存储所需result2
data2=np.zeros((8,8))

for b in np.linspace(0,315,8):
    for x in np.linspace(0,2.1,8):
        data2[int(b/45)][int(x/0.3)]=g(120, 1.5, 120, x, b)
data_2=pd.DataFrame(data2)
data_2.to_excel('demoresult2.xlsx') #得到结果

