from math import *

import numpy as np
import pandas as pd


#设海域中心点深度 D ，测线距中心点距离 x ，测线距离 d ，坡面倾角 a ,开角 t
#此处角为角度制
def f(D,x,a,t,d):
    """函数f：计算距中心点距离为x的海水深度 h
       计算海水深度为 h 时，开角为 t 的光束覆盖宽度 W
       计算与前一条测线的重叠率 y"""
    alpha=a*pi/180
    theta=t*pi/180
    h=D-x*tan(alpha)
    W=h*sin(theta/2)*(1/cos(theta/2+alpha)+1/cos(theta/2-alpha))
    y=1-d/W
    # return print('距中心点距离为{}的海水深度 h={}\n光束覆盖宽度 W={}\n前一条测线的重叠率 y={}'.format(x,h,W,y))
    return h,W,y
#定义data1来存储result1
data1=np.zeros((3,9))
for x in np.linspace(-800,800,9):
    data1[0][int((x/200)+4)],data1[1][int((x/200)+4)],data1[2][int((x/200)+4)]=f(70,x,1.5,120,200)
data_1=pd.DataFrame(data1)
data_1.to_excel('demoresult1.xlsx') ##得到结果



