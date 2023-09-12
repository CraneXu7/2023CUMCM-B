from math import *
#统一单位
D0=110
a=1.5
t=120
b=4*1852
c=2*1852
alpha=radians(a)
theta=radians(t)

#计算第一条测线横坐标 x1:
x1=D0*tan(theta/2)-b/2*cos(theta/2+alpha)/(cos(theta/2)*cos(alpha))

def D(x):
    """计算横坐标x处的海深"""
    return D0-tan(alpha)*x
#计算常系数C1，C2
C1=sin(theta/2)*cos(alpha)/cos(theta/2-alpha)
C2=sin(theta/2)*(1/cos(theta/2+alpha)+1/cos(theta/2-alpha))

#求解测线间距离
x=[] #存储测线横坐标
d=[] #存储测线间距离
x_k = x1
while D(x_k)*C1<b/2-x_k:
    x.append(x_k)
    d_k=0.9*C2*D(x_k)/(1+0.9*C2*tan(alpha))
    d.append(d_k)
    x_k=x_k+d_k
del d[-1]
print("测线坐标：{}，共{}条测线".format(x,len(x)))
print("侧线间距离：{}".format(d))