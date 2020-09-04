import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def UU(X, order):
    temp = 10*(X[:,0]+X[:,1])**2 + (X[:,0]-X[:,1])**2 + 0.5
    if order[0]==0 and order[1]==0:
        return torch.log(temp)
    if order[0]==1 and order[1]==0:
        return temp**(-1) * (20*(X[:,0]+X[:,1]) + 2*(X[:,0]-X[:,1]))
    if order[0]==0 and order[1]==1:
        return temp**(-1) * (20*(X[:,0]+X[:,1]) - 2*(X[:,0]-X[:,1]))
    if order[0]==2 and order[1]==0:
        return - temp**(-2) * (20*(X[:,0]+X[:,1])+2*(X[:,0]-X[:,1])) ** 2 \
               + temp**(-1) * (22)
    if order[0]==0 and order[1]==2:
        return - temp**(-2) * (20*(X[:,0]+X[:,1])-2*(X[:,0]-X[:,1])) ** 2 \
               + temp**(-1) * (22)
'''
def UU(X,order):
    if order == [0,0]:
        return X[:,0]**2 + X[:,1]**2
    if order == [2,0]:
        return 2*torch.ones(X.shape[0])
    if order == [0,2]:
        return 2*torch.ones(X.shape[0])
'''
def FF(X):
    return -UU(X,[2,0]) - UU(X,[0,2])

class TESET():
    def __init__(self, bounds, nx):
        self.bounds = bounds
        self.nx = nx
        self.hx = [(self.bounds[0,1]-self.bounds[0,0])/self.nx[0],
                   (self.bounds[1,1]-self.bounds[1,0])/self.nx[1]]

        self.size = self.nx[0]*self.nx[1]
        self.X = torch.zeros(self.size,2)
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.X[m,0] = self.bounds[0,0] + (i+0.5)*self.hx[0]
                self.X[m,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
                m = m+1

        self.u_acc = UU(self.X[:,0:2],[0,0]).reshape(self.size,1)
class FENET():
    def __init__(self,bounds,nx):
        self.dim = 2
        self.bounds = bounds
        self.hx = [(bounds[0,1] - bounds[0,0])/nx[0],
                  (bounds[1,1] - bounds[1,0])/nx[1]]
        self.nx = nx
        self.gp_num = 2
        self.gp_pos = [(1 - np.sqrt(3)/3)/2,(1 + np.sqrt(3)/3)/2]
        
        self.Node()
        self.Unit()
        self.matrix()
        
    def Node(self):#生成网格点(M + 1)*(N + 1)
        self.Nodes_size = (self.nx[0] + 1)*(self.nx[1] + 1)
        self.Nodes = torch.zeros(self.Nodes_size,self.dim)
        m = 0
        for i in range(self.nx[0] + 1):
            for j in range(self.nx[1] + 1):
                self.Nodes[m,0] = self.bounds[0,0] + i*self.hx[0]
                self.Nodes[m,1] = self.bounds[1,0] + j*self.hx[1]
                m = m + 1
    def Unit(self):#生成所有单元，单元数目(M*N)
        self.Units_size = self.nx[0]*self.nx[1]
        self.Units_Nodes = np.zeros([self.Units_size,4],np.int)#每个单元有4个点，记录这4个点的编号
        self.Units_Int_Points = torch.zeros(self.Units_size,#划分成M*N个小区域，每个区域有4个积分点
                                            self.gp_num*self.gp_num,self.dim)
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.Units_Nodes[m,0] = i*(self.nx[1] + 1) + j
                self.Units_Nodes[m,1] = i*(self.nx[1] + 1) + j + 1
                self.Units_Nodes[m,2] = (i + 1)*(self.nx[1] + 1) + j
                self.Units_Nodes[m,3] = (i + 1)*(self.nx[1] + 1) + j + 1
                n = 0
                for k in range(self.gp_num):
                    for l in range(self.gp_num):
                        self.Units_Int_Points[m,n,0] = \
                            self.bounds[0,0] + (i + self.gp_pos[k])*self.hx[0]
                        self.Units_Int_Points[m,n,1] = \
                            self.bounds[1,0] + (j + self.gp_pos[l])*self.hx[1]
                        n = n + 1
                m = m + 1
    def matrix(self):
        self.A = torch.zeros(self.Nodes_size,self.Nodes_size)#self.Nodes_size = (M+ 1)*(N + 1)
        for m in range(self.Units_size):#self.Units_size = M*N，第m个区域单元的4个积分点
            for k in range(4):
                ind0 = self.Units_Nodes[m,k]#self.Units_Nodes = [M*N,4],第m个区域中第k个网格点，第k个基函数编号
                for l in range(4):
                    ind1 = self.Units_Nodes[m,l]#self.Units_Nodes = [M*N,4],第m个区域中第l网格点，第k个基函数编号
                    #第m个区域上，两个基函数梯度的乘积的积分a(u,v)
                    self.A[ind0,ind1] += self.Int_basic_basic(ind0,ind1,m)
        #边界上的网格点特殊处理
        #左右边界
        for j in range(self.nx[1] + 1):#刚才设置过A的元素，现在需要把对应边界上的元素覆盖掉
            ind = j
            self.A[ind,:] = torch.zeros(1,self.Nodes_size)#self.Nodes_size = (M + 1)*(N + 1)
            self.A[ind,ind] = 1
            ind = self.nx[0]*(self.nx[1] + 1) + j
            self.A[ind,:] = torch.zeros(1,self.Nodes_size)
            self.A[ind,ind] = 1
        #上下边界
        for i in range(1,self.nx[0]):
            ind = i*(self.nx[1] + 1)
            self.A[ind,:] = torch.zeros(1,self.Nodes_size)
            self.A[ind,ind] = 1
            ind = (i + 1)*(self.nx[0] + 1) - 1
            self.A[ind,:] = torch.zeros(1,self.Nodes_size)
            self.A[ind,ind] = 1
    def right(self,UU,FF):
        self.b = torch.zeros(self.Nodes_size,1)#self.Nodes_size = (M + 1)*(N + 1)
        for m in range(self.Units_size):#self.Units_size = M*N
            for k in range(4):
                ind = self.Units_Nodes[m,k]#第m个单元区域内第k个网格点，第k个基函数的编号
                self.b[ind] += self.Int_F_basic(ind,m,FF)
        #边界上的网格点需要特殊处理
        #左右边界
        for j in range(self.nx[1] + 1):# b = [(nx[0] + 1)*(nx[1] + 1),1]
            ind = j#b[i*(nx[1] + 1) + j],b按列存储，这是第一列
            self.b[ind] = UU(self.Nodes[ind:ind + 1,:],[0,0])#self.Nodes存储(M + 1)*(N + 1)个网格点
            ind  = self.nx[0]*(self.nx[1] + 1) + j#这是最后一列元素
            self.b[ind] = UU(self.Nodes[ind:ind + 1,:],[0,0])
        #上下边界
        for i in range(1,self.nx[0]):
            ind = i*(self.nx[1] + 1)#每一列中取第一个元素就是下边界
            self.b[ind] = UU(self.Nodes[ind:ind + 1],[0,0])#self.Nodes存储(M + 1)*(N + 1)个网格点
            ind = (i + 1)*(self.nx[1] + 1) - 1#每一列中取最后一个元素就是上边界
            self.b[ind] = UU(self.Nodes[ind:ind + 1],[0,0])#self.Nodes存储(M + 1)*(N + 1)个网格点
                    
    def phi(self,X,order):#[-1,1]*[-1,1]，在原点取值为1，其他网格点取值为0的基函数
        ind00 = (X[:,0] >= -1);ind01 = (X[:,0] >= 0);ind02 = (X[:,0] >= 1)
        ind10 = (X[:,1] >= -1);ind11 = (X[:,1] >= 0);ind12 = (X[:,1] >= 1)
        if order == [0,0]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,0])*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(1 - X[:,0])*(1 + X[:,1]) + \
                    (ind00*~ind01*ind11*~ind12).float()*(1 + X[:,0])*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).float()*(1 - X[:,0])*(1 - X[:,1])
        if order == [1,0]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(-(1 + X[:,1])) + \
                    (ind00*~ind01*ind11*~ind12).float()*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).float()*(-(1 - X[:,1]))
        if order == [0,1]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,0]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(1 - X[:,0]) + \
                    (ind00*~ind01*ind11*~ind12).float()*(-(1 + X[:,0])) + \
                    (ind01*~ind02*ind11*~ind12).float()*(-(1 - X[:,0]))
    def basic(self,X,order,i):#根据网格点的存储顺序，遍历所有网格点，取基函数
        temp = (X - self.Nodes[i,:])/torch.tensor([self.hx[0],self.hx[1]])
        if order == [0,0]:
            return self.phi(temp,order)
        if order == [1,0]:
            return self.phi(temp,order)/self.hx[0]
        if order == [0,1]:
            return self.phi(temp,order)/self.hx[1]
        
    # 计算两个基函数梯度的内积，并积分
    # u_ind：计算积分的单元(同样是索引)
    def Int_basic_basic(self,i,j,u_ind):#表示第i,j个基函数的梯度的乘积，以及在第u_ind个区域的积分
        X0 = self.Units_Int_Points[u_ind,:,:]
        basic0 = torch.zeros(X0.shape)
        basic0[:,0] = self.basic(X0,[1,0],i)
        basic0[:,1] = self.basic(X0,[0,1],i)
        
        X1 = self.Units_Int_Points[u_ind,:,:]
        basic1 = torch.zeros(X1.shape)
        basic1[:,0] = self.basic(X1,[1,0],j)
        basic1[:,1] = self.basic(X1,[0,1],j)
        return ((basic0*basic1).sum(1)).mean()*self.hx[0]*self.hx[1]
    def Int_F_basic(self,i,u_ind,FF):#第i个基函数与右端项乘积，在第u_ind个单元积分
        X = self.Units_Int_Points[u_ind,:,:]
        return (FF(X)*self.basic(X,[0,0],i)).mean()*self.hx[0]*self.hx[1]
    def solve(self,UU,FF):
        self.right(UU,FF)
        self.Nodes_V,lu = torch.solve(self.b,self.A)# 求解方程组得到所有网格点的值
    def Uh(self,X):
        uh = torch.zeros(X.shape[0])
        for i in range(self.Nodes_size):#self.Nodes_size = (M + 1)*(N + 1)
            # 计算数据集 关于 基函数中心 的相对位置
            uh += self.Nodes_V[i]*self.basic(X,[0,0],i)
        return uh.view(-1,1)
def error(u_pred,u_acc):
    fenzi = ((u_pred - u_acc)**2).sum()
    fenmu = (u_acc**2 + 1e-8).sum()
    return (fenzi/fenmu)**(0.5)
def main():
    st = time.time()
    bounds = torch.tensor([[-1,2],[-1,0]]).float()
    nx_tr = [20,20]    # 网格大小
    nx_te = [100,100]  # 测试集大小
    fe_net = FENET(bounds,nx_tr)
    fe_net.solve(UU,FF)
    
    teset = TESET(bounds,nx_te)
    u_pred = fe_net.Uh(teset.X)
    ERROR = error(u_pred,teset.u_acc).item()
    ela = time.time() - st
    print('the error of FE is %.3e,use time :%.2f'%(ERROR,ela))
if __name__ == '__main__':
    main()
