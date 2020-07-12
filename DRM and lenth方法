import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F

xa,xb = 0,1
ya,yb = 0,1

def f(x,y):
    return - np.exp(-x)*(x - 2 + y**3 + 6*y)
def u_accuracy(x,y):
    return np.exp(-x)*(x + y**3)
'''
def f(x,y):#微分方程的右边函数f
    return - (2*np.pi**2)*np.exp(np.pi*(x + y))*np.sin(np.pi*(x + y))
def u_accuracy(x,y):
    return np.exp(np.pi*(x + y))*np.sin(np.pi*x)*np.sin(np.pi*y)



def f(x,y):
    return 0*x*y
def u_accuracy(x,y):
    return x*y
'''
dim = 2
class INSET():
    def __init__(self):
        
        self.xa,self.xb = xa,xb
        self.ya,self.yb = ya,yb
        self.nx,self.ny = 20,30
        self.hx = (self.xb - self.xa)/self.nx
        self.hy = (self.yb - self.ya)/self.ny
        self.size = self.nx*self.ny
        self.X = torch.zeros(self.size,dim)
        for i in range(self.nx):
            for j in range(self.ny):
                self.X[i*self.ny + j,0] = self.xa + (i + 0.5)*self.hx
                self.X[i*self.ny + j,1] = self.ya + (j + 0.5)*self.hy
        self.u_acc = u_accuracy(self.X[:,0],self.X[:,1]).view(-1,1)
        self.right = f(self.X[:,0],self.X[:,1]).view(-1,1)
       
class BDSET():
    def __init__(self):
        self.xa,self.xb = xa,xb
        self.ya,self.yb = ya,yb
        self.nx,self.ny = 20,30
        self.hx = (self.xb - self.xa)/self.nx
        self.hy = (self.yb - self.ya)/self.ny
        self.size = 2*(self.nx + self.ny)
        self.X = torch.zeros(self.size,dim)
        
        m = 0
        for i in range(self.nx):
            self.X[m,0] = xa + (i+0.5)*self.hx
            self.X[m,1] = self.ya
            m = m+1
        for j in range(self.ny):
            self.X[m,0] = xb
            self.X[m,1] = ya + (j+0.5)*self.hy
            m = m+1
        for i in range(self.nx):
            self.X[m,0] = xa + (i+0.5)*self.hx
            self.X[m,1] = yb
            m = m+1
        for j in range(self.ny):
            self.X[m,0] = xa
            self.X[m,1] = ya + (j+0.5)*self.hy
            m = m+1
                
        self.ub = u_accuracy(self.X[:,0],self.X[:,1]).view(-1,1)
        

class TESET():
    def __init__(self):
        
        self.xa,self.xb = xa,xb
        self.ya,self.yb = ya,yb
        self.hx = 0.1
        self.hy = 0.05
        self.nx = int((self.xb - self.xa)/self.hx) + 1
        self.ny = int((self.yb - self.ya)/self.hy) + 1
        self.size = self.nx*self.ny
        self.X = torch.zeros(self.size,dim)
        for j in range(self.ny):
            for i in range(self.nx):
                self.X[j*self.nx + i,0] = self.xa + i*self.hx
                self.X[j*self.nx + i,1] = self.ya + j*self.hy
        self.u_acc = u_accuracy(self.X[:,0],self.X[:,1]).view(-1,1)
       np.random.seed(1234)
torch.manual_seed(1234)

class NETF(nn.Module):#u = netg*lenthfactor + netf，此为netf
    def __init__(self):
        super(NETF,self).__init__()
        self.fc1 = torch.nn.Linear(dim,10)
        self.fc2 = torch.nn.Linear(10,10)
        self.fc3 = torch.nn.Linear(10,1)
    def forward(self,x):
        out = torch.sin(self.fc1(x))
        out = torch.sin(self.fc2(out))
        out[:,0:dim] = out[:,0:dim] + x
        return self.fc3(out)
def lossf(netf,bdset,beta):#lenthfactor满足在边界上取值为0，为此训练netf，使得netf逼近边界函数值
    return beta*((netf.forward(bdset.X) - bdset.ub)**2).mean()
def trainf(netf,bdset,beta,optimf,epochf):
    print('train neural network f')
    loss = lossf(netf,bdset,beta)
    lossbest = loss
    print('epoch:%d,lossf:%.3e'%(0,loss.item()))
    torch.save(netf.state_dict(),'best_netf.pkl')
    cycle = 100
    for i in range(epochf):
        st = time.time()
        for j in range(cycle):
            optimf.zero_grad()
            loss = lossf(netf,bdset,beta)
            loss.backward()
            optimf.step()
        if loss < lossbest:
            lossbest = loss
            torch.save(netf.state_dict(),'best_netf.pkl')
        ela = time.time() - st
        print('epoch:%d,lossf:%.3e,time:%.2f'%((i + 1)*cycle,loss.item(),ela))
mi = torch.zeros(1,dim).float()
ma = (1e-5)*torch.ones(1,dim).float()
mi[0,0],mi[0,1] = xa,ya
ma[0,0],ma[0,1] = xb,yb

class lenthfactor():#长度因子函数
    def __init__(self,order):
        self.e = order
    def forward(self,X):
        out = 1
        for i in range(X.size(1)):
            out *= (X[:,i] - mi[:,i])*(ma[:,i] - X[:,i])
        return (out**(self.e)).view(-1,1)

class SIN(nn.Module):
    def __init__(self,order):
        super(SIN,self).__init__()
        self.e = order
    def forward(self,x):
        return torch.sin(x)**self.e
class Res(nn.Module):
    def __init__(self,input_size,output_size):
        super(Res,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,output_size),
            SIN(1),
            nn.Linear(output_size,output_size),
            SIN(1)
           
        )
        self.input = input_size
        self.output = output_size
    def forward(self,x):
        x = self.model(x) + x@torch.eye(x.size(1),self.output)
        return x
class NETG(nn.Module):#u = netg*lenthfactor + netf，此为netf，此netg逼近内部点取值
    def __init__(self):
        super(NETG,self).__init__()
        self.model = nn.Sequential(
            Res(dim,10),
            Res(10,10),
            Res(10,1)
        )
    def forward(self,x):
        out = self.model(x)
        return out

def pred(netg,netf,lenth,model):#u = netg*lenthfactor + netf
    return netg.forward(model.X)*lenth.forward(model.X) + netf.forward(model.X)

def error(netg,netf,lenth,model):
    fenzi = ((pred(netg,netf,lenth,model) - model.u_acc)**2).detach().numpy().sum()
    fenmu = (model.u_acc**2 + 1e-7).detach().numpy().sum()
    return np.sqrt(fenzi/fenmu)

def lossg(netg,netf,lenth,inset):
    inset.X.requires_grad = True
    u_in = pred(netg,netf,lenth,inset)
    ux, = torch.autograd.grad(u_in,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = torch.ones(inset.size,1))
   
    return 0.5*(ux**2).sum() - (inset.right*u_in).sum()
def traing(netg,netf,lenth,inset,optimg,epochg):
    print('train neural network g')
    ERROR,BUZHOU = [],[]
    loss = lossg(netg,netf,lenth,inset)
    lossoptimal = loss
    trainerror = error(netg,netf,lenth,inset)
    print('epoch:%d,loss:%.3e,train error:%.3e'%(0,loss.item(),trainerror))
    torch.save(netg.state_dict(),'best_netg.pkl')
    cycle = 100
    for i in range(epochg):
        st = time.time()
        for j in range(cycle):
            optimg.zero_grad()
            loss = lossg(netg,netf,lenth,inset)
            loss.backward()
            optimg.step()
        if loss < lossoptimal:
            lossoptimal = loss
            torch.save(netg.state_dict(),'best_netg.pkl')
        ela = time.time() - st
        trainerror = error(netg,netf,lenth,inset)
        ERROR.append(trainerror)
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,lossg:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,loss.item(),trainerror,ela))
    return ERROR,BUZHOU
            

netf = NETF()
netg = NETG()

lenth = lenthfactor(1)
inset = INSET()
bdset = BDSET()
teset = TESET()
optimg = torch.optim.Adam(netg.parameters(),lr = 5e-3)
beta = 5e2
optimf = torch.optim.Adam(netf.parameters(),lr = 5e-2)
epochg = 10
epochf = 10

trainf(netf,bdset,beta,optimf,epochf)
trainf(netf,bdset,beta,optimf,epochf)
NEWERROR,BUZHOU = traing(netg,netf,lenth,inset,optimg,epochg)
#然而训练过程中发现不管netg中激活函数选谁，损失函数不变，误差不变
testerror = error(netg,netf,lenth,teset)
print(testerror)
M,N = teset.nx,teset.ny
x_train = np.linspace(teset.xa,teset.xb,M)
y_train = np.linspace(teset.ya,teset.yb,N)
x,y = np.meshgrid(x_train,y_train)
u_pred = pred(netg,netf,lenth,teset).detach().numpy().reshape(N,M)
plt.contourf(x,y,u_pred,40,cmap = 'Blues')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('the new solution')
def deep_pred(net,model):#u = netg*lenthfactor + netf
    return net.forward(model.X)

def deep_error(net,model):
    fenzi = ((deep_pred(net,model) - model.u_acc)**2).detach().numpy().sum()
    fenmu = (model.u_acc**2 + 1e-7).detach().numpy().sum()
    return np.sqrt(fenzi/fenmu)

def loss(net,inset,bdset,beta):
    inset.X.requires_grad = True
    u_in = deep_pred(net,inset)
    ux, = torch.autograd.grad(u_in,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = torch.ones(inset.size,1))
    ub = deep_pred(net,bdset)
    return 0.5*(ux**2).sum() - (inset.right*u_in).sum()\
            + beta*((ub - bdset.ub)**2).sum()

def train(net,inset,bdset,beta,optim,epoch):
    print('train Deep Ritz neural network ')
    ERROR,BUZHOU = [],[]
    LOSS = loss(net,inset,bdset,beta)
    lossoptimal = LOSS
    trainerror = deep_error(net,inset)
    print('epoch:%d,loss:%.3e,train error:%.3e'%(0,LOSS.item(),trainerror))
    torch.save(net.state_dict(),'best_net.pkl')
    cycle = 100
    for i in range(epoch):
        st = time.time()
        for j in range(cycle):
            optim.zero_grad()
            LOSS = loss(net,inset,bdset,beta)
            LOSS.backward()
            optim.step()
        if LOSS < lossoptimal:
            lossoptimal = LOSS
            torch.save(net.state_dict(),'best_net.pkl')
        ela = time.time() - st
        trainerror = deep_error(net,inset)
        ERROR.append(trainerror)
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,loss:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,LOSS.item(),trainerror,ela))
    return ERROR,BUZHOU
net = NETG() 
optim = torch.optim.Adam(net.parameters(),lr = 5e-3)
epoch = epochg
OLDERROR,BUZHOU = train(net,inset,bdset,beta,optim,epoch)
testerror = deep_error(net,teset)
print(testerror)
plt.plot(BUZHOU,OLDERROR,'r*')
plt.plot(BUZHOU,OLDERROR,'b-',label = 'DRM',linewidth = 2)
plt.plot(BUZHOU,NEWERROR,'ko')
plt.plot(BUZHOU,NEWERROR,'g-',label = 'new method',linewidth = 2)
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
