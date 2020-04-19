import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
tic = time.time()

h1 = 0.1
h2 = 0.05
M = int(1/h1) + 1
N = int(1/h2) + 1

x_train = np.linspace(0,1,M)
y_train = np.linspace(0,1,N)

matr = np.zeros([M,N,2])#将所有网格点存储在矩阵matrix上，每一行代表一个网格点的坐标
for i in range(M):
    for j in range(N):
        matr[i,j,0] = x_train[i]
        matr[i,j,1] = y_train[j]
ma = matr.reshape(-1,2)
print(ma.shape)
ma_in = matr[1:-1,1:-1].reshape(-1,2)#将所有网格点存储在矩阵matrix上，每一行代表一个网格点的坐标
print(ma_in.shape)
ma_b = np.concatenate([matr[0],matr[-1],matr[:,0][1:-1],matr[:,-1][1:-1]],0)
print(ma_b.shape)
def f(x,y):#微分方程的右边函数f
    return - (2*np.pi**2)*np.exp(np.pi*(x + y))*np.sin(np.pi*(x + y))
def u_accuracy(x,y):
    return np.exp(np.pi*(x + y))*np.sin(np.pi*x)*np.sin(np.pi*y)
def u_boundary(x,y):
    return 0*x*y

right_in = f(ma_in[:,0],ma_in[:,1]).reshape(-1,1)
right_b = u_boundary(ma_b[:,0],ma_b[:,1]).reshape(-1,1)
print(right_in.shape,right_b.shape)

ma_min = np.array([[0,0]])
ma_max = np.array([[1,1]])


np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)
#定义tensorflow框架
prob_tf = tf.compat.v1.placeholder(tf.float32)
x_tf = tf.compat.v1.placeholder(tf.float32,shape = [None,2])
x_in_tf = tf.compat.v1.placeholder(tf.float32,shape = [None,2])
x_b_tf = tf.compat.v1.placeholder(tf.float32,shape = [None,2])

right_in_tf = tf.compat.v1.placeholder(tf.float32,shape = [None,1])
right_b_tf = tf.compat.v1.placeholder(tf.float32,shape = [None,1])
x_min_tf = tf.compat.v1.placeholder(tf.float32,shape = [1,2])
x_max_tf = tf.compat.v1.placeholder(tf.float32,shape = [1,2])
layers = [2,10,10,1]


H = 2*(x_tf - x_min_tf)/(x_max_tf - x_min_tf + 1e-7) - 1
H_in = 2*(x_in_tf - x_min_tf)/(x_max_tf - x_min_tf + 1e-7) - 1
H_b = 2*(x_b_tf - x_min_tf)/(x_max_tf - x_min_tf + 1e-7) - 1
#定义权重和偏置
def w_init(in_dim,out_dim):
    w_std = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = w_std), dtype=tf.float32)

weights = []
biases = []
num_layers = len(layers)
for i in range(0,num_layers - 1):
    w = w_init(in_dim = layers[i],out_dim = layers[i + 1])
    b = tf.Variable(tf.random.truncated_normal([1,layers[i + 1]], dtype = tf.float32), dtype=tf.float32)
    weights.append(w)
    biases.append(b)
#输出近似解
num_layers = len(weights)#比layers长度少1
for i in range(0,num_layers - 1):
    w = weights[i]
    b = biases[i]
    E = tf.eye(tf.shape(w)[0],tf.shape(w)[1]) 
    tf.nn.dropout(w,rate = prob_tf)
    H_in = tf.tanh(tf.add(tf.matmul(H_in,w), b)) + tf.matmul(H_in,E)
    H_b = tf.tanh(tf.add(tf.matmul(H_b,w), b)) + tf.matmul(H_b,E)
    H = tf.tanh(tf.add(tf.matmul(H,w), b)) + tf.matmul(H,E)
W = weights[-1]
b = biases[-1]
u_in = tf.add(tf.matmul(H_in,W),b)
u_b = tf.add(tf.matmul(H_b,W),b)
u = tf.add(tf.matmul(H,W),b)
#定义损失函数
u_x = tf.gradients(u_in,x_in_tf)[0][:,0]
u_y = tf.gradients(u_in,x_in_tf)[0][:,1]
u_xx = tf.gradients(u_x,x_in_tf)[0][:,0]
u_yy = tf.gradients(u_y,x_in_tf)[0][:,1]

loss_in = 0.5*tf.reduce_mean(tf.square(u_x) + tf.square(u_y) - 2*u_in*right_in_tf)
#loss_in = 0.5*tf.reduce_mean((-u_xx - u_yy - 2*right_in_tf)*u_in)
loss_b = tf.reduce_mean(tf.square(u_b - right_b_tf))
belta = 5e3
loss = tf.add(loss_in,belta*loss_b)
#初始化



def plot_curve(data):#用于损失函数训练过程的可视化
    fig = plt.figure(num = 1,figsize = (4,3),dpi = None,facecolor = None,edgecolor = None,frameon = True)#编号，宽和高，分辨率，背景颜色，边框颜色，是否显示边框
    plt.plot(range(len(data)),data,color = 'blue')
    plt.legend(['value'],loc = 'upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

lr = 5*1e-3
optim = tf.compat.v1.train.AdamOptimizer(learning_rate = lr)
trainer = optim.minimize(loss)
sess = tf.compat.v1.Session()
init =  tf.compat.v1.global_variables_initializer()
sess.run(init)
train_loss = []

step = []
error = []
train_step = 6000
for i in range(train_step):
    batch = 19
    st = i*batch%len(ma_in)
    end = st+ batch
    feed = {prob_tf:0.5,x_tf:ma,x_in_tf:ma_in[st:end],x_b_tf:ma_b,right_in_tf:right_in[st:end],right_b_tf:right_b,x_min_tf:ma_min,x_max_tf:ma_max}
    
    sess.run(trainer,feed_dict = feed)
    if i%600 == 0:
        print('train_step = {},loss = {}'.format(i,sess.run(loss,feed_dict = feed)))
        train_loss.append(sess.run(loss,feed_dict = feed))
        feed_val = {x_tf:ma,x_in_tf:ma_in,x_b_tf:ma_b,right_in_tf:right_in,right_b_tf:right_b,x_min_tf:ma_min,x_max_tf:ma_max}
        u_pred = sess.run(u,feed_dict = feed_val).reshape(M,N)
        x,y = np.meshgrid(x_train,y_train)
        error_square = ((u_pred - u_accuracy(x,y).T)**2).sum()/(u_accuracy(x,y)**2 + 1e-7).sum()
        error_step = np.sqrt(error_square)
        error.append(error_step)
        step.append(i)
        print(error_step)
        
        
toc = time.time()
plot_curve(train_loss)
print(toc - tic)

feed = {prob_tf:0.5,x_tf:ma,x_in_tf:ma_in,x_b_tf:ma_b,right_in_tf:right_in,right_b_tf:right_b,x_min_tf:ma_min,x_max_tf:ma_max}
u_pred = sess.run(u,feed_dict = feed).reshape(M,N)
print(type(u_pred),u_pred.shape)
x,y = np.meshgrid(x_train,y_train)
print(type(u_accuracy(x,y)),u_accuracy(x,y).shape)
error_square = ((u_pred - u_accuracy(x,y).T)**2).sum()/(u_accuracy(x,y)**2 + 1e-7).sum()
error = np.sqrt(error_square)
print(error)
plt.subplot(2,1,1)
x,y = np.meshgrid(x_train,y_train)
plt.contourf(x,y,u_accuracy(x,y),40,cmap = 'Blues')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('accuracy solution')
plt.subplot(2,1,2)
x,y = np.meshgrid(x_train,y_train)
plt.contourf(x,y,u_pred.T,40,cmap = 'Blues')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('numerical solution')
