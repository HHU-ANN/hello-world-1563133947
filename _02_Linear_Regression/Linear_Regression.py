# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import numpy as np
import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(X,Y,lamuda):
    X = np.mat(X)
    Y = np.mat(Y).T
    XTX=X.T*X
    rXTX=XTX+np.eye(X.shape[1])*lamuda
    new_w=rXTX.I*X.T*Y
    return new_w

def lasso(X,Y,alpha,step,item):
    X = np.mat(X)  # 404*6
    Y = np.mat(Y).T  # 404*1
    m = X.shape[0]
    w = np.ones((X.shape[1], 1))
    for i in range(1):
        y_hat=np.dot(X,w)
        w=w-step*((1/m)*X.T*(y_hat-Y)+alpha*alpha_L1(w))
    return w

def alpha_L1(w):
    n=w.shape[0]
    for i in range(n):
        if(w[i]>0):
            w[i]=1
        elif(w[i]<0):
            w[i] = -1
        else:
            w[i]=0
    return w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

# x,y=read_data('D:/Git/hello-world-1563133947/data/exp02/')
# print(ridge(x,y,0.00001))
# print(lasso(x,y,0.0001,0.0001,10))


