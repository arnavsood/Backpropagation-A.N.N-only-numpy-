# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:34:31 2018

@author: arnav
"""
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)


def sigmoid(a):
    return 1/(1+np.exp(-a))

def softmax(a):
    expa=np.exp(a)
    return expa/np.sum(expa,axis=1,keepdims=True)    

def cost(T,Y):
    total=T * np.log(Y)
    return total.sum()



def forward(x,w1,b1,w2,b2):
    a1=x.dot(w1)+b1
    z1=sigmoid(a1)
    a2=z1.dot(w2)+b2
    z2=softmax(a2)
    return z2,z1


def der_b2(T,Y):
    return (T-Y).sum(axis=0)


def der_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
                

def der_w2(Z,T,Y):
    der=Z.T.dot(T-Y)
    return der

def der_w1(X,Z,T,Y,W2):
    dz=(T-Y).dot(W2.T)*Z*(1-Z)
    der=X.T.dot(dz)
    return der


def back_prop(epochs,X,w1,b1,w2,b2,T,lr):
    for epoch in range(epochs):
        output,hidden=forward(X,w1,b1,w2,b2)
        w2+=lr*der_w2(hidden,T,output)
        b2+=lr*der_b2(T,output)
        w1+=lr*der_w1(X,hidden,T,output,w2)
        b1+=lr*der_b1(T,output,w2,hidden)
    return w1,b1,w2,b2 

def ANN(D,M,K,X,T,epochs):
    w1=np.random.randn(D,M)
    w2=np.random.randn(M,K)
    b1=np.random.randn(M)
    b2=np.random.randn(K)
    lr=0.0001
    return back_prop(epochs,X,w1,b1,w2,b2,T,lr)
  
def accuracy(y,ypred):
    return np.mean(y==ypred)


def predict(X,w1,b1,w2,b2):
    y_dummy,_=forward(X,w1,b1,w2,b2)
    return np.argmax(y_dummy,axis=1)




def main():
    Nclass=500
    D=2
    M=3
    K=3
    X1 = np.random.randn(Nclass, D) + np.array([0, -3])
    X2 = np.random.randn(Nclass, D) + np.array([3, 3])
    X3 = np.random.randn(Nclass, D) + np.array([-3, 3])
    X = np.vstack([X1, X2, X3])
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y) 
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1  
        
        
    w1,b1,w2,b2=ANN(D,M,K,X,T,10000)
    
    ypred=predict(X,w1,b1,w2,b2)
    print(accuracy(Y,ypred))
    
    
    
    
if __name__ == '__main__':
    main()