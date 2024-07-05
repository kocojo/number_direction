import numpy as np
import cv2
soft = lambda x: (1/(1+np.exp(-x)))
def anh(file_path):
    result=[]
    img=cv2.imread(file_path,0)
    x,y=img.shape
    for i in range(x):
        for j in range(y):
            result.append((255-img[i][j])/255)
    return result
def setup_b(a=[784,64,16,10]):
    b=[]
    for i in range(len(a)-1):
        b.append(np.zeros((a[i+1],1)))
    with open("result1.txt","r",encoding="utf8") as file:
        for i in b:
            for j in range(len(i)):
                for k in range(len(i[j])):
                    i[j][k]=float(file.readline())
    return b
def setup_weight(a=[ 784, 64, 16,10]):
    b=[]
    for i in range(len(a)-1):
        b.append(np.zeros((a[i+1],a[i])))
    with open("results.txt","r",encoding="utf8") as file:
        for i in b:
            for j in range(len(i)):
                for k in range(len(i[j])):
                    i[j][k]=float(file.readline())
    return b
def coputing(a,w,b):
    c=[a]
    for index,i in enumerate(w):
        k = b[index]
        sd = i@a
        a = soft(sd+k)
        c.append(a)
    return c
w=setup_weight()
b=setup_b()
a=coputing(np.array([anh("41.png")]).T,w,b)[-1]
krew=[0,1,2,3,4,5,6,7,8,9]
krew.sort(key=lambda x:a[x],reverse=True)
print(krew)
print(a)
