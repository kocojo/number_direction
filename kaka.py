import numpy as np
from numpy import random
import cv2
import csv
import random as rd
soft = lambda x:(1/(1+np.exp(-x)))
dsoft= lambda x:soft(x)*(1-soft(x))
def setup_b(a=[784,64,16,10]):
    b=[]
    for i in range(len(a)-1):
        b.append(random.random((a[i+1],1))/a[i+1])
    return b
def setup_weight(a=[784,64,16,10]):
    b=[]
    for i in range(len(a)-1):
        b.append(random.random((a[i+1],a[i]))/a[i])
    return b
def coputing(a,w,b):
    c=[a]
    for index,i in enumerate(w):
        k=b[index]
        sd=i@a
        a = soft(sd+k)
        c.append(a)
    return c
def anh(file_path):
            result = []
            img = cv2.imread(file_path, 0)
            x, y = img.shape
            for i in range(x):
                for j in range(y):
                    result.append((255 - img[i][j])/255)
            return result
def z(a,w,b):
    c = []
    for index, i in enumerate(w):
        c.append(i @ a + b[index])
        a = soft(i @ a + b[index])
    return c
def jacobian_w(a0,y):
    a = coputing(a0, w, b)
    j_w=[]
    z1 = z(a0, w, b)
    j_a = 2 * (a[-1]-y)
    for i in range(1, len(w) + 1):
        dz=j_a*dsoft(z1[len(b)-i])
        dw = dz@a[len(a)-i-1].T
        j_w.append(dw)
        k = w[len(w) - i]
        j_a = (dz.T @ k).T
    return j_w
def jacobian_b(a0,y):
    a=coputing(a0,w,b)
    j_b=[]
    z1=z(a0,w,b)
    j_a=2*(a[-1]-y)
    for i in range(1,len(b)+1):
        dz=j_a*dsoft(z1[len(b)-i])
        j_b.append(dz)
        k=w[len(w)-i]
        j_a=(dz.T@k).T
    return j_b
def te():
        tong=0
        result=test_list
        for index,i in enumerate(result):
            a2 = np.array([i[0]]).T
            y1=coputing(a2,w,b)[-1]
            krew = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            krew.sort(key=lambda x: y1[x], reverse=True)
            krew1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            krew1.sort(key=lambda x: i[1][x], reverse=True)
            if krew[0] == krew1[0]:
               tong=tong+1
        return tong/len(test_list)
def generate_random_0_1():
    return 0 if rd.random() < 0.8 else 1
def re(a0,y):
    jw=jacobian_w(a0,y)
    jb=jacobian_b(a0,y)
    global w
    global b
    for i in range(len(b)):
        b[i]=b[i]-0.05*jb[len(b)-i-1]
    for i in range(len(w)):
        w[i] = w[i] - 0.05*jw[len(b) - i - 1]
def data():
        result = []
        y = []
        with open("data/mnist_train.csv", "r", encoding="utf8") as file:
            train = csv.reader(file)
            for line in train:
                b = []
                y1 = np.zeros((10, 1))
                y1[int(line[0])][0] = 1
                for i in line[1:len(line)]:
                    b.append(int(i) / 255)
                if generate_random_0_1() == 1:
                    y.append((b,y1))
                else:
                    result.append((b,y1))
        return y,result


def data1():
    result = []
    y = []
    with open("data/mnist_train.csv", "r", encoding="utf8") as file:
        train = csv.reader(file)
        for line in train:
            b = []
            y1 = np.zeros((10, 1))
            y1[int(line[0])][0] = 1
            y.append(y1)
            for i in line[1:len(line)]:
                b.append(int(i) / 255)
            result.append(b)
    return y, result
if __name__ == '__main__':
    test_list ,res=data()
    w=setup_weight()
    b=setup_b()
    print(te())
    while True:
        for index,i in enumerate(res):
            a2 = np.array([i[0]]).T
            re(a2, i[1])
        print(te())
        n=input()
        if n=="n":
            with open("d:/nhandienchuso/results.txt","w",encoding="utf8") as file:
                for i in w:
                    for j in range(len(i)):
                       for k in range(len(i[j])):
                           file.write(str(i[j][k])+"\n")
            with open("d:/nhandienchuso/result1.txt", "w", encoding="utf8") as file:
                for i in b:
                    for j in range(len(i)):
                        for k in range(len(i[j])):
                            file.write(str(i[j][k]) + "\n")

            break
