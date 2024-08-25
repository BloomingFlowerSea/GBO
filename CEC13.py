# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 18:25:14 2021

@Author : Hao Guo
@Email  : gho97@foxmail.com 
@School : Zhengzhou University
@Version : 1.0
@Description : None
"""

#%% 导入用到的模块
import tqdm
import time
import copy
import math
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

#%% 导入旋转偏移矩阵
global O
O = np.loadtxt('.\\input_data\\shift_data.txt')
O = O.ravel()
M2 = np.loadtxt('.\\input_data\\M_D2.txt')
M5 = np.loadtxt('.\\input_data\\M_D5.txt')
M10 = np.loadtxt('.\\input_data\\M_D10.txt')
M20 = np.loadtxt('.\\input_data\\M_D20.txt')
M30 = np.loadtxt('.\\input_data\\M_D30.txt')
M40 = np.loadtxt('.\\input_data\\M_D40.txt')
M50 = np.loadtxt('.\\input_data\\M_D50.txt')
M60 = np.loadtxt('.\\input_data\\M_D60.txt')
M70 = np.loadtxt('.\\input_data\\M_D70.txt')
M80 = np.loadtxt('.\\input_data\\M_D80.txt')
M90 = np.loadtxt('.\\input_data\\M_D90.txt')
M100 = np.loadtxt('.\\input_data\\M_D100.txt')


#%% 调用函数
def CEC13(x, fun_num):
    if fun_num == 1:
        return sphere(x)
    elif fun_num == 2:
        return ellips(x)
    elif fun_num == 3:
        return bent_cigar(x)
    elif fun_num == 4:
        return discus(x)
    elif fun_num == 5:
        return dif_powers(x)
    elif fun_num == 6:
        return rosenbrock(x)
    elif fun_num == 7:
        return schaffer_F7(x)
    elif fun_num == 8:
        return ackley(x)
    elif fun_num == 9:
        return weierstrass(x)
    elif fun_num == 10:
        return griewank(x)
    elif fun_num == 11:
        return rastrigin(x)
    elif fun_num == 12:
        return rotated_rastrigin(x)
    elif fun_num == 13:
        return step_rastrigin(x)
    elif fun_num == 14:
        return schwefel(x)
    elif fun_num == 15:
        return rotated_schwefel(x)
    elif fun_num == 16:
        return katsuura(x)
    elif fun_num == 17:
        return bi_rastrigin(x)
    elif fun_num == 18:
        return rotated_bi_rastrigin(x)
    elif fun_num == 19:
        return grie_rosen(x)
    elif fun_num == 20:
        return escaffer6(x)
    elif fun_num == 21:
        return cf01(x)
    elif fun_num == 22:
        return cf02(x)
    elif fun_num == 23:
        return cf03(x)
    elif fun_num == 24:
        return cf04(x)
    elif fun_num == 25:
        return cf05(x)
    elif fun_num == 26:
        return cf06(x)
    elif fun_num == 27:
        return cf07(x)
    elif fun_num == 28:
        return cf08(x)


#%% 辅助函数
@nb.jit(nopython=True)
def shiftfunc(x,y,O_M_flag):
    Dim = x.shape[1]
    y = x - O[O_M_flag*Dim:O_M_flag*Dim+Dim]
    return y
@nb.jit(nopython=True)
def rotatefunc(y,z,O_M_flag):
    (pop,Dim) = np.shape(y)
    z = np.zeros((pop,Dim))
    if Dim == 2:M=M2
    elif Dim == 5:M=M5
    elif Dim == 10:M=M10
    elif Dim == 20:M=M20
    elif Dim == 30:M=M30
    elif Dim == 40:M=M40
    elif Dim == 50:M=M50
    elif Dim == 60:M=M60
    elif Dim == 70:M=M70
    elif Dim == 80:M=M80
    elif Dim == 90:M=M90
    elif Dim == 100:M=M100
    else:print('维度不支持，请重新选择！')
    for i in range(pop):
        for j in range(Dim):
            for k in range(Dim):
                z[i,j] += y[i,k]*M[O_M_flag*Dim+j,k]
    return z
@nb.jit(nopython=True)
def oszfunc(x,x_osz):
    (Population_size,Dim) = np.shape(x)
    x_osz = np.zeros((Population_size,Dim))
    for i in range(0,Population_size):
        for j in range(0,Dim):
            if j == 0 or j == Dim-1:
                if x[i,j] != 0:
                    xx = math.log(abs(x[i,j]))
                if x[i,j] > 0:
                    c1,c2 = 10,7.9
                else:
                    c1,c2 = 5.5,3.1
                if x[i,j] > 0:
                    sx = 1
                elif x[i,j] == 0:
                    sx = 0
                else:
                    sx = -1
                x_osz[i,j] = sx*math.exp(xx+0.049*(math.sin(c1*xx)+math.sin(c2*xx)))
            else:
                x_osz[i,j] = x[i,j]
    return x_osz
@nb.jit(nopython=True)
def asyfunc(z,y,beta):
    (Population_size,Dim) = np.shape(z)
    for i in range(Population_size):
        for j in range(Dim):
            if z[i,j] > 0:
                y[i,j] = pow(z[i,j],1.0+beta*j/(Dim-1)*pow(z[i,j],0.5))
    return y
@nb.jit(nopython=True)
def cf_cal(x,fit,Population_size,Dim,delta,bias,cf_num):
    w = np.zeros((Population_size,cf_num))
    f = np.zeros(Population_size)
    for i in range(0,Population_size):
        w_max = 0 ; w_sum = np.zeros(Population_size)
        for j in range(0,cf_num):
            fit[:,j] += bias[j]
            for k in range(0,Dim):
                w[i,j] += pow(x[i,k] - O[j*Dim+k],2.0)
            if w[i,j] != 0:
                w[i,j] = pow(1.0/w[i,j],0.5)*math.exp(-w[i,j]/2.0/Dim/pow(delta[j],2))
            else:
                w[i,j] = 0
            if w[i,j] > w_max:
                w_max = w[i,j]
        for j in range(0,cf_num):
            w_sum[i] = w_sum[i] + w[i,j]
        if w_max == 0:
            for j in range(0,cf_num):
                w[i,j] = 1
            w_sum[i] = cf_num
        for j in range(0,cf_num):
            f[i] = f[i] + w[i,j]/w_sum[i]*fit[i,j]
    return f

# 测试函数
@nb.jit(nopython=True)
def sphere(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    f = np.zeros(pop)
    for j in range(Dim):
        f += z[:,j] * z[:,j]
    return f-1400
@nb.jit(nopython=True)
def ellips(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    y = oszfunc(z,y)
    f = np.zeros(pop)
    for j in range(Dim):
        f += pow(10,6*j/(Dim-1))*y[:,j]*y[:,j]
    return f-1300
@nb.jit(nopython=True)
def bent_cigar(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    y = asyfunc(z,y,0.5)
    z = rotatefunc(y,z,O_M_flag+1)
    f = np.zeros(pop)
    f = z[:,0] * z[:,0]
    for j in range(1,Dim):
        f += pow(10,6) * z[:,j] * z[:,j]
    return f-1200
@nb.jit(nopython=True)
def discus(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    y = oszfunc(z,y)
    f = np.zeros(pop)
    f = pow(10,6)*y[:,0]*y[:,0]
    for j in range(1,Dim):
        f += y[:,j]*y[:,j]
    return f-1100
@nb.jit(nopython=True)
def dif_powers(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim):
            f[i] += pow(abs(y[i,j]),math.floor(2+4*j/(Dim-1)))
        f[i] = pow(f[i],0.5)
    return f-1000
@nb.jit(nopython=True)
def rosenbrock(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y *= 2.048/100
    z = rotatefunc(y,z,O_M_flag)
    z += 1
    f = np.zeros(pop)
    for j in range(Dim-1):
        tmp1 = z[:,j]*z[:,j] - z[:,j+1]
        tmp2 = z[:,j] - 1.0
        f += 100*tmp1*tmp1 + tmp2*tmp2
    return f-900
@nb.jit(nopython=True)
def schaffer_F7(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    y = asyfunc(z,y,0.5)
    for j in range(0,Dim):
        z[:,j] = y[:,j] * pow(10.0,1.0*j/(Dim-1)/2.0)
    y = rotatefunc(z,y,O_M_flag+1)
    for i in range(pop):
        for j in range(0,Dim-1):
            z[i,j] = pow(y[i,j]*y[i,j]+y[i,j+1]*y[i,j+1],0.5)
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim-1):
            tmp = math.sin(50.0*pow(z[i,j],0.2))
            f[i] += pow(z[i,j],0.5) + pow(z[i,j],0.5)*tmp*tmp
        f[i] = f[i]*f[i]/(Dim-1)/(Dim-1)
    return f-800
@nb.jit(nopython=True)
def ackley(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    y = asyfunc(z,y,0.5)
    for j in range(0,Dim):
        z[:,j] = y[:,j] * pow(10.0,1.0*j/(Dim-1)/2.0)
    y = rotatefunc(z,y,O_M_flag+1)
    f = np.zeros(pop)
    sum1 = np.zeros(pop)
    sum2 = np.zeros(pop)
    for j in range(Dim):
        sum1 += y[:,j]*y[:,j]
        sum2 += np.cos(2.0*np.pi*y[:,j])
    for i in range(pop):
        sum1[i] = -0.2 * pow(sum1[i]/Dim,0.5)
        sum2[i] = sum2[i] / Dim
        f[i] = math.exp(1) - 20*math.exp(sum1[i]) - math.exp(sum2[i]) + 20.0
    return f-700
@nb.jit(nopython=True)
def weierstrass(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y * 0.5/100.0
    z = rotatefunc(y,z,O_M_flag)
    y = asyfunc(z,y,0.5)
    for j in range(0,Dim):
        z[:,j] = y[:,j] * pow(10.0,1.0*j/(Dim-1)/2.0)
    y = rotatefunc(z,y,O_M_flag+1)
    f = np.zeros(pop)
    a,b,k_max = 0.5,3.0,20
    for i in range(pop):
        for j in range(Dim):
            sum1,sum2 = 0,0
            for k in range(0,k_max):
                sum1 += pow(a,k)*math.cos(2*math.pi*pow(b,k)*(y[i,j]+0.5))
                sum2 += pow(a,k)*math.cos(2*math.pi*pow(b,k)*0.5)
            f[i] += sum1
        f[i] -= Dim*sum2
    return f-600
@nb.jit(nopython=True)
def griewank(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y * 600.0/100.0
    z = rotatefunc(y,z,O_M_flag)
    for j in range(0,Dim):
        z[:,j] = z[:,j] * pow(100.0,1.0*j/(Dim-1)/2.0)
    f = np.zeros(pop)
    for i in range(pop):
        sum1,tmp1 = 0,1
        for j in range(Dim):
            sum1 += z[i,j]*z[i,j]
            tmp1 *= math.cos(z[i,j]/pow(j+1.0,0.5))
        f[i] = 1.0 + sum1/4000.0 - tmp1
    return f-500
@nb.jit(nopython=True)
def rastrigin(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y * 5.12/100
    z = y.copy()
    y = oszfunc(z,y)
    z = asyfunc(y,z,0.2)
    y = z.copy()
    for j in range(0,Dim):
        y[:,j] *= pow(10,1.0*j/(Dim-1)/2)
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim):
            f[i] += y[i,j]*y[i,j] - 10*math.cos(2*math.pi*y[i,j]) + 10
    return f-400
@nb.jit(nopython=True)
def rotated_rastrigin(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y * 5.12/100
    z = rotatefunc(y,z,O_M_flag)
    y = oszfunc(z,y)
    z = asyfunc(y,z,0.2)
    y = rotatefunc(z,y,O_M_flag+1)
    for j in range(0,Dim):
        y[:,j] *= pow(10,1.0*j/(Dim-1)/2)
    z = rotatefunc(y,z,O_M_flag)
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim):
            f[i] += z[i,j]*z[i,j] - 10*math.cos(2*math.pi*z[i,j]) + 10
    return f-300
@nb.jit(nopython=True)
def step_rastrigin(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y*5.12/100
    z = rotatefunc(y,z,O_M_flag)
    for i in range(pop):
        for j in range(0,Dim):
            if abs(z[i,j]) > 0.5:
                z[i,j] = math.floor(2*z[i,j]+0.5)/2
    y = oszfunc(z,y)
    z = asyfunc(y,z,0.2)
    y = rotatefunc(z,y,O_M_flag+1)
    for j in range(0,Dim):
        y[:,j] *= pow(10,1.0*j/(Dim-1)/2)
    z = rotatefunc(y,z,O_M_flag)
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim):
            f[i] += z[i,j]*z[i,j] - 10*math.cos(2*math.pi*z[i,j]) + 10
    return f-200
@nb.jit(nopython=True)
def schwefel(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y * 1000.0/100.0
    z = y.copy()
    for j in range(0,Dim):
        y[:,j] = z[:,j] * pow(10,1.0*j/(Dim-1)/2.0)
    z = y + 4.209687462275036e+002
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim):
            if z[i,j] > 500:
                f[i] -= (500 - z[i,j]%500)*math.sin(pow(500 - z[i,j]%500,0.5))
                tmp = (z[i,j] - 500.0)/100
                f[i] += tmp*tmp/Dim
            elif z[i,j] < -500:
                f[i] -= (-500 + abs(z[i,j])%500)*math.sin(pow(500-abs(z[i,j])%500,0.5))
                tmp = (z[i,j] + 500)/100
                f[i] += tmp*tmp/Dim
            else:
                f[i] -= z[i,j]*math.sin(pow(abs(z[i,j]),0.5))
        f[i] += 4.189828872724338e+002*Dim
    return f-100  
@nb.jit(nopython=True)
def rotated_schwefel(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y = y * 1000.0/100.0
    z = rotatefunc(y,z,O_M_flag)
    for j in range(0,Dim):
        y[:,j] = z[:,j] * pow(10,1.0*j/(Dim-1)/2.0)
    z = y + 4.209687462275036e+002
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim):
            if z[i,j] > 500:
                f[i] -= (500 - z[i,j]%500)*math.sin(pow(500 - z[i,j]%500,0.5))
                tmp = (z[i,j] - 500.0)/100
                f[i] += tmp*tmp/Dim
            elif z[i,j] < -500:
                f[i] -= (-500 + abs(z[i,j])%500)*math.sin(pow(500-abs(z[i,j])%500,0.5))
                tmp = (z[i,j] + 500)/100
                f[i] += tmp*tmp/Dim
            else:
                f[i] -= z[i,j]*math.sin(pow(abs(z[i,j]),0.5))
        f[i] += 4.189828872724338e+002*Dim
    return f+100  
@nb.jit(nopython=True)
def katsuura(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y *= 5.0/100.0
    z = rotatefunc(y,z,O_M_flag)
    for j in range(0,Dim):
        z[:,j] *= pow(100.0,1.0*j/(Dim-1)/2.0)
    y = rotatefunc(z,y,O_M_flag+1)
    f = np.ones(pop)
    tmp3 = pow(1.0*Dim,1.2)
    for i in range(pop):
        for j in range(Dim):
            temp = 0
            for k in range(1,33):
                tmp1 = pow(2.0,k)
                tmp2 = tmp1 * y[i,j]
                temp += abs(tmp2-math.floor(tmp2+0.5))/tmp1
            f[i] *= pow(1.0+(j+1)*temp,10.0/tmp3)
        tmp1 = 10.0/Dim/Dim
        f[i] = f[i]*tmp1 - tmp1
    return f+200
@nb.jit(nopython=True)
def bi_rastrigin(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    tmpx = np.zeros((pop,Dim))
    mu0 = 2.5 ; d = 1.0
    s = 1.0-1.0/(2.0*pow(Dim+20.0,0.5)-8.2)
    mu1 = -pow((mu0*mu0-d)/s,0.5)
    y = shiftfunc(x,y,O_M_flag)
    y *= 10.0/100.0
    for j in range(0,Dim):
        tmpx[:,j] = 2*y[:,j]
        if O[j] < 0:
            tmpx[:,j] *= -1
    for j in range(0,Dim):
        z[:,j] = tmpx[:,j]
        tmpx[:,j] += mu0
    y = z.copy()
    for j in range(0,Dim):
        y[:,j] *= pow(100.0,1.0*j/(Dim-1)/2.0)
    f = np.ones(pop)
    for i in range(pop):
        tmp1 = 0 ; tmp2 = 0
        for j in range(Dim):
            tmp = tmpx[i,j] - mu0
            tmp1 += tmp*tmp
            tmp = tmpx[i,j] - mu1
            tmp2 += tmp*tmp
        tmp2 *= s
        tmp2 += 1.0*Dim
        tmp = 0
        for j in range(Dim):
            tmp += math.cos(2.0*math.pi*y[i,j])
        if tmp1 < tmp2:
            f[i] = tmp1
        else:
            f[i] = tmp2
        f[i] += 10.0*(Dim - tmp)
    return f+300
@nb.jit(nopython=True)
def rotated_bi_rastrigin(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    tmpx = np.zeros((pop,Dim))
    mu0 = 2.5 ; d = 1.0
    s = 1.0-1.0/(2.0*pow(Dim+20.0,0.5)-8.2)
    mu1 = -pow((mu0*mu0-d)/s,0.5)
    y = shiftfunc(x,y,O_M_flag)
    y *= 10.0/100.0
    for j in range(0,Dim):
        tmpx[:,j] = 2*y[:,j]
        if O[j] < 0:
            tmpx[:,j] *= -1
    for j in range(0,Dim):
        z[:,j] = tmpx[:,j]
        tmpx[:,j] += mu0
    y = rotatefunc(z,y,O_M_flag)
    for j in range(0,Dim):
        y[:,j] *= pow(100.0,1.0*j/(Dim-1)/2.0)
    z = rotatefunc(y,z,O_M_flag+1)
    f = np.ones(pop)
    for i in range(pop):
        tmp1 = 0 ; tmp2 = 0
        for j in range(Dim):
            tmp = tmpx[i,j] - mu0
            tmp1 += tmp*tmp
            tmp = tmpx[i,j] - mu1
            tmp2 += tmp*tmp
        tmp2 *= s
        tmp2 += 1.0*Dim
        tmp = 0
        for j in range(Dim):
            tmp += math.cos(2.0*math.pi*z[i,j])
        if tmp1 < tmp2:
            f[i] = tmp1
        else:
            f[i] = tmp2
        f[i] += 10.0*(Dim - tmp)
    return f+400
@nb.jit(nopython=True)
def grie_rosen(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    y *= 5.0/100.0
    z = rotatefunc(y,z,O_M_flag)
    z = y + 1
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim-1):
            tmp1 = z[i,j]*z[i,j] - z[i,j+1]
            tmp2 = z[i,j] - 1.0
            temp = 100.0*tmp1*tmp1 + tmp2*tmp2
            f[i] += (temp*temp)/4000.0 - math.cos(temp) + 1.0
        tmp1 = z[i,Dim-1]*z[i,Dim-1] - z[i,0]
        tmp2 = z[i,Dim-1] - 1.0
        temp = 100.0 * tmp1 *tmp1 + tmp2 *tmp2
        f[i] += (temp*temp)/4000.0 - math.cos(temp) + 1.0
    return f+500
@nb.jit(nopython=True)
def escaffer6(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    y = x.copy()
    z = x.copy()
    y = shiftfunc(x,y,O_M_flag)
    z = rotatefunc(y,z,O_M_flag)
    y = asyfunc(z,y,0.5)
    z = rotatefunc(y,z,O_M_flag+1)
    f = np.zeros(pop)
    for i in range(pop):
        for j in range(Dim-1):
            tmp1 = math.sin(pow(z[i,j]*z[i,j] + z[i,j+1]*z[i,j+1],0.5))
            tmp1 = tmp1 * tmp1
            tmp2 = 1.0+0.001*(z[i,j]*z[i,j] + z[i,j+1]*z[i,j+1])
            f[i] += 0.5 + (tmp1-0.5)/(tmp2*tmp2)
        tmp1 = math.sin(pow(z[i,Dim-1]*z[i,Dim-1]+z[i,0]*z[i,0],0.5))
        tmp1 = tmp1 * tmp1
        tmp2 = 1.0 + 0.001*(z[i,Dim-1]*z[i,Dim-1]+z[i,0]*z[i,0])
        f[i] += 0.5 + (tmp1-0.5)/(tmp2*tmp2)
    return f+600
@nb.jit(nopython=True)
def cf01(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 5
    delta = [10, 20, 30, 40, 50]
    bias = [0, 100, 200, 300, 400]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = rosenbrock(xx,O_M_flag)
    fit[:,i] += 900
    fit[:,i] = 10000.0*fit[:,i]/1e+4
    i = 1
    xx = x.copy()
    fit[:,i] = dif_powers(xx,O_M_flag+1)
    fit[:,i] += 1000
    fit[:,i] = 10000.0*fit[:,i]/1e+10
    i = 2
    xx = x.copy()
    fit[:,i] = bent_cigar(xx,O_M_flag+2)
    fit[:,i] += 1200
    fit[:,i] = 10000.0*fit[:,i]/1e+30
    i = 3
    xx = x.copy()
    fit[:,i] = discus(xx,O_M_flag+3)
    fit[:,i] += 1100
    fit[:,i] = 10000.0*fit[:,i]/1e+10
    i = 4
    xx = x.copy()
    fit[:,i] = sphere(xx,O_M_flag+4)
    fit[:,i] += 1400
    fit[:,i] = 10000.0*fit[:,i]/1e+5
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+700
@nb.jit(nopython=True)
def cf02(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 3
    delta = [20, 20, 20]
    bias = [0, 100, 200]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = schwefel(xx,O_M_flag)
    fit[:,i] += 100
    i = 1
    xx = x.copy()
    fit[:,i] = schwefel(xx,O_M_flag+1)
    fit[:,i] += 100
    i = 2
    xx = x.copy()
    fit[:,i] = schwefel(xx,O_M_flag+2)
    fit[:,i] += 100
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+800
@nb.jit(nopython=True)
def cf03(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 3
    delta = [20, 20, 20]
    bias = [0, 100, 200]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag)
    fit[:,i] -= 100
    i = 1
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag+1)
    fit[:,i] -= 100
    i = 2
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag+2)
    fit[:,i] -= 100
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+900
@nb.jit(nopython=True)
def cf04(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 3
    delta = [20, 20, 20]
    bias = [0, 100, 200]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag)
    fit[:,i] -= 100
    fit[:,i] = 1000.0*fit[:,i]/4e+3
    i = 1
    xx = x.copy()
    fit[:,i] = rotated_rastrigin(xx,O_M_flag+1)
    fit[:,i] += 300
    fit[:,i] = 1000.0*fit[:,i]/1e+3
    i = 2
    xx = x.copy()
    fit[:,i] = weierstrass(xx,O_M_flag+2)
    fit[:,i] += 600
    fit[:,i] = 1000.0*fit[:,i]/400
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+1000   
@nb.jit(nopython=True)
def cf05(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 3
    delta = [10,30,50]
    bias = [0, 100, 200]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag)
    fit[:,i] -= 100
    fit[:,i] = 1000.0*fit[:,i]/4e+3
    i = 1
    xx = x.copy()
    fit[:,i] = rotated_rastrigin(xx,O_M_flag+1)
    fit[:,i] += 300
    fit[:,i] = 1000.0*fit[:,i]/1e+3
    i = 2
    xx = x.copy()
    fit[:,i] = weierstrass(xx,O_M_flag+2)
    fit[:,i] += 600
    fit[:,i] = 1000.0*fit[:,i]/400
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+1100 
@nb.jit(nopython=True)
def cf06(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 5
    delta = [10,10,10,10,10]
    bias = [0, 100, 200, 300, 400]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag)
    fit[:,i] -= 100
    fit[:,i] = 1000.0*fit[:,i]/4e+3
    i = 1
    xx = x.copy()
    fit[:,i] = rotated_rastrigin(xx,O_M_flag+1)
    fit[:,i] += 300
    fit[:,i] = 1000.0*fit[:,i]/1e+3
    i = 2
    xx = x.copy()
    fit[:,i] = ellips(xx,O_M_flag+2)
    fit[:,i] += 1300
    fit[:,i] = 1000.0*fit[:,i]/1e+10
    i = 3
    xx = x.copy()
    fit[:,i] = weierstrass(xx,O_M_flag+3)
    fit[:,i] += 600
    fit[:,i] = 1000.0*fit[:,i]/400
    i = 4
    xx = x.copy()
    fit[:,i] = griewank(xx,O_M_flag+4)
    fit[:,i] += 500
    fit[:,i] = 1000.0*fit[:,i]/100
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+1200
@nb.jit(nopython=True)
def cf07(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 5
    delta = [10,10,10,20,20]
    bias = [0, 100, 200, 300, 400]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = griewank(xx,O_M_flag)
    fit[:,i] += 500
    fit[:,i] = 10000.0*fit[:,i]/100
    i = 1
    xx = x.copy()
    fit[:,i] = rotated_rastrigin(xx,O_M_flag+1)
    fit[:,i] += 300
    fit[:,i] = 10000.0*fit[:,i]/1e+3
    i = 2
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag+2)
    fit[:,i] -= 100
    fit[:,i] = 10000.0*fit[:,i]/4e+3
    i = 3
    xx = x.copy()
    fit[:,i] = weierstrass(xx,O_M_flag+3)
    fit[:,i] += 600
    fit[:,i] = 10000.0*fit[:,i]/400
    i = 4
    xx = x.copy()
    fit[:,i] = sphere(xx,O_M_flag+4)
    fit[:,i] += 1400
    fit[:,i] = 10000.0*fit[:,i]/1e+5
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+1300
@nb.jit(nopython=True)
def cf08(x,O_M_flag=0):
    (pop,Dim) = np.shape(x)
    cf_num = 5
    delta = [10,20,30,40,50]
    bias = [0, 100, 200, 300, 400]
    fit = np.zeros((pop,cf_num))
    i = 0
    xx = x.copy()
    fit[:,i] = grie_rosen(xx,O_M_flag)
    fit[:,i] -= 500
    fit[:,i] = 10000.0*fit[:,i]/4e+3
    i = 1
    xx = x.copy()
    fit[:,i] = schaffer_F7(xx,O_M_flag+1)
    fit[:,i] += 800
    fit[:,i] = 10000.0*fit[:,i]/4e+6
    i = 2
    xx = x.copy()
    fit[:,i] = rotated_schwefel(xx,O_M_flag+2)
    fit[:,i] -= 100
    fit[:,i] = 10000.0*fit[:,i]/4e+3
    i = 3
    xx = x.copy()
    fit[:,i] = escaffer6(xx,O_M_flag+3)
    fit[:,i] -= 600
    fit[:,i] = 10000.0*fit[:,i]/2e+7
    i = 4
    xx = x.copy()
    fit[:,i] = sphere(xx,O_M_flag+4)
    fit[:,i] += 1400
    fit[:,i] = 10000.0*fit[:,i]/1e+5
    f = cf_cal(x,fit,pop,Dim,delta,bias,cf_num)
    return f+1400