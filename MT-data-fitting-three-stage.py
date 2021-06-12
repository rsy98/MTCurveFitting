'''磁镊数据拟合程序说明
本程序是磁镊数据拟合程序,用于解释金属离子--DNA相互作用曲线的多段动力学
Author: Ran S.Y.
E-Mail: rsy98@163.com
2018.09
'''

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import  pandas as pd
import  matplotlib.patches as patches
import  matplotlib.transforms as transforms
from pylab import *
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy.optimize import leastsq #最小二乘法模块

'''数据导入
##将程序放于ExperimentData.txt数据同文件夹内或者导入时设定该文件绝对路径,如data_path='C:\data\ExperimentData.txt'
# data=np.genfromtxt('ExperimentData1.txt',delimiter=' ')
'''

data_path='ExperimentData.txt'
data = pd.read_csv(data_path,sep='\t')
index = data['index']
time = data['time(s)']
ylength = data['y(um)']
xdev = data['x-deviation(um)']
ydev = data['y-deviation(um)']

'''平台数据剪切，可用于计算外力, 放这里备用

'''

rowstart_0 = 200  #剪切起始行,人工输入
rowend_0 = 4180   #剪切末行数，人工输入

time_cut_0 = time[rowstart_0:rowend_0]
ylength_cut_0 = ylength[rowstart_0:rowend_0]
ylength_cut_0_mean = sum(i for i in ylength_cut_0)/len(ylength_cut_0)

'''第一段S型下降部分数据剪切

'''
rowstart_1 = 1    #剪切起始行,人工输入
rowend_1 = 6977   #剪切末行数，人工输入

time_cut_1 = time[rowstart_1:rowend_1]
ylength_cut_1 = ylength[rowstart_1:rowend_1]
xdev_cut_1 = xdev[rowstart_1:rowend_1]

tm = 273.7   #下降中段数据点时间,根据所画的第一张图，鼠标落在相应位置，将该点坐标读出填入
Lm = 11.6	  #下降中段数据点长度

'''第二段指数下降部分数据剪切

'''
rowstart_2 = rowend_1 + 1   #剪切起始行
rowend_2 = 22530 			#剪切末行数，人工输入

time_cut_2 = time[rowstart_2:rowend_2]
ylength_cut_2 = ylength[rowstart_2:rowend_2]
xdev_cut_2 = xdev[rowstart_2:rowend_2]

t2 = time[rowstart_2]

'''第三段指数下降部分数据剪切

'''
rowstart_3 = rowend_2 + 1 #剪切起始行
rowend_3 = 22536   #剪切末行数，人工输入
# rowend=len(index)
time_cut_3 = time[rowstart_3:rowend_3]
ylength_cut_3 = ylength[rowstart_3:rowend_3]
xdev_cut_3 = xdev[rowstart_3:rowend_3]

# L3 = 11.40
t3 = time[rowstart_3] #起始时间点
# Lf3 = 10.7


'''S型拟合函数和拟合误差

'''
def sigmoid_fitting_func(q,t):
	# tm = time[rowmiddle_1]
	# Lm = ylength[rowmiddle_1]
	# L1 = ylength_cut_0_mean
	k1,L1,Lf1 = q
	return Lf1+(L1-Lf1)/(1+((L1-Lf1)/(Lm-Lf1))*np.exp(k1*(L1-Lf1)*(t-tm)))

def error_sigmoid(q,t,y):
	return sigmoid_fitting_func(q,t)-y

'''最小二乘法拟合S型函数

'''
# L1 = ylength_cut_0_mean

q0 = [2,12,11.2] #拟合迭代初始值
para_sigmoid = leastsq(error_sigmoid,q0,args =(time_cut_1,ylength_cut_1)) #最小二乘法
k1,L1,Lf1 = para_sigmoid[0] #拟合得到的参数
print ("k1 = ",k1,'\n',"L1 = ",L1, '\n', "tm = ",tm,'\n', "Lm = ",Lm,'\n', "Lf1 = ",Lf1,'\n')

t_fit_1 = np.linspace(rowstart_1/25,rowend_1/25,100)
y_sigmoid = Lf1+(L1-Lf1)/(1+((L1-Lf1)/(Lm-Lf1))*np.exp(k1*(L1-Lf1)*(t_fit_1-tm)))

'''指数下降拟合函数和拟合误差(第二段)

'''
def exponential_fitting_func(s,t):
	k2,L2,Lf2 = s
	return Lf2+(L2-Lf2)*np.exp(-k2*(t-t2))

def error_exponential(s,t,y):
	return exponential_fitting_func(s,t)-y

'''最小二乘法拟合指数下降函数(第二段)

'''
s0 = [0.01,10,10] #拟合迭代初始值
para_exponential_2 = leastsq(error_exponential,s0,args =(time_cut_2,ylength_cut_2)) #最小二乘法
k2,L2,Lf2 = para_exponential_2[0] #拟合得到的参数
print ("k2 = ", k2, '\n', "t2 = ",t2, '\n', "L2 = ",L2, '\n', 'Lf2 = ', Lf2, '\n')

t_fit_2 = np.linspace(rowstart_2/25,rowend_2/25,100)
y_exponential_2 = Lf2+(L2-Lf2)*np.exp(-k2*(t_fit_2 - t2))


'''指数下降拟合函数和拟合误差(第三段)

'''
def exponential_fitting_func(r,t):
	k3,L3,Lf3 = r
	return Lf3+(L3-Lf3)*np.exp(-k3*(t-t3))

def error_exponential(r,t,y):
	return exponential_fitting_func(r,t)-y

'''最小二乘法拟合指数下降函数(第三段)

'''
r0 = [0.01,10,10] #拟合迭代初始值
para_exponential_3 = leastsq(error_exponential,r0,args =(time_cut_3,ylength_cut_3)) #最小二乘法
k3,L3,Lf3 = para_exponential_3[0] #拟合得到的参数
print ("k3 = ", k3, '\n', "t3 = ",t3, '\n', "L3 = ",L3, '\n', "Lf3 = ",Lf3)

t_fit_3 = np.linspace(rowstart_3/25,rowend_3/25,100)
y_exponential_3 = Lf3+(L3-Lf3)*np.exp(-k3*(t_fit_3 - t3))


'''画图框大小和字号设置
'''
fig = pl.figure(figsize=(10,8))
font_size = 16 #字体大小设置
line_width = 3 #线条宽度


'''1号图,用于确定各段曲线起始行，将鼠标落在曲线上读出相应行数以设定rowstart和rowend值
'''
pl.subplot(121)

pl.plot(time*25,ylength,'-b')

# #设定框内标识文本
pl.figtext(0.1,0.85,'k1 = ' + format(k1,'.5f')+ '; L1 = ' + format(L1,'.2f') +' um; ' + 'tm = ' + format(tm,'.1f') +' s; ' + 'Lm = ' + format(Lm,'.2f') +' um; ' + 'Lf1 = ' + format(Lf1,'.2f') +' um; ' +'delta L1 =' + format(ylength[rowstart_1]-ylength[rowend_1],'.2f')+ ' um',fontsize = 8)

pl.figtext(0.1,0.8,'k2 = ' + format(k2,'.5f')+ '; L2 = ' + format(L2,'.2f') +' um; ' + 'Lf2 = ' + format(Lf2,'.2f') +' um; ' + 'delta L2 =' + format(ylength[rowstart_2]-ylength[rowend_2],'.2f')+ ' um; ' + 't2 =' + format(t2,'.2f')+ ' s;'+ ' delta t2 =' + format(time[rowend_2]-time[rowstart_2],'.2f')+ ' s',fontsize = 8)

pl.figtext(0.1,0.75,'k3 = ' + format(k3,'.5f')+ '; t3 = ' + format(t3,'.2f') +' s; ' + 'L3 = ' + format(L3,'.2f') +' um; ' + 'Lf3 = ' + format(Lf3,'.2f') +' um; ' +'delta L3 =' + format(ylength[rowstart_3]-ylength[rowend_3],'.2f')+ ' um; ' + 't3 =' + format(t3,'.2f')+ ' s;'+ ' delta t3 =' + format(time[rowend_3]-time[rowstart_3],'.2f')+ ' s',fontsize = 8)

pl.figtext(0.1,0.95,'rowstart_1 = ' + format(rowstart_1,'d')+ '; rowend_1 = ' + format(rowend_1,'d') + '; rowend_2 = ' + format(rowend_2,'d')+ '; rowend_3 = ' + format(rowend_3,'d'), fontsize = 8)


'''2号图
'''
pl.subplot(122)
ax = pl.gca()

pl.plot(time,ylength,'-b',linewidth = 1)
pl.plot(t_fit_1, y_sigmoid,'r-', linewidth = line_width)
pl.plot(t_fit_2, y_exponential_2,'m-', linewidth = line_width)
pl.plot(t_fit_3, y_exponential_3,'c-', linewidth = line_width)

#设定xy轴标识,标题
pl.xlabel('time (s)',fontsize = font_size,labelpad = 0) #labelpad 设定x轴标注与坐标轴距离
pl.ylabel('Length ($\mathrm{\mu m}$)',fontsize = font_size,labelpad = 0)

#设定xy轴标度数字大小
pl.xticks(fontsize = font_size)
pl.yticks(fontsize = font_size)

#刻度标向内还是向外等参数设置
ax.tick_params(which = 'major',direction = 'in',length = 6,width = 1,color ='k')
ax.tick_params(which = 'minor',direction = 'in',length = 3,width = 1,color ='k')

#右边和上部是否显示刻度
ax.xaxis.set_ticks_position('bottom')  #括号内选项 'top'，'bottom', 'both', 'default', 'none'
ax.yaxis.set_ticks_position('left')

#使用默认设置的次坐标刻度
pl.matplotlib.pyplot.minorticks_on()

'''保存图片
#
'''
pl.tight_layout() #让图片不留空白部分
pl.savefig( 'MT-' + 'fig-'+'rowstart_1 = '+ format(rowstart_1) + ' rowend_3 = '+ format(rowend_3) +'.jpg',dpi=300)
pl.show()

'''保存剪切后的数据,令起始时间为0

'''
raw_data = np.genfromtxt(data_path,delimiter='\t')
column_time=[]
column_ylength=[]
column_xdev=[]

for k in range(len(data[1:])):
    column_time.append(raw_data[1:][k][1])
    column_ylength.append(raw_data[1:][k][3])
    column_xdev.append(raw_data[1:][k][4])

column_time_cut = column_time[rowstart_1:rowend_3] - column_time[rowstart_1]
column_ylength_cut = column_ylength[rowstart_1:rowend_3]
column_xdev_cut = column_xdev[rowstart_1:rowend_3]

file=open('ExperimentData_cut.txt','w')
for k in range(len(column_time_cut)):
    txt_1 = str(column_time_cut[k])+'\t'+str(column_ylength_cut[k])+'\t'+str(column_xdev_cut[k])+'\n'
    file.write(txt_1)
file.close()


'''保存拟合函数的拟合数据,令起始时间为0

'''
file=open('fitting_data.txt','w')
for k in range(len(t_fit_1)):
    txt_2 = str(t_fit_1[k] - column_time[rowstart_1])+'\t'+str(y_sigmoid[k])+'\n'
    file.write(txt_2)
for k in range(len(t_fit_2)):
    txt_2 = str(t_fit_2[k] - column_time[rowstart_1])+'\t'+str(y_exponential_2[k])+'\n'
    file.write(txt_2)
for k in range(len(t_fit_3)):
    txt_2 = str(t_fit_3[k] - column_time[rowstart_1])+'\t'+str(y_exponential_3[k])+'\n'
    file.write(txt_2)
file.close()


