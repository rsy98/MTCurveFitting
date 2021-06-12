import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as tkf
from tkinter.ttk import * 
import pandas as pd
import numpy as np

import matplotlib.pyplot as pl 
from scipy import signal
from scipy import optimize
from scipy.signal import savgol_filter
import os

from pylab import *
import peakutils
from peakutils.plot import plot as pplot
from scipy.optimize import leastsq, fsolve

win = tk.Tk()
win.title('阶跃数据分析程序')
screenWidth = win.winfo_screenmmwidth()
screenHeight = win.winfo_screenheight()
width = 950
height = 575
locationX = (screenWidth - width)/2
locationY = (screenHeight - height)/2
win.geometry('%dx%d+%d+%d'% (width, height, locationX, locationY))
win.resizable(False, False)
win.configure(bg = 'WhiteSmoke')

labFrame = tk.LabelFrame(win, text = '文件与数据输入')
labels = ['数据文件名','平滑窗口数据点数(奇数)','拟合函数级数','窗口宽度设置(2w)','阶跃判断阈值(theta)','高度阈值(height,nm)','吻合阈值(height_filter_val)','阶跃统计图bins','帧率']

for r,i in enumerate(labels):
    # tk.Label(win, text = i, relief = 'ridge',width = 20).grid(row = r, column = 0)
    tk.Label(labFrame, text = i, width = 30).grid(row = r, column = 0)
    
Entry_names = ['file_name','smooth_win_width', 'fitting_order','detect_win_width', 'theta_val', 'height_val', 'height_filter_val','step_bins_width','frame_rate']
default_text = ['MTDataV1.txt','21','3','30','20','15','500','20','25.0']

for r, j in enumerate(Entry_names):
    globals()[j] = tk.Entry(labFrame)
    globals()[j].insert(1,format(default_text[r]))
    globals()[j].grid(row = r, column = 1)

labFrame.grid(row = 0, column = 0)

class StepDetect:
    def openFile(self):
        self.path = tkf.askopenfilename()
        file_name.delete(0,'end')
        file_name.insert(1, self.path)  
    def loadData(self):
        '''数据导入
        # data = np.genfromtxt('ExperimentData1.txt',delimiter = ' ')
        '''
        data_path  =  file_name.get()
        data  =  pd.read_csv(data_path,sep = '\t')
        return data
    def MT_index(self):
        index  =  self.loadData()['index']
        return index
    def MT_time(self):
        time = self.loadData()['time(s)']
        return time
    def MT_ylength(self):
        ylength = self.loadData()['y(um)']
        return ylength
    def MT_xdev(self):
        xdev = self.loadData()['x-deviation(um)']
        return xdev
    def row_number(self):
        return len(self.MT_index())
    def y_smooth(self):
        '''数据平滑函数，可改变第二个参数，该参数对应于平滑窗口数据点个数，必须设为奇数，参数过大会造成阶跃信号失真，
	    过小则会导致平滑效果较差，很多噪音信号会被误判为阶跃信号，第三个参数是拟合函数的级数，如设为3意味着用三次函数拟合
	    '''
        window_data_number = eval(smooth_win_width.get())
        fitting_degree = eval(fitting_order.get())
        ylength_smooth = savgol_filter(self.MT_ylength(), window_data_number, fitting_degree) 
        return ylength_smooth       
    def step_detect(self):
        m = []
        m0 = []
        t0= []
        tr = []
        tl = []
        theta = []
        height = []
        time_fit = []

        '''阶跃寻找算法所用变量声明
        '''
        sum_xy = 0
        sum_y_left = 0
        sum_x_left = 0
        sum_y_right = 0
        sum_x_right = 0
        slope_u = 0
        slope_0 = 0
        RSS_g = 0
        RSS_f = 0
        theta_fit = 0.0
        tl_fit = 0
        tr_fit = 0
        t0_fit = 0
        t_interval = 0
        '''算法窗口宽度参数和阶跃判断阈值设定，可更改
        该值需要根据数据特性而设，如果阶跃之间时间间隔很短，建议采用小的w值，为提高阶跃高度测量的准确度，建议采用大w值（如51）
        thres参数设定需要根据theta值大小设定阈值，可通过左下角图预判该值大小。
        thres参数设的较小会将很多信号误判为阶跃，设的较大则会忽略掉很多阶跃信号
        '''
        w = eval(detect_win_width.get())
        N = 2*w
        thres_value = eval(theta_val.get())

        '''阶跃寻找算法, theta_fit值用于判定是否有阶跃发生，如果该值较大且处于极值点，可认为是阶跃发生位置
        算法基本思路：宽度为2w的信号拟合窗口，左侧w宽度和右侧w宽度数据用斜率相同但截距不同的线性函数拟合，整个2w宽度信号
        用一个线性函数拟合，在平坦位置，两种拟合的差异不大，表现为theta_fit值小，但在以阶跃位置为中心点的位置，两种拟合的差异
        较大，theta_fit值会出现极大值，该值所在位置即为阶跃发生位置。
        '''
        for i in range(w, self.row_number() - w):
            sum_xy = sum([self.MT_time()[k]*self.y_smooth()[k] for k in range(i-w,i+w)])
            sum_xx = sum([k*k for k in self.MT_time()[i-w:i+w]])
            sum_x_full = sum([k for k  in self.MT_time()[i-w:i+w]])
            sum_y_full= sum([k for k  in self.y_smooth()[i-w:i+w]])
            sum_y_left = sum([k for k in self.y_smooth()[i-w:i]])
            sum_x_left = sum([k for k  in self.MT_time()[i-w:i]])
            sum_y_right = sum([k for k in self.y_smooth()[i:i+w]])
            sum_x_right= sum([k for k  in self.MT_time()[i:i+w]])
            
            slope_u = ((N/2)*sum_xy - sum_x_left*sum_y_left - sum_x_right*sum_y_right)/((N/2)*sum_xx - sum_x_left**2 - sum_x_right**2)
            slope_0 =  (N*sum_xy - sum_x_full*sum_y_full)/(N*sum_xx - sum_x_full**2)
            
            tl_fit = (2/N)*(sum_y_left - slope_u*sum_x_left)
            tr_fit = (2/N)*(sum_y_right - slope_u*sum_x_right)
            t0_fit = (sum_y_full - slope_0*sum_x_full)/N
            
            Rss_g = sum([(slope_0*self.MT_time()[k] + t0_fit - self.y_smooth()[k])**2 for k in range(i-w, i+w)])
            Rss_f  = sum([(slope_u*self.MT_time()[k] + tl_fit - self.y_smooth()[k])**2 for k in range(i-w,i)]) + sum([(slope_u*self.MT_time()[k]+tr_fit - self.y_smooth()[k])**2 for k in range(i,i+w)])
            
            theta_fit =  (Rss_g - Rss_f)*(tl_fit - tr_fit)*w*1e+4
            
            time_fit.append(self.MT_time()[i])
            m.append(slope_u)
            m0.append(slope_0)
            tl.append(tl_fit)
            tr.append(tr_fit)
            t0.append(t0_fit)
            theta.append(theta_fit)
    
        return (time_fit,theta)
    def time_fit(self):
        return self.step_detect()[0]
    # def slope_u(self):
    #     return self.step_detect()[1]
    # def slope_0(self):
    #     return self.step_detect()[2]      
    # def tl_fit(self):
    #     return self.step_detect()[3]
    # def tr_fit(self):
    #     return self.step_detect()[4]
    # def t0_fit(self):
    #     return self.step_detect()[5]
    def theta(self):
        return self.step_detect()[1]
    def linear_func(self,t, A, B):
        return A*t + B
    def step_location(self):
        indexes = []
        t_fit = []
        y_linear_fit = []
        t_plateau_fit = []
        y_plateau_fit = []
        y_li_minus_pl_square_fit = []
        
        height = []
        time_interval = []
        height_plateau = []
        time_interval_plateau = []

        height_filtered= []
        
        w = eval(detect_win_width.get())
        N = 2*w
        thres_value = eval(theta_val.get())
        fps = eval(frame_rate.get())
        '''寻找阶跃发生位置点
        因为算法从第w个数据点开始得到数据，为与原始数据标签保持一致，在寻找到的阶跃标签上加上w
        '''
        theta_1 = np.array(self.theta()) 
        indexes = peakutils.indexes(theta_1, thres = thres_value, min_dist = 50, thres_abs = True) + w        
        '''用线性函数和平台函数（对台阶之间的数据取平均值）拟合各阶跃数据
        '''		
        for k in range(len(indexes) + 1):	
            if k == 0:
                A1,B1 = optimize.curve_fit(self.linear_func, self.MT_time()[0:indexes[k]], self.MT_ylength()[0:indexes[k]])
                t_fit_region = np.linspace(0, indexes[k]/fps,100)
                for j in t_fit_region:
                    t_fit.append(j)
                    y_linear_fit.append(A1[0]*j+ A1[1]) #线性拟合
                    y_plateau_fit.append(np.mean(self.MT_ylength()[0:indexes[k]])) # 平台拟合
            elif k in range(1, len(indexes)):
                A1,B1 = optimize.curve_fit(linear_func, self.MT_time()[indexes[k-1]:indexes[k]],self.MT_ylength()[indexes[k-1]:indexes[k]])
                t_fit_region = np.linspace(indexes[k-1]/fps, indexes[k]/fps,100)
                for j in t_fit_region:
                    t_fit.append(j)
                    y_linear_fit.append(A1[0]*j+ A1[1])
                    y_plateau_fit.append(np.mean(self.MT_ylength()[indexes[k-1]:indexes[k]]))
            else:
                A1,B1 = optimize.curve_fit(linear_func, self.MT_time()[indexes[len(indexes)-1]:len(self.MT_time())-1],y[indexes[len(indexes)-1]:len(y)-1])
                t_fit_region = np.linspace(indexes[len(indexes)-1]/fps, self.MT_time()[len(self.MT_time())-1],100)
                for j in t_fit_region:
                    t_fit.append(j)
                    y_linear_fit.append(A1[0]*j+ A1[1])
                    y_plateau_fit.append(np.mean(y[indexes[len(indexes)-1]:len(self.MT_ylength())-1]))	

        '''计算阶跃的尺寸，这里乘以1000转化为nm单位, 如果高度h小于15 nm, 倾向于认定该阶跃数据是假信号，不予计入
        并删除掉相应的indexes坐标
        '''	
        height_val = eval(height_val.get())	
        for k in range(1,len(indexes) + 1):
            h = (y_linear_fit[100*(k-1)] - y_linear_fit[100*k])*1000
            h_plateau = (y_plateau_fit[100*(k-1)] - y_plateau_fit[100*k])*1000
            if k != len(indexes) :
                t_interval = (indexes[k] - indexes[k-1])/fps
                time_interval.append(t_interval)
            sum_square = sum([(y_linear_fit[j] - y_plateau_fit[j])**2 for j in range(100*(k-1),100*k)])*1e+5
            # (sum([j*j for j in y_linear_fit[(k-1)*100:k*100]]) + sum([j*j for j in y_plateau_fit[(k-1)*100:k*100]]) - 2*sum([y_linear_fit[j]*y_plateau_fit[j] for j in range(len(y_linear_fit[(k-1)*100:k*100]))]))
            y_li_minus_pl_square_fit.append(sum_square)
            if h > height_val:
                height.append(h)	
            if h_plateau > height_val:
                height_plateau.append(h_plateau)		

        '''计算线性拟合与平台拟合数据的吻合程度，如果某阶跃信号的前后两端拟合数据的吻合指标低于一个阈值，
        则认为该阶跃信号是典型的阶跃信号，可以计入统计，否则认为不满足阶跃特征，该段代码视情况选用
        '''		
        height_filter_val = eval(height_filter_val.get())
        for k in range(1,len(y_li_minus_pl_square_fit)):
            if y_li_minus_pl_square_fit[k-1] < height_filter_val and y_li_minus_pl_square_fit[k] < height_filter_val and (y_plateau_fit[100*(k-1)] - y_plateau_fit[100*k])*1000 > height_val:
            # if y_li_minus_pl_square_fit[k] < height_filter_val and (y_plateau_fit[100*(k-1) ] - y_plateau_fit[100*k])*1000 >15 :
                height_filtered.append((y_plateau_fit[100*(k-1)] - y_plateau_fit[100*k])*1000)   
        return [t_fit, y_linear_fit, y_plateau_fit, height_plateau, time_interval, height_filtered]
    def t_fit(self):
        return self.step_location()[0]
    def y_linear_fit(self):
        return self.step_location()[1]
    def y_plateau_fit(self):
        return self.step_location()[2]
    def height_plateau(self):
        return self.step_location()[3]
    def time_interval(self):
        return self.step_location()[4]
    def height_filtered(self):
        return self.step_location()[5]
    def step_fig(self):
        self.step_location()
        '''画图，包括原始数据，平台拟合数据，阶跃尺寸统计柱状图，阶跃位置确定图，两次阶跃发生时间间隔分布统计图
        '''		
        pl.figure(figsize = (10,8))
        font_size = 10 #字体大小设置
        line_width = 1 #线条宽度
        line_width_thin = 0.5 #线条宽度
        '''原始曲线图，平滑曲线，阶跃拟合曲线图
        '''
        pl.subplot(111)	
        ax = pl.gca()
        pl.plot(self.MT_time(),self.MT_ylength(),'b',linewidth = line_width_thin)
        # pl.plot(time,y_smooth)
        # pl.plot(t_fit, y_linear_fit,'r', linewidth = line_width_thin)
        pl.plot(self.t_fit(), self.y_plateau_fit(),'r', linewidth = line_width)

        #设定xy轴标识,标题
        pl.xlabel('time (s)',fontsize = font_size,labelpad = 0) #labelpad 设定x轴标注与坐标轴距离
        pl.ylabel('extension ($\mathrm{\mu m}$)', fontsize = font_size, labelpad = 0)

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
        pl.tight_layout()
        pl.show()
    def step_statistics(self):
        step_statistics_fig = pl.figure(figsize = (10,8))
        font_size = 10 #字体大小设置
        line_width = 1 #线条宽度
        line_width_thin = 0.5 #线条宽度
        step_bins_width = eval(step_bins_width.get())
        pl.subplot(111)	
        ax = pl.gca()
        # pl.hist(height, bins = np.linspace(0,200,20), normed = 1, facecolor = 'green', alpha = 0.75)
        # pl.hist(height, bins = 20,  histtype='bar', density=True, facecolor = 'green', alpha = 0.75)
        pl.hist(self.height_plateau(), bins = step_bins_width,  histtype='bar', density=True, facecolor = 'green', alpha = 0.75)
        # pl.hist(height_filtered, bins = 20,  histtype='bar', density=True, facecolor = 'orange', alpha = 0.75)

        #设定xy轴标识,标题
        pl.xlabel('step height (nm)',fontsize = font_size,labelpad = 0) #labelpad 设定x轴标注与坐标轴距离
        pl.ylabel('normalized frequency', fontsize = font_size, labelpad = 0)

        #设定xy轴标度数字大小
        pl.xticks(fontsize = font_size)
        pl.yticks(fontsize = font_size)

        #刻度标向内还是向外等参数设置
        ax.tick_params(which = 'major',direction = 'out',length = 6,width = 1,color ='k')
        ax.tick_params(which = 'minor',direction = 'out',length = 3,width = 1,color ='k')

        #右边和上部是否显示刻度
        ax.xaxis.set_ticks_position('bottom')  #括号内选项 'top'，'bottom', 'both', 'default', 'none'
        ax.yaxis.set_ticks_position('left')

        #使用默认设置的次坐标刻度
        pl.matplotlib.pyplot.minorticks_on()
        pl.tight_layout()
        pl.show()
    def step_judge(self):
        step_judge_fig = pl.figure(figsize = (10,8))
        font_size = 10 #字体大小设置
        line_width = 1 #线条宽度
        line_width_thin = 0.5 #线条宽度
        pl.subplot(111)	
        ax = pl.gca()
        pl.plot(self.time_fit(), self.theta())
        for k in indexes:
            pl.plot(self.time_fit()[k-w], self.theta()[k-w],'*',linewidth = 5)
        #设定xy轴标识,标题
        pl.xlabel('time (s)',fontsize = font_size,labelpad = 0) #labelpad 设定x轴标注与坐标轴距离
        pl.ylabel('significance (a.u.)', fontsize = font_size, labelpad = 0)

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
        pl.tight_layout()
        pl.show()
    def step_time_interval(self):
        pl.subplot(111)	
        ax = pl.gca()
        pl.hist(self.time_interval(), bins = np.linspace(0,30,20), density = True, histtype='bar', facecolor = 'blue', alpha = 0.75)
        # pl.plot(y_li_minus_pl_square_fit,'s')

        #设定xy轴标识,标题
        pl.xlabel('time interval (s)',fontsize = font_size,labelpad = 0) #labelpad 设定x轴标注与坐标轴距离
        pl.ylabel('normalized frequency', fontsize = font_size, labelpad = 0)

        #设定xy轴标度数字大小
        pl.xticks(fontsize = font_size)
        pl.yticks(fontsize = font_size)

        #刻度标向内还是向外等参数设置
        ax.tick_params(which = 'major',direction = 'out',length = 6,width = 1,color ='k')
        ax.tick_params(which = 'minor',direction = 'out',length = 3,width = 1,color ='k')

        #右边和上部是否显示刻度
        ax.xaxis.set_ticks_position('bottom')  #括号内选项 'top'，'bottom', 'both', 'default', 'none'
        ax.yaxis.set_ticks_position('left')

        #使用默认设置的次坐标刻度
        pl.matplotlib.pyplot.minorticks_on()
        pl.tight_layout()
        pl.show()
    def save_pic(self):
        pl.tight_layout() #让图片不留空白部分
        pl.savefig('fig-step-detection'+' w = ' + format(w) + ' thres = '+ format(thres_value)+ '.jpg',dpi = 600)     #保存图片在数据同文件夹内，图片名字和格式可改，格式可改为.tif,.jpg,.eps等格式

labFrame_btn = tk.LabelFrame(win, text = '数据分析')
btn_open_file = tk.Button(labFrame_btn, text = '打开数据文件', command = StepDetect().openFile, width = 30)
btn_open_file.grid(row = 0, columnspan = 5)   

# btn_flat = tk.Button(labFrame_btn, text = '数据平滑', command = StepDetect().data_flat, width = 30)
# btn_flat.grid(row = 1, columnspan = 5)  

btn_step_fig = tk.Button(labFrame_btn, text = '阶跃拟合图', command = StepDetect().step_fig, width = 30)
btn_step_fig.grid(row = 1, columnspan = 5)  

btn_step_statistics = tk.Button(labFrame_btn, text = '阶距柱状统计分布图', command = StepDetect().step_statistics, width = 30)
btn_step_statistics.grid(row = 2, columnspan = 5)   

btn_step_judge = tk.Button(labFrame_btn, text = '阶跃信号判定依据图', command = StepDetect().step_judge,width = 30)
btn_step_judge.grid(row = 3, columnspan = 5)   

btn_step_time_interval = tk.Button(labFrame_btn, text = '阶跃时间间隔柱状统计分布图', command = StepDetect().step_time_interval,width = 30)
btn_step_time_interval.grid(row = 4, columnspan = 5)  

btn_save_pic = tk.Button(labFrame_btn, text = '保存图片', command = StepDetect().save_pic,width = 30)
btn_save_pic.grid(row = 5, columnspan = 5)  

labFrame_btn.grid(row = 0, column = 1)
  
win.mainloop()

