# -*- coding: utf-8 -*-
"""Tkinter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fneMHkzeqXThYRlAOYdQ_eTu6SrveHGr
"""

import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from matplotlib.widgets import Slider
from matplotlib import colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib import style
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF
import os
import seaborn
from scipy.stats import kstest

style.use('ggplot')
matplotlib.use('TkAgg')

#changing the count of repeating rows to the next nearest square

def isperfsq(x):
    y = int(math.sqrt(x)) 
    if y**2 == x:
        return y
    else:
        return y+1 #return the root of the perfect square

def remove(widget):
    widget.grid_remove()

def cdf():
    global la,lab
    ind = spin0.get()
    s = spin1.get() 
    
    #error for number given outside the range
    if int(s)>len(no_offiles)-1 or int(s)<1 or int(ind)>6 or int(ind)<1:
        la = tk.Label(root,text='Number outside the Range')
        la.grid(row=3,column=1,columnspan=4)
        widgetlist2.append(la)
        return
    
    #if number is inside the range
    else:
        lab = tk.Label(root,text='                                              ')
        lab.grid(row=3,column=1,columnspan=4)

        past = os.getcwd() #changing the directory
        req = 'testscenarios_uniform_parallel_W_alph_new'
        string = 'sc_'+s
        file = 'other_node_attr.csv'
        folder = open(os.path.join(past,req,string,file),'r') #reading the file

        df1 = pd.read_csv(folder)
        df = pd.DataFrame({'values' : df1['7'],
                        'Simulation Time': df1['30']})
        gbtime = df.groupby('Simulation Time')
        length = len(df1['30'].unique())
        data = pd.read_csv('cosine_input.csv')

        #Function for Sensitivity
        def cdfs():
            #plt.clf()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)

            past = os.getcwd()
            req0 = 'testscenarios_uniform_parallel_W_alpha_new_sense_simple_new_missing_reg'
            req1 = 'testscenarios_uniform_parallel_W_alph_new'

            file0 = 'Look_for_indicator_in laast column.csv'
            file1 = 'other_node_attr.csv'

            folder0 = open(os.path.join(past,req0,file0),'r')
            f0 = pd.read_csv(folder0)
            df = pd.DataFrame({'indicators' : f0['ind']})
            df['index'] = df.index
            df = df.groupby('indicators')
            filenumbers = list(df.get_group((int(ind)))['index'])
            for number in filenumbers:
                string = 'sc_' + str(number+1)
                folders = open(os.path.join(past,req0,string,file1),'r')
                f1 = pd.read_csv(folders)
                dataframe1 = pd.DataFrame({'values' : f1['7']})
                rows= f1[f1.columns[0]].count()
                x = rows/len(f1['30'].unique())
                ser0= dataframe1.iloc[int(rows-x):,0]
                ecdf = ECDF(ser0)
                if(number==1):
                    ax2.plot(ecdf.x,ecdf.y,'k', label='Hyperparameter Scenarios')
                else:
                    ax2.plot(ecdf.x,ecdf.y,'k')

            string = 'sc_'+s
            folder1 = open(os.path.join(past,req1,string,file1),'r')
            f2 = pd.read_csv(folder1)
            dataframe1 = pd.DataFrame({'values' : f2['7']})
            rows= f2[f2.columns[0]].count()
            x = rows/len(f2['30'].unique())
            ser0= dataframe1.iloc[int(rows-x):,0]
            ecdf = ECDF(ser0)
            
            #Graph between tolerance and cdf
            ax2.plot(ecdf.x,ecdf.y,'g', label='Baseline Scenario')
            plt.xlabel('Tolerance')
            plt.ylabel('CDF')
            plt.title('Sensitivity')
            plt.legend()
            plt.show()

        #Function for Tolerance
        def localcdf():
            #plt.clf()
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            
            for times in range(1,length-1):
                serie= pd.Series(gbtime.get_group((times))['values'])
                ecdf = ECDF(serie)
                if(times==1):
                    ax3.plot(ecdf.x,ecdf.y,'k', label='Simulations')
                else:
                    ax3.plot(ecdf.x,ecdf.y,'k')
            
            data1 = data.loc[:,'Tolerance']
            ecdf = ECDF(data1)
            ax3.plot(ecdf.x,ecdf.y,'g', label='Real Data') #Plot for real data
            serie= pd.Series(gbtime.get_group((0))['values'])
            ecdf = ECDF(serie)
            ax3.plot(ecdf.x,ecdf.y,'b', label='First Simulation') #Plot for first simulation
            serie= pd.Series(gbtime.get_group((length-1))['values'])
            ecdf = ECDF(serie)
            ax3.plot(ecdf.x,ecdf.y,'r', label='Final Simulation') #Plot for final simulation
            plt.xlabel('Tolerance')
            plt.ylabel('CDF')
            plt.title('T')
            plt.legend()
            plt.show() 
        
        #Function for Global CDF
        def globalcdf():
            #plt.clf()
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)

            #initial
            serie= pd.Series(gbtime.get_group((0))['values'])
            ecdf = ECDF(serie)
            ax4.plot(ecdf.x,ecdf.y,'k', label='First Simulation') #Plot for first simulation
            #final
            serie= pd.Series(gbtime.get_group((length-1))['values'])
            ecdf = ECDF(serie)
            ax4.plot(ecdf.x,ecdf.y,'r', label='Final Simulation') #Plot for final simulation
            #actual
            data1 = data.loc[:,'Tolerance']
            ecdf = ECDF(data1)
            ax4.plot(ecdf.x,ecdf.y,'g', label='Real Data') #Plot for real data
            
            #Plotting Global CDF
            plt.xlabel('Tolerance')
            plt.ylabel('CDF')
            plt.title('Global Cumulative Distribution Function')
            plt.legend()
            plt.show()

        #Function for local CDFs
        def difference():
            fig1, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
            fig1.suptitle('Local Cumulative Distributive Function',fontsize=14)

            length = len(df1['30'].unique())
            rows = df1[df1.columns[0]].count()
            x = rows/length

            df = df1.iloc[int(rows-x):,:]
            df = pd.DataFrame(df)

            data['cat_type']=data['cat_type'].fillna(method ='ffill')
            
            arr = []
            list =[]
            list1=[]
            list2=[]
            list3=[]
            arr1=[]
            arr2=[]
            arr3=[]
            list0 = ['market_cap','Region','cat_type']

            mc0 = data.groupby('market_cap')
            for i in data['market_cap'].unique():
                serie = pd.Series(mc0.get_group((i))['Tolerance']).tolist()
                arr1.append(serie)
                list1.append(i)
            list.append(list1)
            arr.append(arr1)

            reg0= data.groupby('Region')
            for i in data['Region'].unique():
                serie = pd.Series(reg0.get_group((i))['Tolerance']).tolist()
                arr2.append(serie)
                list2.append(i)
            list.append(list2)
            arr.append(arr2)

            ct0=data.groupby('cat_type')
            for i in data['cat_type'].unique():
                serie = pd.Series(ct0.get_group((i))['Tolerance']).tolist()
                arr3.append(serie)
                list3.append(i)
            list.append(list3)
            arr.append(arr3)

            #calcdata
            ar =[]
            ar1 =[]
            ar2=[]
            ar3=[]

            mc1 = df.groupby('12') 
            for i in df['12'].unique():
                serie = pd.Series(mc1.get_group((i))['7']).tolist()
                ar1.append(serie)
            ar.append(ar1)

            reg1= df.groupby('9')
            for i in df['9'].unique():
                serie = pd.Series(reg1.get_group((i))['7']).tolist()
                ar2.append(serie)
            ar.append(ar2)

            ct1=df.groupby('10')
            for i in df['10'].unique():
                serie = pd.Series(ct1.get_group((i))['7']).tolist()
                ar3.append(serie)
            ar.append(ar3)

            # clear subplots
            for ax in axs:
                ax.remove()

            # add subfigure per subplot
            gridspec = axs[0].get_subplotspec().get_gridspec()
            subfigs = [fig1.add_subfigure(gs) for gs in gridspec]

            for row, subfig in enumerate(subfigs):
                subfig.suptitle(list0[row],fontsize = 13)
                # create 1x3 subplots per subfig
                axs = subfig.subplots(nrows=1, ncols=3)
                listt = list[row]
                array = arr[row]
                array0 = ar[row]
                for col, ax in enumerate(axs):
                    ecdf = ECDF(array[col])
                    ecdf0 = ECDF(array0[col])
                    ax.plot(ecdf.x,ecdf.y,'b')
                    ax.plot(ecdf0.x,ecdf0.y,'k')
                    ax.set_title(listt[col],fontsize =11)

            plt.show()

        #Function for sensitivity index
        def sindex():
            #plt.clf()
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(111)

            past = os.getcwd()
            req = 'testscenarios_uniform_parallel_W_alpha_new_sense_simple_new_missing_reg'
            file0 = 'Look_for_indicator_in laast column.csv'
            file1 = 'other_node_attr.csv'

            kstr =[]
            kstat =[]

            folders = open(os.path.join(past,req,file0),'r')
            f = pd.read_csv(folders)
            dff = pd.DataFrame({'indicators' : f['ind']})
            rows= df1[df1.columns[0]].count()
            x = rows/len(df1['30'].unique())
            d = np.array(df1.loc[int(rows-x):,'7'])
            #data = pd.read_csv('cosine_input.csv')
            #data1 = np.array(data['Tolerance'])

            filenumbers = dff.shape[0]
            for number in range(filenumbers):
                string = 'sc_' + str(number+1)
                folders = open(os.path.join(past,req,string,file1),'r')
                f = pd.read_csv(folders)
                arr = np.array(f.loc[int(rows-x):,'7'])
                p = kstest(d,arr)
                kstat.append(p[0])
                kstr.append(p)
            dff['kstat'] = kstat
            values = np.array(dff.groupby('indicators').mean()).flatten()
            dataframe2 = pd.DataFrame({'index':range(1,7),'values':values})
            h= seaborn.barplot(data= dataframe2, x= 'index',y = 'values')
            plt.title('Sensitivity Index')
            ax5.plot(h)

        button5 = tk.Button(root, text="Sensitivity",command=cdfs)
        button5.grid(row=1, column=8, padx=5, pady=10)
        button6 = tk.Button(root, text="T",command = localcdf)
        button6.grid(row=1, column=5, padx=5, pady=10)
        button7 = tk.Button(root, text="Global CDF",command = globalcdf)
        button7.grid(row=1, column=7, padx=5, pady=10)
        button8 = tk.Button(root, text="Local CDF",command = difference)
        button8.grid(row=1, column=6, padx=5, pady=10)
        button9 = tk.Button(root, text="Sensitivity Index",command = sindex)
        button9.grid(row=1, column=9, padx=5, pady=10)


    widgetlist2.append(button5)
    widgetlist2.append(button6)
    widgetlist2.append(button7)
    widgetlist2.append(button8)
    widgetlist2.append(button9)
    widgetlist2.append(lab)

def graph(rt,length,gbtime,df,s):

    #v1 = tk.StringVar()

    #def stime():
    #    v= s_time.get()
        

    def update(val):
        global pos
        pos = s_time.val

        ax.clear()
        cax.cla()
        ax.grid(False)
        
        rc = rt**2
        y_values = np.array(gbtime.get_group((int(pos)))['values'])
        if np.size(y_values)/rc!=0:
            q = rc- np.size(y_values)
            y_values = np.pad(y_values,(0,q),'constant')

        w = y_values.reshape(rt,rt)
        
        im = ax.imshow(w, cmap="Greens", vmin=0)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        
        ax.set_xticks(np.arange(rt+1)-0.5,minor=True)
        ax.set_yticks(np.arange(rt+1)-0.5,minor=True)
        
        plt.show()
        fig.canvas.draw_idle()
        
    def open():
        global bp
        bp = df.boxplot(by='Simulation Time')
        plt.ylabel('Tolerance')
        plt.title('Tolerance vs Simulation Time')
        plt.tight_layout()
        
    def plot():
        global frame
        frame = tk.Frame(root)
        frame.grid(row=2, column=0, columnspan=9)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill = tk.BOTH, expand =1)
        
        fig.subplots_adjust(bottom=0.25)

        ax.axis([0, rt, 0, rt])
        ax_time = fig.add_axes([0.1, 0.16, 0.8, 0.03])
        
        return ax_time

    def save():
        fig.savefig(s+'_'+str(int(pos))+'.png')

    global button3,button4
    fig = plt.Figure(figsize = (10,9),dpi = 100)
    ax=fig.add_subplot(111)

    button3 = tk.Button(root, text="Boxplot",command=open)
    button3.grid(row=1, column=5, padx=5, pady=10)
    button4 = tk.Button(root, text="Save grid",command=save)
    button4.grid(row=1, column=7, padx=5, pady=10)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.4)

    s_time = Slider(plot(), 'Time', 0, length-1, valinit=0, valstep=1)
    s_time.on_changed(update)

    widgetlist1.append(button3)
    widgetlist1.append(button4)
    widgetlist1.append(frame)

#function to read the file and call graph function for grid visualization
def action():
    s = e1.get() 

    if int(s)>len(no_offiles)-1 or int(s)<1:
        l = tk.Label(root,text='Number outside the Range')
        l.grid(row=3,column=0,columnspan=4)
        widgetlist1.append(l)
        if frame:
            frame.grid_remove()
            button3.grid_remove()
            button4.grid_remove()
        return
        
    past = os.getcwd() #changing the directory
    req = 'testscenarios_uniform_parallel_W_alph_new'
    string = 'sc_'+s
    file = 'other_node_attr.csv'
    folder = open(os.path.join(past,req,string,file),'r') #reading the file

    df1 = pd.read_csv(folder)
    df = pd.DataFrame({'values' : df1['7'],
                       'Simulation Time': df1['30']})
    gbtime = df.groupby('Simulation Time')
    length = len(df1['30'].unique())
    x = df1[df1.columns[0]].count()/length

    rt = isperfsq(x) 
    graph(rt,length,gbtime,df,s) #calling grid function

    #e1.delete(0,'end')

def actionplan():
    for widget in widgetlist2:
        remove(widget)
        
    global e1,no_offiles
    no_offiles = os.listdir(os.path.join(os.getcwd(),'testscenarios_uniform_parallel_W_alph_new'))
    label1= tk.Label(root,text="Test Scenario Number")
    label1.grid(row=1,column=0,padx =5,pady=10)
    e1 = tk.Spinbox(root,from_=1,to=len(no_offiles)-1)
    e1.grid(row=1, column=1, padx=5, pady=10)
    button = tk.Button(root, text = "Click for the Grid",command=action)
    button.grid(row=1,column=2,padx =5,pady=10,columnspan=2)
    
    widgetlist1.append(button)
    widgetlist1.append(e1)
    widgetlist1.append(label1)

def actionplan2():
    for widget in widgetlist1:
        remove(widget)

    global spin0,spin1
    buttonn = tk.Button(root, text = "Click for cdfs",command=cdf)
    buttonn.grid(row=1,column=4,padx =5,pady=10)
    label2= tk.Label(root,text="Factor")
    label2.grid(row=1,column=0,padx =5,pady=10)
    spin0 = tk.Spinbox(root,from_=1,to=6)
    spin0.grid(row=1,column=1,padx =5,pady=10)
    label3= tk.Label(root,text="Baseline")
    label3.grid(row=1,column=2,padx =5,pady=10)
    spin1 = tk.Spinbox(root,from_=1,to=len(no_offiles)-1)
    spin1.grid(row=1,column=3,padx =5,pady=10)

    widgetlist2.append(label2)
    widgetlist2.append(spin0)
    widgetlist2.append(label3)
    widgetlist2.append(spin1)    
    widgetlist2.append(buttonn)
    
def help():
    
    
    top = tk.Toplevel(root)
    top.title("Documentation")
    top.geometry("500x500")
    T = tk.Text(top, height=200, width=250)
    T.pack()
    T.insert(tk.END, "A paragraph is a series of sentences that are organized and coherent, and are all related to a single topic. Almost every piece of writing you do that is longer than a few sentences should be organized into paragraphs. This is because paragraphs show a reader where the subdivisions of an essay begin and end, and thus help the reader see the organization of the essay and grasp its main points.'\nin two lines\n'Paragraphs can contain many different kinds of information. A paragraph could contain a series of brief examples or a single long illustration of a general point. It might describe a place, character, or process; narrate a series of events; compare or contrast two or more things; classify items into categories; or describe causes and effects. Regardless of the kind of information they contain, all paragraphs share certain characteristics. One of the most important of these is a topic sentence.")
    
    #label = tk.Label(top,text = 'A paragraph is a series of sentences that are organized and coherent, and are all related to a single topic.\n \nAlmost every piece of writing you do that is longer than a few sentences should be organized into paragraphs. This is because paragraphs show a reader where the subdivisions of an essay begin and end,\nand thus help the reader see the organization of the essay and grasp its main points.')
    #label1=tk.Label(top, text= 'Paragraphs can contain many different kinds of information. A paragraph could contain a series of brief examples or a single long illustration of a general point. It might describe a place, character, or process; narrate a series of events; compare or contrast two or more things; classify items into categories; or describe causes and effects. Regardless of the kind of information they contain, all paragraphs share certain characteristics. One of the most important of these is a topic sentence.')
    #label1.grid()
    #label.grid()

from tkinter import ttk
#from tkmacosx import Button
#calling the root window and setting basic widgets
root = tk.Tk()
root.title("Graph")
#root.geometry("1000x900")
#root.geometry("1240x900")

global widgetlist1,widgetlist2
widgetlist1=[]
widgetlist2=[]

button1 = tk.Button(root, text = "Grid Visualization",command=actionplan) #Grid Visualization
button1.grid(row=0,column=0,padx =5,pady=10)

button2 = tk.Button(root,text = 'CDF Visualization',command=actionplan2) #CDF Visualization
button2.grid(row=0,column=1,padx =5,pady=10,columnspan=4)

helpbutton = tk.Button(root, text ='Help!', command = help)
helpbutton.grid(row=0,column=5,padx =5,pady=10)

root.mainloop()