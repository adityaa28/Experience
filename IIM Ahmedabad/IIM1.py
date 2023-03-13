#!/usr/bin/env python
# coding: utf-8

# In[59]:


import networkx as nx
import random
import pandas as pd
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import matplotlib.colors as mclr


# In[60]:


ncons=1000  #number of consumers
norg=50    #number of organisations
nemp=200  #number of employers

#scons=np.random.rand(ncons) #consumer score (randomly generated)
scons=np.array(pd.read_csv('Desktop/Data/IIM intern/final cons scores (16th dec).csv')['x'])
sorg=np.random.normal(0.5, 0.1, norg)   #organisations score
semp=np.random.normal(0.5, 0.1, nemp)  #employer score
semp[semp>1]=1
semp[semp<0]=0


# In[61]:


'''
#edgelist between consumers and employers (with weights)
consprop=[0.2,0.3,0.5]
consprop = [element * ncons for element in consprop]
consprop = [int(x) for x in consprop]
conswt=[0.25,0.5,0.7]
cnt1=0
ed3=pd.DataFrame()
for i in range(len(consprop)):
    g=nx.random_partition_graph([consprop[i], nemp], 0.5, conswt[i])
    edg=pd.DataFrame(g.edges())
    edg3=edg[(((edg[0]<consprop[i]) & (edg[1]>consprop[i]-1)) | ((edg[0]>consprop[i]-1) & (edg[1]<consprop[i])))]
    edg3[1]-=consprop[i]
    edg3[0]+=cnt1
    print(edg3)
    cnt1+=consprop[i]
    ed3=ed3.append(edg3)
ed3.reset_index(inplace=True, drop=True)
ed3
'''
#edgelist between consumers and employers
g=nx.random_partition_graph([ncons,nemp],0.5, 0.25)  #connection probability=0.25 between emp and consumers
#partition = g.graph["partition"]
#org=list(partition[0])
#cons=list(partition[1])
ed=pd.DataFrame(g.edges())
ed3=ed[(((ed[0]<ncons) & (ed[1]>ncons-1)) | ((ed[0]>ncons-1) & (ed[1]<ncons)))] #Dataframe with only connections between org and cons
ed3[1]-=ncons
ed3.reset_index(inplace=True, drop=True)
ed3


# In[62]:


import scipy.stats

plt.hist(sorg, 10, facecolor='green', alpha=0.5);

# plot density estimates
t_range = np.linspace(0,1,200)
bw_values =  [None, 0.1, 0.01]
# generate a list of kde estimators for each bw
kde = [scipy.stats.gaussian_kde(sorg,bw_method=bw) for bw in bw_values]
for i, bw in enumerate(bw_values):
    plt.plot(t_range,kde[i](t_range),lw=2, label='bw = '+str(bw))
plt.xlim(0,1)
plt.legend(loc='best')


# In[63]:


#for different weight proportion of organisations

orgprop=[0.2,0.3,0.5]
orgprop = [element * norg for element in orgprop]
orgprop = [int(x) for x in orgprop]
orgwt=[0.25,0.5,0.7]
cnt1=0
ed1=pd.DataFrame()
for i in range(len(orgprop)):
    g=nx.random_partition_graph([orgprop[i], ncons], 0.5, orgwt[i])
    edg=pd.DataFrame(g.edges())
    edg1=edg[(((edg[0]<orgprop[i]) & (edg[1]>orgprop[i]-1)) | ((edg[0]>orgprop[i]-1) & (edg[1]<orgprop[i])))]
    edg1[1]-=orgprop[i]
    edg1[0]+=cnt1
    print(edg1)
    cnt1+=orgprop[i]
    ed1=ed1.append(edg1)
ed1.reset_index(inplace=True, drop=True)
ed1

''''
#for uniform connections
g=nx.random_partition_graph([norg,ncons],0.5, 0.25)  #connection probability=0.25 between org and consumers
partition = g.graph["partition"]
org=list(partition[0])
cons=list(partition[1])
ed=pd.DataFrame(g.edges())
ed1=ed[(((ed[0]<norg) & (ed[1]>norg-1)) | ((ed[0]>norg-1) & (ed[1]<norg)))] #Dataframe with only connections between org and cons
ed1[1]-=norg
ed1.reset_index(inplace=True, drop=True)
ed1
'''


# In[64]:



#for connections between organisations and employers
cnt2=0
ed2=pd.DataFrame()
for i in range(len(orgprop)):
    g=nx.random_partition_graph([orgprop[i], nemp], 0.5, orgwt[i])
    edg=pd.DataFrame(g.edges())
    edg2=edg[(((edg[0]<orgprop[i]) & (edg[1]>orgprop[i]-1)) | ((edg[0]>orgprop[i]-1) & (edg[1]<orgprop[i])))]
    edg2[1]-=orgprop[i]
    edg2[0]+=cnt2
    print(edg2)
    cnt2+=orgprop[i]
    ed2=ed2.append(edg2)
ed2.reset_index(inplace=True, drop=True)
print(ed2)

'''
#for connections between organisations and employers (with weights)
orgprop=[0.2,0.3,0.5]
orgprop = [element * norg for element in orgprop]
orgprop = [int(x) for x in orgprop]
orgwt=[0.25,0.5,0.7]
cnt1=0
ed2=pd.DataFrame()
for i in range(len(orgprop)):
    g=nx.random_partition_graph([orgprop[i], nemp], 0.5, orgwt[i])
    edg=pd.DataFrame(g.edges())
    edg2=edg[(((edg[0]<orgprop[i]) & (edg[1]>orgprop[i]-1)) | ((edg[0]>orgprop[i]-1) & (edg[1]<orgprop[i])))]
    edg2[1]-=orgprop[i]
    edg2[0]+=cnt1
    print(edg2)
    cnt1+=orgprop[i]
    ed2=ed2.append(edg2)
ed2.reset_index(inplace=True, drop=True)
ed2
'''


# In[65]:


#Manuel version of the above code cell
''''
org1=int(0.3*norg)
org2=int(0.3*norg)
org3=int(0.3*norg)
g1=nx.random_partition_graph([org1,ncons],0.2, 0.25)
g2=nx.random_partition_graph([org2,ncons],0.5, 0.5)
g3=nx.random_partition_graph([org3,ncons],0.7, 0.75)
edg1=pd.DataFrame(g1.edges())
edg11=edg1[(((edg1[0]<org1) & (edg1[1]>org1-1)) | ((edg1[0]>org1-1) & (edg1[1]<org1)))]
edg2=pd.DataFrame(g2.edges())
edg12=edg2[(((edg2[0]<org2) & (edg2[1]>org2-1)) | ((edg2[0]>org2-1) & (edg2[1]<org2)))]
edg12[0]+=org1
edg3=pd.DataFrame(g3.edges())
edg13=edg3[(((edg3[0]<org3) & (edg3[1]>org3-1)) | ((edg3[0]>org3-1) & (edg3[1]<org3)))]
edg13[0]+=org1+org2
ed1 = pd.concat([edg11,edg12,edg13])
ed1
'''


# In[66]:


'''
iter1=2  #iterations
mode=1
df1=pd.DataFrame()   #df1=dataframe for storing org scores at every iteration
edsave=[]   #list to store ed1 at every iteration
for x in range(iter1):
    sm=np.random.rand(1)  #d 
    sorg=np.random.normal(sm, 0.3, norg)
    
    for i in range(norg):
        c=np.array(ed1[ed1[0]==i][1])  #c=array containing cons. index connected to the ith org.
        if(sorg[i]>1):
            sorg[i]=1
        elif(sorg[i]<0):
            sorg[i]=0
        #print(len(c))
        sum1=0
        for j in range(len(c)):
            sum1+=scons[c[j]]
        avg1=sum1/len(c)
        org_p=0
        Q=1
        p=(np.exp(org_p+avg1))/(1+np.exp(org_p+avg1)) #ptotalh 
        #print(p)
        p=p*Q
        if(mode==1):
            if(p>0.5):
                sorg[i]=random.uniform(p,1)
            elif(p<0.5):
                sorg[i]=random.uniform(0,p)
            else:
                sorg[i]=random.random()
        elif(mode==2):
            ran1=random.randint(0,1)
            if(ran1==0):
                var=random.choices([0,1], weights=(p,100-p), k=1)[0]
            else:
                var=random.choices([0,1], weights=(100-p,p), k=1)[0]
            if(var==1):
                sorg[i]=random.uniform(p,1)
            elif(var==0):
                sorg[i]=random.uniform(0,p)
            
        df1[x]=sorg
        #ed2=pd.DataFrame(columns=(0,1))
        for j in range(ncons):
            cons_org=(ed1[0].value_counts()[i])/ncons #number of consumers connected to org/ncons
            #cons_org*=100
            #print(cons_org)
            diff=(scons[j]-sorg[i])  #Doubt
            #print(str(diff) + ' ' +str(cons_org))
            p2=(np.exp(0.5*diff+0.5*cons_org))/(1+np.exp(0.5*diff+0.5*cons_org))
            print(str(i)+' '+str(j)+ ' ' + str(diff) + ' ' +str(cons_org)+ ' '+str(p2))
            if(p2>random.random()):
                if((len(ed1[(ed1[0]==i) & (ed1[1]==j)]))==0):   #if edge not found
                    ed1=ed1.append({0:i, 1:j}, ignore_index=True)   #then make the edge
            elif(p2==random.random()):
                if(random.random()>0.5):
                    if((len(ed1[(ed1[0]==i) & (ed1[1]==j)]))==0):
                        ed1=ed1.append({0:i, 1:j}, ignore_index=True)
                else:
                    if((len(ed1[(ed1[0]==i) & (ed1[1]==j)]))>0):
                        ed1=ed1.drop(ed1[((ed1[0] == i) & ( ed1[1] == j))].index)
            else:
                if((len(ed1[(ed1[0]==i) & (ed1[1]==j)]))>0):   #if edge found
                    ed1=ed1.drop(ed1[((ed1[0] == i) & ( ed1[1] == j))].index)   #remove the edge
    print(ed1)                
    
        #after allp:
'''
        


# In[67]:


def func(norg, sorg, data, mode, ed, cc):
    ndata=len(data)
    for i in range(norg):
        '''
        c=np.array(ed[ed[0]==i][1])  #c=array containing cons. index connected to the ith org.
        #print(len(c))
        sum1=0
        for j in range(len(c)):
            sum1+=data[c[j]]   
        avg1=sum1/len(c)
        org_p=0
        Q=1
        p=(np.exp(org_p+avg1))/(1+np.exp(org_p+avg1)) #ptotalh 
        p=p*Q
        
        if(mode==1):
            if(p>0.5):
                sorg[i]=random.uniform(p,1)
            elif(p<0.5):
                sorg[i]=random.uniform(0,p)
            else:
                sorg[i]=random.random()
        elif(mode==2):
            ran1=random.randint(0,1)
            if(ran1==0):
                var=random.choices([0,1], weights=(p,100-p), k=1)[0]
            else:
                var=random.choices([0,1], weights=(100-p,p), k=1)[0]
            if(var==1):
                sorg[i]=random.uniform(p,1)
            elif(var==0):
                sorg[i]=random.uniform(0,p)
        '''
        
        #df[x]=sorg if above commented code isnt there, then this will be before the function call
        for j in range(ndata):
            data_org=(ed[0].value_counts()[i])/ndata #number of consumers/employers connected to org
            #data_org*=100
            #print(data_org)
            diff=(data[j]-sorg[i]) 
            E=np.random.normal(0, 0.1, 1)
            E=E[0]
            mean = [0, 0, 0]
            cov = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]  # diagonal covariance
            e1,e2,e3=np.random.multivariate_normal(mean, cov, 1).T
            if(cc==0):
                #calculate semp
                c=np.array(ed3[ed3[0]==j][1])
                sum1=0
                for k in range(len(c)):
                    sum1+=semp[c[k]]
                avg1=sum1/len(c)
                p2=(np.exp((0.5+e1)*diff+(0.5+e2)*data_org+(0.5+e3)*avg1+E))/(1+np.exp((0.5+e1)*diff+(0.5+e2)*data_org+(0.5+e3)*avg1+E))
            elif(cc==1):
                #calculate scons
                c=np.array(ed3[ed3[1]==j][0])
                sum1=0
                for k in range(len(c)):
                    sum1+=scons[c[k]]
                avg1=sum1/len(c)
                p2=(np.exp((0.5+e1)*diff+(0.5+e2)*data_org+(0.5+e3)*avg1+E))/(1+np.exp((0.5+e1)*diff+(0.5+e2)*data_org+(0.5+e3)*avg1+E))
                
            #p2=(np.exp(0.5*diff+0.5*data_org))/(1+np.exp(0.5*diff+0.5*data_org))
            
            print(str(i)+' '+str(j)+ ' ' + str(diff) + ' ' +str(data_org)+ ' '+str(p2))
            if(p2>random.random()):
                if((len(ed[(ed[0]==i) & (ed[1]==j)]))==0):   #if edge not found
                    ed=ed.append({0:i, 1:j}, ignore_index=True)   #then make the edge
            elif(p2==random.random()):
                if(random.random()>0.5):
                    if((len(ed[(ed[0]==i) & (ed[1]==j)]))==0):
                        ed=ed.append({0:i, 1:j}, ignore_index=True)
                else:
                    if((len(ed[(ed[0]==i) & (ed[1]==j)]))>0):
                        ed=ed.drop(ed[((ed[0] == i) & ( ed[1] == j))].index)
            else:
                if((len(ed[(ed[0]==i) & (ed[1]==j)]))>0):   #if edge found
                    ed=ed.drop(ed[((ed[0] == i) & ( ed[1] == j))].index)   #remove the edge
        
    
    return ed
    
            
            

iter1=1  #iterations
mode=1
df2=pd.DataFrame()   #df1=dataframe for storing org scores at every iteration
edsave=[]   #list to store ed1 at every iteration
for x in range(iter1):
    sm=np.random.rand(1)  #d 
    sorg=np.random.normal(sm, 0.3, norg)
    sorg[sorg>1]=1
    sorg[sorg<0]=0
    ed1=func(norg, sorg, scons, mode, ed1, 0) #for consumers
    ed2=func(norg, sorg, semp, mode, ed2, 1)  #for employers


# In[443]:


mean = [0, 0, 0]
cov = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]  # diagonal covariance
a,b,c=np.random.multivariate_normal(mean, cov, 1).T


# In[444]:


print(a)
print(b)
print(c)


# In[12]:


ee=np.random.normal(0, 1, 1)
ee


# In[13]:


ee=ee[0]
ee


# In[ ]:





# In[182]:


'''
ed1.drop(ed1[ed1[0]==org[i]].index, inplace=True)  
for j in range(ncons):
    if(sorg[i]>scons[j]):
        ed1=ed1.append({0:i, 1:j}, ignore_index=True)
    elif(sorg[i]==scons[j]):
        if(random.random()>0.5):
            ed1=ed1.append({0:i, 1:j}, ignore_index=True)
    edsave.append(np.array(ed1))            
            '''


# In[523]:


ed2


# In[77]:


df1


# In[909]:


#var
'''
1) lets say p=0.62 var=pick 1 with prob 0.62,
if (var==1) score = unifor(p,1)
if(var==0) score = uniform(0,p)

read about parameter estimation by agent based simulation
   reduce Computational cost (using sampling approaches)
    optimisations
    


 discrete choice models
 



1)paramter est. by sampling
2)*multiple* parameter estimation in simulations
3)parameter est. by simulated moments/minimum simulation distance
4)parameter est. using simulated annealing 
'''


# In[57]:


df1


# In[25]:


ed1


# In[913]:


#test
''''
edt=pd.DataFrame(columns=(0,1))
edt=edt.append({0:1, 1:2}, ignore_index=True)
edt=edt.append({0:1, 1:3}, ignore_index=True)
edt=edt.append({0:2, 1:4}, ignore_index=True)
edt=edt.append({0:2, 1:5}, ignore_index=True)
edt
'''


# In[914]:


#test
''''
edt.drop(edt[edt[0]==1].index, inplace=True)
print(edt)
'''


# In[917]:


#test
''''
random.choices([0,1], weights=(62,38), k=1)[0]
'''


# In[918]:


#test
''''
test=[]
#test=pd.DataFrame()
for i in range(4):
    test.append(np.array(ed1))
test
'''


# In[1132]:


#for calculating final consumer scores
''''
from sklearn import preprocessing
dfcons=pd.read_excel('Desktop/Data/IIM intern/Final_Result (1).xlsx')
arr=np.array(dfcons.iloc[:,2])
arr=arr.reshape(-1, 1)
#arr
from numpy import inf
min_max_scaler = preprocessing.MinMaxScaler()
scaledcons = min_max_scaler.fit_transform(arr)
scaledcons=scaledcons*100
scaledcons=1/scaledcons
scaledcons
scaledcons[scaledcons == inf] = 1
pd.DataFrame(scaledcons).to_csv("cons scores.csv")
'''
#Rest part was done in R


# In[1120]:


#test (epdf sampling)
''''
nBins=1000
count_c, bins_c, = np.histogram(scaledcons, bins=nBins)
print(count_c)
myPDF = count_c/np.sum(count_c)
dxc = np.diff(bins_c)[0];   xc = bins_c[0:-1] + 0.5*dxc
#plot_distrib1(xc,myPDF)


myCDF = np.zeros_like(bins_c)
myCDF[1:] = np.cumsum(myPDF)
#plot_line(bins_c,myCDF,xc,myPDF)

arr1=[]
def get_sampled_element():
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF>=a)-1

def run_sampling(myCDF, nRuns=5000):
    X = np.zeros_like(myPDF,dtype=int)
    for k in np.arange(nRuns):
        variab=get_sampled_element()
        #arr1.append(variab)
        X[variab] += 1
    arr1.append(X/np.sum(X))
    return X/np.sum(X)

X = run_sampling(myCDF)
X
'''


# In[991]:


#test
''''
#df.loc[(df[‘Color’] == ‘Green’) & (df[‘Shape’] == ‘Rectangle’)]
if((len(ed1[(ed1[0]==0) & (ed1[1]==3)]))==0):
    print("not found")
else:
    print("found")
'''


# In[998]:


#test
''''
ed3=ed1.copy()
i = ed3[((ed3[0] == 0) &( ed3[1] == 1) )].index
ed3=ed3.drop(ed3[((ed3[0] == 0) &( ed3[1] == 1) )].index)
ed3
'''


# In[10]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
X = [[9,4],[-9,-4],[6,-6],[-6,6]]
X=pd.DataFrame(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
new_X = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
print(per_var)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('percentange of explained variance')
plt.xlabel('principal component')
plt.title('scree plot')
plt.show()


# In[6]:


X=pd.DataFrame(X)
X

