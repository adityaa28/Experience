#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data=pd.read_csv('Desktop/Data/Uni of Essex Project/network dataset/network dataset.csv')
data.head()


# In[3]:


df1= data
df1.drop(df1.tail(2).index,inplace=True)
l=len(df1)


# In[44]:


df3=data[['vehicle_angle', 'vehicle_pos', 'vehicle_speed']]
plt.figure(figsize=(10,10))
corr = df3.corr()
corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[45]:


df1.head(50)


# In[46]:


xc= df1['vehicle_x'].copy()
yc= df1['vehicle_y'].copy()
vangle= df1['vehicle_angle'].copy()
vspeed= df1['vehicle_speed'].copy()


# In[47]:


rows, cols, time = (l, l, 3601)
''''wifidur =  [[0 for i in range(cols)] for j in range(rows)]
ieeedur =  [[0 for i in range(cols)] for j in range(rows)]
celldur =  [[0 for i in range(cols)] for j in range(rows)]
wifideg =  [[0 for i in range(time+1)] for j in range(rows)]
ieeedeg =  [[0 for i in range(time+1)] for j in range(rows)]
celldeg =  [[0 for i in range(time+1)] for j in range(rows)]
starttimewifi = [[-1 for i in range(cols)] for j in range(rows)]
starttimeieee = [[-1 for i in range(cols)] for j in range(rows)]
starttimecell = [[-1 for i in range(cols)] for j in range(rows)]
#dist=[[0 for i in range(cols)] for j in range(rows)]'''
wifidur = np.zeros((cols,rows))
ieeedur = np.zeros((cols,rows))
celldur = np.zeros((cols,rows))
wifideg = np.zeros((rows,time+1))
ieeedeg = np.zeros((rows,time+1))
celldeg = np.zeros((rows,time+1))
starttimewifi = np.full((cols,rows), -1)
starttimeieee = np.full((cols,rows), -1)
starttimecell = np.full((cols,rows), -1)


# In[48]:


from math import pi, sin, cos, sqrt


# In[49]:


for i in range(rows):
        for j in range(cols):
            if(i!=j):
                dist=sqrt((xc[i]-xc[j])*(xc[i]-xc[j]) + (yc[i]-yc[j])*(yc[i]-yc[j]))
                #print(dist, " ")
                if(dist<=10):
                    wifideg[i][0]+=1
                    if(starttimewifi[i][j]==-1):
                        starttimewifi[i][j]=0
                elif(dist>10 and dist<=400):
                    ieeedeg[i][0]+=1
                    if(starttimeieee[i][j]==-1):
                        starttimeieee[i][j]=0
                elif(dist>400):
                    celldeg[i][0]+=1
                    if(starttimecell[i][j]==-1):
                        starttimecell[i][j]=0


# In[50]:


for t in range(time):
    for j in range (rows):
        xc[j]=xc[j]+vspeed[j]*cos((pi/180)*vangle[j])
        yc[j]=yc[j]+vspeed[j]*sin((pi/180)*vangle[j])
    for i in range(rows):
        for j in range(cols):
            if(i!=j):
                dist=sqrt((xc[i]-xc[j])*(xc[i]-xc[j]) + (yc[i]-yc[j])*(yc[i]-yc[j]))
                #print(dist, " ")
                if(dist<=10):
                    wifidur[i][j]+=1
                    wifideg[i][t+1]+=1
                    if(starttimewifi[i][j]==-1):
                        starttimewifi[i][j]=t+1
                elif(dist>10 and dist<=400):
                    ieeedur[i][j]+=1
                    ieeedeg[i][t+1]+=1
                    if(starttimeieee[i][j]==-1):
                        starttimewifi[i][j]=t+1
                elif(dist>400):
                    celldur[i][j]+=1
                    celldeg[i][t+1]+=1
                    if(starttimecell[i][j]==-1):
                        starttimecell[i][j]=t+1


# In[51]:


rows2, cols2, time = (l, l, 3601)
wifideg2=np.zeros((rows2, time+1))
ieeedeg2=np.zeros((rows2, time+1))
dur2= np.zeros((rows2, rows2))
df2=data[:100]
xc2= df2['vehicle_x'].copy()
yc2= df2['vehicle_y'].copy()
vangle2= df2['vehicle_angle'].copy()
vspeed2= df2['vehicle_speed'].copy()


# In[52]:


#Removing random 5% nodes
import random
dur30=[] #total connections with duration greater than 30 sec
dur30wifi=[]
dur30ieee=[]
dur2wifi =  [[0 for i in range(rows2)] for j in range(rows2)]
dur2ieee =  [[0 for i in range(rows2)] for j in range(rows2)]
#unconn2 =  [[0 for i in range(rows2)] for j in range(rows2)]
dur20=[]
dur10=[]
dur60=[]
dur40=[]
dur50=[]
#unconn10=[] #nodes unconnected for more than 10 sec
tcrand=[] #total connections at time t
#smrand=[] #number of nodes with connectivity>=5
for t in range(time+1):
    randomlist = random.sample(range(0,100),90)
    sumdur30=0 #number of connections with duration greater than 30 sec
    sumdur30wifi=0
    sumdur30ieee=0
    sumdur20=0
    sumdur10=0
    sumdur60=0
    sumdur40=0
    sumdur50=0
    #unconndur10=0
    nc=0 #number of connections
    if(t!=0):
        for j in range(rows2):
            xc2[j]=xc2[j]+vspeed2[j]*cos((pi/180)*vangle2[j])
            yc2[j]=yc2[j]+vspeed2[j]*sin((pi/180)*vangle2[j])
    for i in range(rows2):
        for j in range(rows2):
            if(i!=j):
                dist=sqrt((xc2[i]-xc2[j])*(xc2[i]-xc2[j]) + (yc2[i]-yc2[j])*(yc2[i]-yc2[j]))
               # if(dist<=10):
                #    wifideg2[i][t]+=1
                #elif(dist>10 and dist<=400):
                 #   ieeedeg2[i][t]+=1
                if(dist<=400 and t!=0):
                    dur2[i][j]+=1
                    #unconn2[i][j]=0
                if(dist<=10 and t!=0):
                    dur2wifi[i][j]+=1
                if(dist>10 and dist<=400 and t!=0):
                    dur2ieee[i][j]+=1
                    dur2wifi[i][j]=0
                if(dist>400 and t!=0):
                    dur2[i][j]=0
                    dur2wifi[i][j]=0
                    dur2ieee[i][j]=0
                   # unconn2[i][j]+=1
    for i in (randomlist):
        for j in (randomlist):
            if(i!=j):
                dist=sqrt((xc2[i]-xc2[j])*(xc2[i]-xc2[j]) + (yc2[i]-yc2[j])*(yc2[i]-yc2[j]))
                if(dist<=10):
                    wifideg2[i][t]+=1
                elif(dist>10 and dist<=400):
                    ieeedeg2[i][t]+=1
                if(dist<=400):
                    nc+=1
    nwifidur30=0 #Number of nodes which are connected for more than 30 sec
    nieeedur30=0
    #nunconndur10=0
    tcrand.append(nc/2)
    for i in (randomlist):
        for j in (randomlist):
            if(i<j):
                if(dur2[i][j]>=30):
                    sumdur30+=1
                if(dur2[i][j]>=10):
                    sumdur10+=1
                if(dur2[i][j]>=40):
                    sumdur40+=1
                if(dur2[i][j]>=60):
                    sumdur60+=1
                if(dur2[i][j]>=20):
                    sumdur20+=1
                if(dur2[i][j]>=50):
                    sumdur50+=1
    for i in range(rows2):
        for j in range(rows2):
            if(i!=j):
                if(dur2wifi[i][j]>=30):
                    nwifidur30+=1
                    break
    for i in range(rows2):
        for j in range(rows2):
            if(i!=j):
                if(dur2ieee[i][j]>=30):
                    nieeedur30+=1
                    break
    ''''for i in range(rows2):
        for j in range(rows2):
            if(i<j):
                if(unconn2[i][j]>=10):
                    nunconndur10+=1'''
                    
    dur30.append(sumdur30)
    dur10.append(sumdur10)
    dur20.append(sumdur20)
    dur40.append(sumdur40)
    dur60.append(sumdur60)
    dur50.append(sumdur50)
    dur30wifi.append(nwifidur30)
    dur30ieee.append(nieeedur30)
    #unconn10.append(nunconndur10)


# In[53]:


timelist=[]
for i in range(3602):
    timelist.append(i)


# In[54]:


#print(dur30ieee)


# In[55]:


#print(dur2)


# In[56]:


from scipy import stats
print("mean of Connected nodes  with connectivity more than 30 seconds with WiFi: ", np.mean(dur30wifi))
print("mean of Connected nodes  with connectivity more than 30 seconds with 802.11p: ", np.mean(dur30ieee))
#print("mean of unconnected nodes  for more than 10 sec at any instant: ", np.mean(unconn10))
print("median of Connected nodes  with connectivity more than 30 seconds with WiFi: ",np.median(dur30wifi))
print("median of Connected nodes  with connectivity more than 30 seconds with 802.11p: ",np.median(dur30ieee))
#print("median of unconnected nodes  for more than 10 sec at any instant: ", np.median(unconn10))
print("mode of Connected nodes  with connectivity more than 30 seconds with WiFi: ",stats.mode(dur30wifi))
print("mode of Connected nodes  with connectivity more than 30 seconds with 802.11p: ",stats.mode(dur30ieee))
#print("mode of unconnected nodes  for more than 10 sec at any instant: ", stats.mode(unconn10))


# In[57]:


plt.figure(figsize=(10, 10))
plt.plot(timelist, dur10, color='green', label="Duration>=10 sec")
plt.plot(timelist, dur20, color='red', label="Duration>=20 sec")
plt.plot(timelist, dur30, color='blue', label="Duration>=30 sec")
plt.plot(timelist, dur40, color='black', label="Duration>=40 sec")
plt.plot(timelist, dur50, color='pink', label="Duration>=50 sec")
plt.plot(timelist, dur60, color='orange', label="Duration>=60 sec")
plt.title('Total connections with duration greater than x seconds (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Total connections with duration>=x sec')
plt.legend()
plt.grid(True)
plt.show()


# In[58]:


#print(dur10)
#print(dur20)
#print(dur30)
#print(dur40)
#print(dur60)
#print(unconn10)


# In[59]:


dur30node=dur30.copy()
dur10node=dur10.copy()
dur20node=dur20.copy()
dur40node=dur40.copy()
dur60node=dur60.copy()
dur50node=dur50.copy()
#dur50node=dur50.copy()
for i in range(len(dur30node)):
    dur30node[i]=dur30node[i]/2
for i in range(len(dur10node)):
    dur10node[i]=dur10node[i]/2
for i in range(len(dur20node)):
    dur20node[i]=dur20node[i]/2
for i in range(len(dur40node)):
    dur40node[i]=dur40node[i]/2
for i in range(len(dur60node)):
    dur60node[i]=dur60node[i]/2
for i in range(len(dur50node)):
    dur50node[i]=dur50node[i]/2


# In[60]:


plt.figure(figsize=(10, 10))
plt.plot(timelist, dur10node, color='green', label="Duration>=10 sec")
plt.plot(timelist, dur20node, color='red', label="Duration>=20 sec")
plt.plot(timelist, dur30node, color='blue', label="Duration>=30 sec")
plt.plot(timelist, dur40node, color='black', label="Duration>=40 sec")
plt.plot(timelist, dur50node, color='pink', label="Duration>=50 sec")
plt.plot(timelist, dur60node, color='orange', label="Duration>=60 sec")
plt.title('Total nodes which are connected for more than x seconds vs Time (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Total nodes which are connected for more than x sec')
plt.legend()
plt.grid(True)
plt.show()


# In[61]:


xcdf1=np.sort(dur30)
ycdf1=np.arange(1,len(xcdf1)+1)/len(xcdf1)
plt.figure(figsize=(10, 10))
plt.title('Total connections with duration greater than 30 seconds cdf (Morning Dataset, random)')
plt.xlabel('total connections')
plt.ylabel('cdf')
plt.grid(True)
#plt.plot(xcdf1, ycdf1, marker='.', linestyle='none')
plt.plot(xcdf1,ycdf1,'k')


# In[62]:


plt.figure(figsize=(10, 10))
plt.plot(timelist, tcrand)
plt.title('Total connections with time (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Total connections')
plt.grid(True)
plt.show()


# In[63]:


wifidegdf2= pd.DataFrame(wifideg2)
ieeedegdf2= pd.DataFrame(ieeedeg2)
wifidegdf2=wifidegdf2.transpose()
ieeedegdf2=ieeedegdf2.transpose()
totalwificonn2=pd.DataFrame()
totalwificonn2["Total wifi"]=wifidegdf2.sum(axis=1)
totalieeeconn2=pd.DataFrame()
totalieeeconn2["Total ieee"]=ieeedegdf2.sum(axis=1)
totalwificonn2=totalwificonn2/2
totalieeeconn2=totalieeeconn2/2
totalconn2=pd.DataFrame()
totalconn2["Total connections"]=totalieeeconn2["Total ieee"]+totalwificonn2["Total wifi"]
totalconn2.reset_index(inplace=True)
totalconn2 = totalconn2.rename(columns = {'index':'Time'})


# In[64]:


plt.figure(figsize=(10, 10))
plt.plot(totalconn2['Time'], totalconn2["Total connections"])
plt.title('Total connections with time (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Total connections')
plt.grid(True)
plt.show()


# In[65]:


xcdf1=np.sort(totalconn2["Total connections"])
ycdf1=np.arange(1,len(xcdf1)+1)/len(xcdf1)
plt.figure(figsize=(10, 10))
plt.title('Total connections cdf (Morning Dataset, random)')
plt.xlabel('total connections at a time')
plt.ylabel('cdf')
plt.grid(True)
#plt.plot(xcdf1, ycdf1, marker='.', linestyle='none')
plt.plot(xcdf1,ycdf1,'k')


# In[66]:


row=l
timer=3602
sm2rand5=[] #number of nodes with connectivity >=5(removal of random 5%)
for i in range(timer):
    summ=0
    for j in range(row):
        if((wifidegdf2[j][i]+ieeedegdf2[j][i])>=5):
            summ+=1
    sm2rand5.append(summ)
sm2rand0=[] #number of nodes with connectivity >=0(removal of random 5%)
for i in range(timer):
    summ=0
    for j in range(row):
        if((wifidegdf2[j][i]+ieeedegdf2[j][i])>=0):
            summ+=1
    sm2rand0.append(summ) 
sm2rand4=[] #number of nodes with connectivity >=4(removal of random 5%)
for i in range(timer):
    summ=0
    for j in range(row):
        if((wifidegdf2[j][i]+ieeedegdf2[j][i])>=4):
            summ+=1
    sm2rand4.append(summ)
sm2rand3=[] #number of nodes with connectivity >=3(removal of random 5%)
for i in range(timer):
    summ=0
    for j in range(row):
        if((wifidegdf2[j][i]+ieeedegdf2[j][i])>=3):
            summ+=1
    sm2rand3.append(summ)
sm2rand2=[] #number of nodes with connectivity >=2(removal of random 5%)
for i in range(timer):
    summ=0
    for j in range(row):
        if((wifidegdf2[j][i]+ieeedegdf2[j][i])>=2):
            summ+=1
    sm2rand2.append(summ)
sm2rand1=[] #number of nodes with connectivity >=1(removal of random 5%)
for i in range(timer):
    summ=0
    for j in range(row):
        if((wifidegdf2[j][i]+ieeedegdf2[j][i])>=1):
            summ+=1
    sm2rand1.append(summ)
            


# In[67]:


#Morning dataset
plt.figure(figsize=(10, 10))
plt.plot(timelist, sm2rand1, color='orange', label="Connectivity>=1")
plt.plot(timelist, sm2rand2, color='purple', label="Connectivity>=2")
plt.plot(timelist, sm2rand3, color='green', label="Connectivity>=3")
plt.plot(timelist, sm2rand4, color='red', label="Connectivity>=4")
plt.plot(timelist, sm2rand5, color='blue', label="Connectivity>=5")
#plt.plot(timelist, sm2rand0, color='yellow', label="Connectivity>=0")
plt.title('Number of nodes with connectivity vs Time (Morning dataset)-removing 10% random nodes')
plt.xlabel('Time (sec)')
plt.ylabel('Number of nodes with connectivity')
plt.grid(True)
plt.legend()
plt.show()


# In[68]:


xcdf2=np.sort(sm2rand5)
ycdf2=np.arange(1,len(xcdf2)+1)/len(xcdf2)
plt.figure(figsize=(10, 10))
plt.title('Number of nodes with connectivity>=5 cdf')
plt.xlabel('number of nodes')
plt.ylabel('cdf')
plt.grid(True)
plt.plot(xcdf2,ycdf2,'k')

xcdf3=np.sort(sm2rand4)
ycdf3=np.arange(1,len(xcdf3)+1)/len(xcdf3)
plt.figure(figsize=(10, 10))
plt.title('Number of nodes with connectivity>=4 cdf')
plt.xlabel('number of nodes')
plt.ylabel('cdf')
plt.grid(True)
plt.plot(xcdf3,ycdf3,'k')

xcdf4=np.sort(sm2rand3)
ycdf4=np.arange(1,len(xcdf4)+1)/len(xcdf4)
plt.figure(figsize=(10, 10))
plt.title('Number of nodes with connectivity>=3 cdf')
plt.xlabel('number of nodes')
plt.ylabel('cdf')
plt.grid(True)
plt.plot(xcdf4,ycdf4,'k')

xcdf5=np.sort(sm2rand2)
ycdf5=np.arange(1,len(xcdf5)+1)/len(xcdf5)
plt.figure(figsize=(10, 10))
plt.title('Number of nodes with connectivity>=2 cdf')
plt.xlabel('number of nodes')
plt.ylabel('cdf')
plt.grid(True)
plt.plot(xcdf5,ycdf5,'k')

xcdf6=np.sort(sm2rand1)
ycdf6=np.arange(1,len(xcdf6)+1)/len(xcdf6)
plt.figure(figsize=(10, 10))
plt.title('Number of nodes with connectivity>=1 cdf')
plt.xlabel('number of nodes')
plt.ylabel('cdf')
plt.grid(True)
plt.plot(xcdf6,ycdf6,'k')


# In[69]:


plt.figure(figsize=(10, 10))
plt.title('Number of nodes with connectivity cdf (Morning Dataset, random)')
plt.xlabel('number of nodes')
plt.ylabel('cdf')
plt.grid(True)
plt.plot(xcdf6,ycdf6,'k', color='orange', label="Connectivity>=1")
plt.plot(xcdf5,ycdf5,'k', color='violet', label="Connectivity>=2")
plt.plot(xcdf4,ycdf4,'k', color='green', label="Connectivity>=3")
plt.plot(xcdf3,ycdf3,'k', color='blue', label="Connectivity>=4")
plt.plot(xcdf2,ycdf2,'k', color='red', label="Connectivity>=5")
plt.legend()


# In[70]:


plt.figure(figsize=(10, 10))
plt.plot(timelist, tcrand, color='green', label="Total connections")
plt.plot(timelist, sm2rand1, color='red', label="Number of connected nodes")
plt.title('Total connections with time and number of nodes connected vs time (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Total connections')
plt.grid(True)
plt.legend()
plt.show()


# In[71]:


#print(sm2rand1)


#  print(sm2rand1)

# In[72]:


#print(ieeedegdf2)


# In[73]:


column_names = ["v1", "v2", "wifidur", "ieeedur", "celldur", "starttimewifi", "starttimeieee", "starttimecell", "endtimewifi", "endtimeieee", "endtimecell"]
durdata = pd.DataFrame(columns = column_names)


# In[74]:


print(durdata)


# In[75]:


for i in range(rows):
    for j in range(cols):
        if(i<j):
            durdata= durdata.append({'v1':i,'v2':j,'wifidur':wifidur[i][j],'ieeedur':ieeedur[i][j],'celldur':celldur[i][j], 'starttimewifi': starttimewifi[i][j], 'starttimeieee': starttimeieee[i][j], 'starttimecell': starttimecell[i][j], 'endtimewifi': starttimewifi[i][j]+wifidur[i][j], 'endtimeieee': starttimeieee[i][j]+ieeedur[i][j], 'endtimecell': starttimecell[i][j]+celldur[i][j]},ignore_index=True)


# In[76]:


#print(durdata)


# In[77]:


#durdata.to_csv('Desktop/Data/Duration100-data.csv')


# In[78]:


wifidegdf= pd.DataFrame(wifideg)
ieeedegdf= pd.DataFrame(ieeedeg)
celldegdf=pd.DataFrame(celldeg)


# In[79]:


wifidegdf=wifidegdf.transpose()
ieeedegdf=ieeedegdf.transpose()
celldegdf=celldegdf.transpose()


# In[80]:


#print(wifidegdf)


# In[81]:


#wifidegdf.to_csv('Desktop/Data/Wifi-degree100.csv', index=True)
#ieeedegdf.to_csv('Desktop/Data/IEEE-degree100.csv', index=True)
#celldegdf.to_csv('Desktop/Data/Cellular-degree100.csv', index=True)


# In[82]:


totalwificonn=pd.DataFrame()
totalwificonn["Total wifi"]=wifidegdf.sum(axis=1)


# In[84]:


totalieeeconn=pd.DataFrame()
totalieeeconn["Total ieee"]=ieeedegdf.sum(axis=1)
#print(totalieeeconn)


# In[85]:


totalwificonn=totalwificonn/2
totalieeeconn=totalieeeconn/2


# In[88]:


totalconn=pd.DataFrame()
totalconn["Total connections"]=totalieeeconn["Total ieee"]+totalwificonn["Total wifi"]
#print(totalconn)


# In[89]:


totalconn.reset_index(inplace=True)
totalconn = totalconn.rename(columns = {'index':'Time'})
#print(totalconn)


# In[90]:


totalwificonn.reset_index(inplace=True)
totalwificonn = totalwificonn.rename(columns = {'index':'Time'})


# In[92]:


totalieeeconn.reset_index(inplace=True)
totalieeeconn = totalieeeconn.rename(columns = {'index':'Time'})
#print(totalieeeconn)


# In[93]:


plt.figure(figsize=(10, 10))
plt.plot(totalconn['Time'], totalconn["Total connections"])
plt.title('Total connections with time')
plt.xlabel('Time (sec)')
plt.ylabel('Total connections')
plt.grid(True)
plt.show()


# In[94]:


twc=totalwificonn['Total wifi']
tic=totalieeeconn['Total ieee']


# In[95]:


twc2=pd.DataFrame(columns={"wifiConnectivity>=5"})
tic2=pd.DataFrame(columns={"ieeeConnectivity>=5"})


# In[96]:


print(twc[0]>5)


# In[97]:


twc2=twc2.append({'wifiConnectivity>=5':0},ignore_index=True)


# In[98]:


print(twc2)


# In[99]:


''''for i in range (302):
    if(twc[i]>=5):
        twc2= twc2.append({'wifiConnectivity>=5':twc[i]}, ignore_index=True)
    else twc2=twc2.append({'wifiConnectivity>=5':0},ignore_index=True)
    if(tic[i]>=5):
        tic2=tic2.append({'ieeeConnectivity>=5':tic[i]}, ignore_index=True)
    else tic2=tic2.append({'ieeeConnectivity>=5':0},ignore_index=True)'''


# In[100]:


print(twc2)


# In[101]:


xr=df1['vehicle_x'][0] #x coordinate reference
yr=df1['vehicle_y'][0] #y coordinate reference
print(xr)
print(yr)


# In[102]:


xc= df1['vehicle_x'].copy()
yc= df1['vehicle_y'].copy()


# In[103]:


r=0;
vd=[] #vehicle density
tc=[] #total connections
while r<=400:
    area=pi*r*r
    nv=1 #number of vehicles
    vid=[0] #vehicle id
    for i in range (1,l):
        dist=sqrt((xr-xc[i])*(xr-xc[i]) + (yr-yc[i])*(yr-yc[i]))
        if(dist<=r):
            nv+=1
            vid.append(i)
    vd.append(nv/pi)
    row=len(vid)
    c=0 #total connections in the area
    for i in range (row):
        for j in range (row):
            if(i!=j):
                dist=sqrt((xc[vid[i]]-xc[vid[j]])*(xc[vid[i]]-xc[vid[j]]) + (yc[vid[i]]-yc[vid[j]])*(yc[vid[i]]-yc[vid[j]]))
                if(dist<=400):
                    c+=1
    
    tc.append(c/2)
    r+=2
            
    
    


# In[104]:


plt.figure(figsize=(10, 10))
plt.plot(vd, tc)
plt.title('Connections vs vehicle density plot')
plt.xlabel('Vehicle density')
plt.ylabel('Total connections in area')
plt.grid(True)
plt.show()


# In[105]:


timer=3602
sm5=[] #number of nodes with connectivity >=5
for i in range(timer):
    summ=0
    for j in range(l):
        if((wifidegdf[j][i]+ieeedegdf[j][i])>=5):
            summ+=1
    sm5.append(summ)
sm4=[] #number of nodes with connectivity >=4
for i in range(timer):
    summ=0
    for j in range(l):
        if((wifidegdf[j][i]+ieeedegdf[j][i])>=4):
            summ+=1
    sm4.append(summ)
sm3=[] #number of nodes with connectivity >=3
for i in range(timer):
    summ=0
    for j in range(l):
        if((wifidegdf[j][i]+ieeedegdf[j][i])>=3):
            summ+=1
    sm3.append(summ)
sm2=[] #number of nodes with connectivity >=2
for i in range(timer):
    summ=0
    for j in range(l):
        if((wifidegdf[j][i]+ieeedegdf[j][i])>=2):
            summ+=1
    sm2.append(summ)
sm1=[] #number of nodes with connectivity >=1
for i in range(timer):
    summ=0
    for j in range(l):
        if((wifidegdf[j][i]+ieeedegdf[j][i])>=1):
            summ+=1
    sm1.append(summ)
sm0=[] #number of nodes with connectivity >=0
for i in range(timer):
    summ=0
    for j in range(l):
        if((wifidegdf[j][i]+ieeedegdf[j][i])>=0):
            summ+=1
    sm0.append(summ)
            
        


# In[106]:


#Morning dataset
plt.figure(figsize=(10, 10))
plt.plot(timelist, sm0, color='yellow', label="Connectivity>=0")
plt.plot(timelist, sm1, color='orange', label="Connectivity>=1")
plt.plot(timelist, sm2, color='purple', label="Connectivity>=2")
plt.plot(timelist, sm3, color='green', label="Connectivity>=3")
plt.plot(timelist, sm4, color='red', label="Connectivity>=4")
plt.plot(timelist, sm5, color='blue', label="Connectivity>=5")
plt.title('Number of nodes with connectivity vs Time (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Number of nodes with connectivity')
plt.grid(True)
plt.legend()
plt.show()


# In[107]:


plt.figure(figsize=(10, 10))
plt.plot(timelist, sm5, color='blue', label="Connectivity>=5")
plt.title('Number of nodes with connectivity>= vs Time (Morning Dataset, random)')
plt.xlabel('Time (sec)')
plt.ylabel('Number of nodes with connectivity')
plt.grid(True)
plt.legend()
plt.show()


# In[109]:


#totalconndf= totalconn[:,['Time, Total connections']].reset_index(inplace=True)

