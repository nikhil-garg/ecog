## Data from MATLAB to python
import os
import math
#os.chdir("C:\Users\Ritwik")
import matplotlib
import numpy as np
import pandas as pd
def Insert_row(row_number, df, row_value):
    start_upper = 0
    end_upper = row_number
   
    start_lower = row_number
    end_lower = df.shape[0]
    upper_half = [*range(start_upper, end_upper, 1)]

    lower_half = [*range(start_lower, end_lower, 1)]
    lower_half = [x.__add__(1) for x in lower_half]
    index_ = upper_half + lower_half
    df.index = index_
   
    
    df.loc[row_number] = row_value
    
    
    df = df.sort_index()
   
   
    return df

def car(data):
    data=data.astype(float)
    transflag=0
    if (np.size(data, 0)<np.size(data, 1)):
        data=np.transpose(data)
        transflag=1
        
    num_chans=np.size(data,1)
    spatfiltmatrix=np.ones((num_chans, num_chans), dtype=float)*(-1)
    spatfiltmatrix=pd.DataFrame(spatfiltmatrix, columns='c' + pd.Series(np.arange(1,num_chans+1)).astype(str))
    for i in range(num_chans):
        spatfiltmatrix.iat[i,i]=num_chans-1
    
    spatfiltmatrix=spatfiltmatrix/num_chans
    
    #perform spatial filtering
    if(data.empty!=1):
        print("Spatial Filtering\n")
        data = np.matmul(data.values,spatfiltmatrix.values)
        if (np.size(data, 1) != np.size(spatfiltmatrix, 0)):
            print("The first dimension in the spatial filter matrix has to equal the second dimension in the data")

    if transflag == 1:
        data = np.transpose(data)
        
    return data

df1=pd.read_csv('neuro_data.csv')
k=list(df1.columns)
for i in range(len(k)):
    k[i]=int(k[i])

row_value = k
df1 = Insert_row(0, df1, row_value)

col_list = 'c' + pd.Series(np.arange(1,len(df1.columns)+1)).astype(str)
df1.columns=col_list

df2=pd.read_csv('neuro_stim.csv')
k=list(df2.columns)
for i in range(len(k)):
    k[i]=int(k[i])

row_value = k
df2 = Insert_row(0, df2, row_value)

col_list = 'c' + pd.Series(np.arange(1,len(df2.columns)+1)).astype(str)
df2.columns=col_list

## --- data matlab fininsh ---- ##

## set defaults

srate=1000
#warning('off', 'signal:psd:PSDisObsolete') ---need to convert to python---
BF_correct='n'
#subjects=['bp', 'ca', 'cc', 'de', 'fp', 'gc', 'gf', 'hh', 'hl', 'jc', 'jf', 'jm', 'jp', 'jt', 'rh', 'rr', 'ug', 'wc', 'zt']

subjects=['bp',
    'ca',
    'cc',
    'de',
    'fp',
    'gc',
    'gf',
    'hh',
    'hl',
    'jc',
    'jf',
    'jm',
    'jp',
    'jt',
    'rh',
    'rr',
    'ug',
    'wc',
    'zt'
]



#print(subjects)
print(df1.head(2))
#print(len(subjects))

def car(data):
    data=data.astype(float)
    transflag=0
    if (np.size(data, 0)<np.size(data, 1)):
        data=np.transpose(data)
        transflag=1
        
    num_chans=np.size(data,1)
    spatfiltmatrix=np.ones((num_chans, num_chans), dtype=float)*(-1)
    spatfiltmatrix=pd.DataFrame(spatfiltmatrix, columns='c' + pd.Series(np.arange(1,num_chans+1)).astype(str))
    for i in range(num_chans):
        spatfiltmatrix.iat[i,i]=num_chans-1
    
    spatfiltmatrix=spatfiltmatrix/num_chans
    
    #perform spatial filtering
    if(data.empty!=1):
        print("Spatial Filtering\n")
        data = np.matmul(data.values,spatfiltmatrix.values) 
        if (np.size(data, 1) != np.size(spatfiltmatrix, 0)):
            print("The first dimension in the spatial filter matrix has to equal the second dimension in the data")

    if transflag == 1:
        data = np.transpose(data)
        
    return data
    
df1=car(df1)
df1=pd.DataFrame(df1, columns='c' + pd.Series(np.arange(1,np.size(df1,1)+1)).astype(str))

a=pd.DataFrame(np.array(df2.iloc[1:571720])-np.array(df2.iloc[0:571719]))
a.columns=['c1']
a=Insert_row(0,a,0)

def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

a1=pd.DataFrame(getIndexes(a, -12))
a1.columns=['c1']
a2=pd.DataFrame(getIndexes(a, -11))
a2.columns=['c1']
a3=pd.DataFrame(getIndexes(a, 11))
a3.columns=['c1']
a4=pd.DataFrame(getIndexes(a, 12))
a4.columns=['c1']

if (np.size(a1, 0)==np.size(a4, 0)):
    blocksize=(abs(a1-a4))['c1'].mean()
    
elif (np.size(a2, 0)==np.size(a3,0)):
    blocksize=(abs(a2-a3))['c1'].mean()
    
else:
    print("can''t determine blocksize b/c task cut between between task block','can''t determine blocksize b/c task cut between between task block")

if math.floor(blocksize/1000)!=(blocksize/1000):
    print('can''t determine blocksize b/c task cut between between task block','can''t determine blocksize b/c task cut between between task block')

#find rest conditions specific to behavior that preceded them
k=np.concatenate([np.zeros((int(blocksize),1)),df2.values[0:(np.size(df2,0)-int(blocksize))]])
ull=np.where(k==12)[0]
bull=np.where(df2.values==0)[0]
df2.values[np.intersect1d(ull, bull),0]=120 #hand rest block condition
ull=np.where(k==11)[0]
df2.values[np.intersect1d(ull, bull),0]=110 #tongue rest block condition
    
#shift stim by half-second to account for behavioral lag
#stim=[zeros(500,1); stim(1:(end-500),1)]

trialnr=0*df2.values
tr_sc=0 #initialize
trtemp=1;
trialnr[0,0]=trtemp
for n in range(1,np.size(df2,0)):
    if (df2.values[n,0] != df2.values[n-1,0]):
        trtemp=trtemp+1
        tr_sc=np.array([tr_sc, df2.values[n,0]])
    
    trialnr[n,0]=trtemp

trials=np.unique(trialnr)    

# Calculate spectra for all trials
num_chans = len(df1.values[0,:]) # Length of row being used, Note: If matrix row has variable lengths then 
mean_PSD=np.zeros([200, 48]),0
all_PSD = np.empty((100,100)) # Change to = [] if using list. Np matrices are faster than lists

for cur_trial in range(0, max(trials)):
    # isolate relevant data
    curr_data = df1.values[np.where(trialnr==cur_trial)[0] , :] 
    
    # keep only .5 to block end of data, for each block
    if ((len(curr_data[:,0])) >= blocksize): 
        curr_data = curr_data[(math.floor(srate/2)+1):blocksize,:]
    
    block_PSD = np.empty((100,100)) 
    for p in range(0,num_chans):
        temp_PSD = matplotlib.pyplot.psd(curr_data[:,p],srate,srate,math.floor(srate*.25),math.floor(srate/10))
        block_PSD = np.concatenate((block_PSD, temp_PSD), axis = 1) #Change if using lists
    
    block_PSD = block_PSD[1:200,:] # downsample - we only want to keep spectra up to 200 hz, not all of the others
    mean_PSD = mean_PSD + block_PSD
    all_PSD=np.concatenate((all_PSD,block_PSD), axis = 3)
 
mean_PSD = np.divide(mean_PSD,max(trials)) 
freq_bins = [range(1:200)]
