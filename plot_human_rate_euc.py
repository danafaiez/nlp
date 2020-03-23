import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats


#data = pd.read_csv("/Users/danafaiez/Desktop/Human_rated_Data_matrix_updated.csv") 
#data = pd.read_csv("/Users/danafaiez/Desktop/test.csv")
data = pd.read_csv("/Users/danafaiez/Desktop/Human_rated_cos_norm_euc_Data_matrix_new.csv")

length = len(data)
hr=list(data.iloc[1:length,3].astype('float'))
be=list(data.iloc[1:length,6].astype('float'))
we=list(data.iloc[1:length,7].astype('float'))

be_pos=[]
hr_bert_pos=[]
for i in range(len(be)):
    if be[i] >=0:
        be_pos.append(be[i])
        hr_bert_pos.append(hr[i])

we_pos=[]
hr_w2v_pos=[]
for i in range(len(we)):
    if we[i] >=0:
        we_pos.append(we[i])
        hr_w2v_pos.append(hr[i])

norm_hr =  1-np.divide(hr,10)

norm_hr_bert_pos = 1-np.divide(hr_bert_pos,10)   
norm_hr_w2v_pos = 1-np.divide(hr_w2v_pos,10)

plt.subplot(111)
plt.scatter(norm_hr,be,marker = '.',color='pink',label="BERT normalized Eucl.")
plt.scatter(norm_hr,we,s=12, facecolors='none', edgecolors='skyblue',label="Word2Vec normalized Eucl.")

plt.ylabel("Euclidean")
plt.xlabel("1 - ave normalized human ratings")

slope_bert, intercept_bert, r_value_bert, p_value_bert, std_err_bert = stats.linregress(norm_hr_bert_pos,be_pos)
slope_w2v, intercept_w2v, r_value_w2v, p_value_w2v, std_err_w2v = stats.linregress(norm_hr_w2v_pos,we_pos)

#straight line fit:
x = np.linspace(-0.04,1,100)
##BERT
y_bert = slope_bert*x+intercept_bert
plt.plot(x, y_bert, ':k', label='BERT FIT')
##W2V
y_w2v = slope_w2v*x+intercept_w2v
plt.plot(x, y_w2v, '--k', label='W2V FIT')
#streight line reference slope=1
y = x
plt.plot(x, y, '-.b', label='slope=1')

plt.legend()
plt.show()

#Pearson's correlation coef:
print("pearsonr(norm_hr,be):",stats.pearsonr(norm_hr_bert_pos,be_pos))
print("pearsonr(norm_hr,we):",stats.pearsonr(norm_hr_w2v_pos,we_pos))

"""
print('BERT: p_value: ' + str(p_value_bert) + '; r_value: ' + str(r_value_bert)
        + 'std: ' +str(std_err_bert))
print('W2V without ones: p_value: ' + str(p_value_w2v) + '; r_value: ' + str(r_value_w2v) + 'std: ' +str(std_err_w2v))
print('W2V without ones: p_value: ' + str(p_value_new_w2v) + '; r_value: ' + str(r_value_new_w2v) + 'std: ' + str( std_err_new_w2v))
"""


#Scaled data:
be_scaled = (be_pos-intercept_bert)/((slope_bert*1+intercept_bert)-intercept_bert)
we_scaled = (we_pos-intercept_w2v)/((slope_w2v*1+intercept_w2v)-intercept_w2v)

plt.subplot(111)
plt.scatter(norm_hr_bert_pos,be_scaled,marker = '.',color='g', label="Scaled +BERT Euc")
plt.scatter(norm_hr_w2v_pos,we_scaled,s=12, facecolors='none', edgecolors='r',label="Scaled +Word2Vec Euc")

#Pearson's correlation coef:
print("pearsonr(scaled +BERT):", stats.pearsonr(norm_hr_bert_pos,be_scaled))
print("pearsonr(scaled +w2v):",  stats.pearsonr(norm_hr_w2v_pos,we_scaled))


#straight line fit fo scaled data:
scaled_slope_bert, scaled_intercept_bert, scaled_r_value_bert, scaled_p_value_bert, scaled_std_err_bert = stats.linregress(norm_hr_bert_pos,be_scaled)
scaled_slope_w2v, scaled_intercept_w2v, scaled_r_value_w2v, scaled_p_value_w2v, scaled_std_err_w2v = stats.linregress(norm_hr_w2v_pos,we_scaled)

x = np.linspace(-0.04,1,100)
##BERT SCALED
y_bert_scaled = scaled_slope_bert*x+scaled_intercept_bert
plt.plot(x, y_bert_scaled, ':k', label='scaled BERT FIT')
##W2V SCALED
y_w2v_scaled = scaled_slope_w2v*x+scaled_intercept_w2v
plt.plot(x, y_w2v_scaled, '--k', label='scaled W2V FIT')
#streight line reference slope=1
y = x
plt.plot(x, y, '-.b', label='slope=1')
plt.legend()
plt.show()
