#!/usr/bin/env python
# coding: utf-8

# ### Load your dataset. 

# In[8]:


import pandas as pd
df= pd.read_csv("CC GENERAL.csv")
df.drop("CUST_ID",axis=1,inplace=True)
df['MINIMUM_PAYMENTS'].fillna(value=df['MINIMUM_PAYMENTS'].mean(),inplace=True)

df['CREDIT_LIMIT'].fillna(value=50,inplace=True)
df.info()
df


# ### 2. Use hierarchical clustering to identify the inherent groupings within your data.

# In[73]:


from sklearn.cluster import AgglomerativeClustering ##Importing our clustering algorithm : Agglomerative
model=AgglomerativeClustering(n_clusters=5)
clust_labels=model.fit_predict(df)  


# In[74]:


agglomerative=pd.DataFrame(clust_labels)
agglomerative


# 3-Plot the clusters.

# In[76]:


import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter (df['PAYMENTS'] , df["PURCHASES"] , c= agglomerative[0], s=50)
ax.set_title("Agglomerative Clutering")
ax.set_xlabel("PAYMENTS")
ax.set_ylabel("PURCHASES")
plt.colorbar(scatter)


# OBJECTIVES
# K-means & Hierarchical clustering 
# This case requires to develop a customer segmentation to define marketing strategy. The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.
# 
# Following is the Data Dictionary for Credit Card dataset :
# 
# CUST_ID: Identification of Credit Card holder (Categorical)
# 
# BALANCE: Balance amount left in their account to make purchases 
# BALANCE_FREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated) 
# 
# PURCHASES: Amount of purchases made from account 
# 
# ONEOFF_PURCHASES: Maximum purchase amount done in one-go
# 
# INSTALLMENTS_PURCHASES: Amount of purchase done in installment
# 
# CASH_ADVANCE: Cash in advance given by the user
# 
# PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# 
#  ONEOFFPURCHASESFREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# 
# PURCHASESINSTALLMENTSFREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# 
# CASHADVANCEFREQUENCY: How frequently the cash in advance being paid 
# 
# CASHADVANCETRX: Number of Transactions made with "Cash in Advanced" 
# 
# PURCHASES_TRX: Number of purchase transactions made
# 
# CREDIT_LIMIT: Limit of Credit Card for user 
# 
# PAYMENTS: Amount of Payment done by user 
# 
# MINIMUM_PAYMENTS: Minimum amount of payments made by user 
# 
# PRCFULLPAYMENT: Percent of full payment paid by user
# 
# TENURE: Tenure of credit card service for user
# 
# 1. Load your dataset. 
# 
# 2. Use hierarchical clustering to identify the inherent groupings within your data.
# 
# 3. Plot the clusters. 
# 
# 4. Plot the dendrogram. Use k-means clustering. 
# 
# 5. Try different k values and select the best one. 
# 
# 6. Plot the clusters. 
# 
# 7. Compare the two results. 
# 
# Bonus: search for another validation metric

# 

# 4-Plot the dendrogram.
# 

# In[37]:


import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("Customer Dendrograms")
dend=shc.dendrogram(shc.linkage(df,method='ward'))


# ### Use k-means clustering.

# In[65]:


from sklearn.cluster import KMeans  #Importing our clustering algorithm: KMeans
kmeans=KMeans(n_clusters=5, random_state=0)  #Cluster our data by choosing 5 as number of clusters
kmeans.fit(df)
df


# In[43]:


labels2=pd.DataFrame(kmeans.labels_)
labels2


# In[66]:


kmeans.predict(df)
print(kmeans.cluster_centers_)
df


# 5-Try different k values and select the best one.

# In[48]:


sum_of_squared_distance=[]
k=range(1,15)
for ka in k:
    km=KMeans(n_clusters=ka)
    km=km.fit(df)
    sum_of_squared_distance.append(km.inertia_)
    


# In[55]:



plt.plot(k,sum_of_squared_distance)
plt.xlabel('k')
plt.ylabel('sum_of_squared_distance')
plt.title('Elbow method for optimal k')
plt.show()


# ### the best k is 5

# 6-Plot the clusters.
# 

# In[ ]:





# In[78]:



plt.scatter(df["PAYMENTS"][labels2 [0]==0],          
            df["PURCHASES"][labels2 [0]==0],s=80,c='magenta')
plt.scatter(df["PAYMENTS"][labels2 [0]==1],          
            df["PURCHASES"][labels2 [0]==1],s=80,c='yellow')
plt.scatter(df["PAYMENTS"][labels2 [0]==2],          
            df["PURCHASES"][labels2 [0]==2],s=80,c='green')
plt.scatter(df["PAYMENTS"][labels2 [0]==3],          
            df["PURCHASES"][labels2 [0]==3],s=80,c='cyan')
plt.scatter(df["PAYMENTS"][labels2 [0]==4],          
            df["PURCHASES"][labels2 [0]==4],s=80,c='burlywood')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')
plt.title('Clusters of Customers')
plt.xlabel('PAYMENTS')
plt.ylabel('PURCHASES')
plt.legend()
plt.show()


# ### there are no big difference between  hierarchical clustering plot and the k-means clustering plot when number of cluster is 5
