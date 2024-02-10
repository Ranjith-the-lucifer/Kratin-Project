#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[3]:


# Load the training dataset
train_data = pd.read_csv("F:/archive (8)/train.csv")

# Load the testing dataset
test_data = pd.read_csv("F:/archive (8)/test.csv")


# In[5]:


print(train_data.columns)


# In[6]:


selected_features_train = train_data[['Protein_g','Fat_g','Carb_g', 'Sugar_g',]]
selected_features_test = test_data[['Protein_g','Fat_g','Carb_g', 'Sugar_g',]]


# In[7]:


# Standardize the features separately for training and testing sets
scaler = StandardScaler()
standardized_data_train = scaler.fit_transform(selected_features_train)
standardized_data_test = scaler.transform(selected_features_test)


# In[8]:


# Choose the number of clusters (you may need to experiment with this)
num_clusters = 4


# In[9]:


# Apply K-Means clustering on the training set
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
train_data['Cluster'] = kmeans.fit_predict(standardized_data_train)


# In[10]:


# Make predictions on the test set
test_data['Cluster'] = kmeans.predict(standardized_data_test)


# In[12]:


#Evaluate the model using silhouette score (a higher score indicates better-defined clusters)
silhouette_avg = silhouette_score(standardized_data_train, train_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.2f}")


# In[13]:


# Print cluster centers for training set (representative nutrient values for each cluster)
print("\nCluster Centers (Training Set):")
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_features_train.columns))


# In[14]:


# Explore the resulting clusters for both training and testing sets
cluster_analysis_train = train_data.groupby('Cluster').mean()
cluster_analysis_test = test_data.groupby('Cluster').mean()


# In[15]:


print("\nCluster Analysis (Training Set):")
print(cluster_analysis_train)


# In[16]:


print("\nCluster Analysis (Testing Set):")
print(cluster_analysis_test)


# # For healthier and live longer , i suggest(on above basis algorithm's result) to sunitha and 65 + ages , the following are given below
# # Protein: The recommended daily intake for adults over 65 is 46-56 grams per day. However, individuals with certain health conditions (e.g., kidney disease) may need different amounts.
# # Fat: Aim for 20-35% of daily calories from fat, prioritizing unsaturated fats like those found in olive oil, nuts, and avocados. Limit saturated and trans fats.
# # Carbohydrates: Consume 45-65% of daily calories from carbohydrates, choosing whole grains, fruits, and vegetables over refined grains and sugary drinks.
# # Sugar: The American Heart Association recommends limiting added sugar to no more than 25 grams per day for women and 36 grams per day for men.

# In[ ]:




