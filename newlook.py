#!/usr/bin/env python
# coding: utf-8

# ### Data plotting

# In[53]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv("dataset adm pyhedu.csv")


# In[8]:


df.head()


# In[9]:


df_new=df.head(7)


# In[10]:


print(df_new)


# In[16]:


# Set the 'TV' column as predictor variable
x = df[['IPS 1']].values

# Set the 'Sales' column as response variable 
y = df['IPK'].values


# In[17]:


# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(x,y)

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel('IPS 1')
plt.ylabel('IPK')
# Add plot title 
plt.title('IPS 1 vs IPK')
plt.show()


# In[18]:


sns.pairplot(df)


# In[19]:


sns.distplot(df['IPK'])


# In[44]:


sns.heatmap(df.corr(), cmap='coolwarm')


# In[81]:


df.corr()


# ### part 1: knn for k=1

# In[39]:


# Get a subset of the data i.e. rows 5 to 13
# Use the IPS column as the predictor, bcs ips 3 have the biggest correlation with ipk with 0.8
x_true = df_new['IPS 3'].iloc[5:13]

# Use the IPK column as the response
y_true = df_new['IPK'].iloc[5:13]

# Sort the data to get indices ordered from lowest to highest IP values
idx = np.argsort(x_true).values 

# Get the predictor data in the order given by idx above
x_true  = x_true.iloc[idx].values

# Get the response data in the order given by idx above
y_true  = y_true.iloc[idx].values


# In[40]:


# Define a function that finds the index of the nearest neighbor 
# and returns the value of the nearest neighbor.  
# Note that this is just for k = 1 where the distance function is 
# simply the absolute value.

def find_nearest(array,value):
    
    # Hint: To find idx, use .idxmin() function on the series
    idx = pd.Series(np.abs(array-value)).idxmin() 

    # Return the nearest neighbor index and value
    return idx, array[idx]


# In[41]:


# Create some synthetic x-values (might not be in the actual dataset)
x = np.linspace(np.min(x_true), np.max(x_true))

# Initialize the y-values for the length of the synthetic x-values to zero
y = np.zeros((len(x)))


# In[42]:


# Apply the KNN algorithm to predict the y-value for the given x value
for i, xi in enumerate(x):

    # Get the Sales values closest to the given x value
    y[i] = y_true[find_nearest(x_true, xi )[0]]


# ### Plotting data

# In[43]:


# Plot the synthetic data along with the predictions    
plt.plot(x, y, '-.')

# Plot the original data using black x's.
plt.plot(x_true, y_true, 'kx')

# Set the title and axis labels
plt.title('IPK vs IPS 3')
plt.xlabel('IPS 3')
plt.ylabel('IPK')


# ### part 2: knn for using k≥1

# In[76]:


# Read the data from the file "Advertising.csv"
data_filename = 'dataset adm pyhedu.csv'
df = pd.read_csv(data_filename)

# Set 'TV' as the 'predictor variable'   
x = df[['IPK']].values

# Set 'Sales' as the response variable 'y' 
y = df['IPS 3'].values


# In[77]:


# Split the dataset in training and testing with 60% training set 
# and 40% testing set with random state = 42
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.60,random_state=42)


# In[78]:


# Choose the minimum k value based on the instructions given on the left
k_value_min = 1

# Choose the maximum k value based on the instructions given on the left
k_value_max = 70

# Create a list of integer k values betwwen k_value_min and k_value_max using linspace
k_list = np.linspace(k_value_min, k_value_max, 70)


# In[79]:


##model build
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor


# In[80]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Ensure j is initialized
j = 0  

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6)) 

# Loop over all the k values
for k_value in k_list:   
   
    # Creating a kNN Regression model 
    model = KNeighborsRegressor(n_neighbors=int(k_value))  
    
    # Fitting the regression model on the training data 
    model.fit(x_train, y_train)  
    
    # Use the trained model to predict on the test data  
    y_pred = model.predict(x_test)  

    # Helper code to plot the data along with the model predictions
    colors = ['grey', 'r', 'b']
    if k_value in [1, 10, 70]:
        xvals = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        ypreds = model.predict(xvals)
        
        ax.plot(xvals, ypreds, '-', label=f'k = {int(k_value)}', linewidth=j+2, color=colors[j])
        j += 1  # Increment j for color selection

# Final plot settings
ax.legend(loc='lower right', fontsize=20)
ax.plot(x_train, y_train, 'x', label='train', color='k')
ax.set_xlabel('IPK', fontsize=20)
ax.set_ylabel('IPS 3', fontsize=20)
plt.tight_layout()  
plt.show()  # Show the plot


# In[64]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Setup a grid for plotting the data and predictions
fig, ax = plt.subplots(figsize=(10,6))

# Create a dictionary to store the k value against MSE fit {k: MSE@k} 
knn_dict = {}

# Variable used for altering the linewidth of values kNN models
j = 0

# Loop over all k values
for k_value in k_list:   
    
    # Create a KNN Regression model for the current k
    model = KNeighborsRegressor(n_neighbors=int(k_value))  
    
    # Fit the model on the train data
    model.fit(x_train, y_train)
    
    # Use the trained model to predict on the test data
    y_pred = model.predict(x_test)  # 
    
    # Calculate the MSE of the test data predictions
    MSE = mean_squared_error(y_test, y_pred)  # 
    
    # Store the MSE values of each k value in the dictionary
    knn_dict[k_value] = MSE  # 
    
    # Helper code to plot the data and various kNN model predictions
    colors = ['grey', 'r', 'b']
    if k_value in [1, 10, 70]:
        xvals = np.linspace(x.min(), x.max(), 100).reshape(-1,1)  # 
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds, '-', label=f'k = {int(k_value)}', linewidth=j+2, color=colors[j])
        j += 1
        
ax.legend(loc='lower right', fontsize=20)
ax.plot(x_train, y_train, 'x', label='test', color='k')
ax.set_xlabel('IPK', fontsize=20)
ax.set_ylabel('IPS 3', fontsize=20)
plt.tight_layout()


# In[65]:


# Plot a graph which depicts the relation between the k values and MSE
plt.figure(figsize=(8,6))

# Extract k-values and corresponding MSE values from the dictionary
plt.plot(list(knn_dict.keys()), list(knn_dict.values()), 'k.-', alpha=0.5, linewidth=2)  # 

# Set the title and axis labels
plt.xlabel('k', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('Test $MSE$ values for different k values - KNN regression', fontsize=20)
plt.tight_layout()
plt.show()


# In[66]:


### edTest(test_mse) ###

# Find the lowest MSE among all the kNN models
min_mse = min(knn_dict.values())  

# Use list comprehensions to find the k value associated with the lowest MSE
best_model = [key for key, value in knn_dict.items() if value == min_mse]  

# Print the best k-value
print("The best k value is", best_model, "with a MSE of", min_mse)


# ### compute to see the best model

# In[82]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Helper code to compute the R2_score of your best model
best_k = best_model[0]  
model = KNeighborsRegressor(n_neighbors=int(best_k))  

# Train the model
model.fit(x_train, y_train)

# Predict on test data
y_pred_test = model.predict(x_test)

# Compute and print the R2 score
print(f"The R2 score for your model is {r2_score(y_test, y_pred_test):.4f}")  


# In[ ]:




