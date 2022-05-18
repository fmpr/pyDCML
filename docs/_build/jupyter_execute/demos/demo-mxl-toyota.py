#!/usr/bin/env python
# coding: utf-8

# # Mixed logit with Toyota data
# 
# This demo uses the dataset that was made available by Kenneth Train at https://eml.berkeley.edu/~train/ec244ps.html
# 
# The data represent consumers' choices among vehicles in stated preference experiments. The data is from a study that Kenneth Train did for Toyota and GM to assist them in their analysis of the potential marketability of electric and hybrid vehicles, back before hybrids were introduced.
# 
# We begin by performing the necessary imports:

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fix random seed for reproducibility
np.random.seed(42)


# ## Load Toyota dataset
# 
# About the data:
# 
# In each choice experiment, the respondent was presented with three vehicles, with the price and other attributes of each vehicle described. The respondent was asked to state which of the three vehicles he/she would buy if the these vehicles were the only ones available in the market. There are 100 respondents in our dataset (which, to reduce estimation time, is a subset of the full dataset which contains 500 respondents.) Each respondent was presented with 15 choice experiments, and most respondents answered all 15. The attributes of the vehicles were varied over experiments, both for a given respondent and over respondents. The attributes are: price, operating cost in dollars per month, engine type (gas, electric, or hybrid), range if electric (in hundreds of miles between recharging), and the performance level of the vehicle (high, medium, or low). The performance level was described in terms of top speed and acceleration, and these descriptions did not vary for each level; for example, "High" performance was described as having a top speed of 100 mpg and 12 seconds to reach 60 mpg, and this description was the same for all "high" performance vehicles. 
# 
# A detailed description of the data is provided by Kenneth Train at https://eml.berkeley.edu/~train/ec244ps.html

# In[2]:


column_names = ["IndID","ObsID", "Chosen", "Price", "OperCost", "Range", "EV", "Gas", "Hybrid", "HighPerf", "MedHighPerf"]
df = pd.read_csv("data/toyota.txt", delimiter=" ", names=column_names)

df["Price"] = df["Price"]/10000     # scale price to be in tens of thousands of dollars.
df["OperCost"] = df["OperCost"]/10  # scale operating cost to be in tens of dollars.

# fix dataframe to match expected format
altID = []
menuID = []
curr_n = -1
curr_o = -1
curr_a = -1
curr_t = -1
for n,o in df[["IndID", "ObsID"]].values:
    if n != curr_n:
        curr_n += 1
        curr_t = 0
    if o != curr_o:
        curr_t += 1
        curr_a = 0
    
    curr_a += 1
    curr_n = n
    curr_o = o
    
    altID.append(curr_a)
    menuID.append(curr_t)
    #print(n,o,curr_t,curr_a)
    
df["AltID"] = altID
df["MenuID"] = menuID

df.head()


# At the moment, the provided interface only supports data in the so-called "wide format", so we need to convert it first:

# In[3]:


# convert to wide format
data_wide = []
for ix in range(0,len(df),3):
    new_row = df.loc[ix][["IndID","ObsID","MenuID"]].values.tolist()
    new_row += df.loc[ix][["Price","OperCost","Range","EV","Hybrid","HighPerf","MedHighPerf"]].values.tolist()
    new_row += df.loc[ix+1][["Price","OperCost","Range","EV","Hybrid","HighPerf","MedHighPerf"]].values.tolist()
    new_row += df.loc[ix+2][["Price","OperCost","Range","EV","Hybrid","HighPerf","MedHighPerf"]].values.tolist()
    choice = np.argmax([df.loc[ix]["Chosen"], df.loc[ix+1]["Chosen"], df.loc[ix+2]["Chosen"]])
    new_row += [choice]
    #print(new_row)
    data_wide.append(new_row)
    
column_names = ["IndID","ObsID","MenuID",
                "Price1","OperCost1","Range1","EV1","Hybrid1","HighPerf1","MedHighPerf1",
                "Price2","OperCost2","Range2","EV2","Hybrid2","HighPerf2","MedHighPerf2",
                "Price3","OperCost3","Range3","EV3","Hybrid3","HighPerf3","MedHighPerf3",
                "Chosen"]
df_wide = pd.DataFrame(data_wide, columns=column_names)
df_wide['ones'] = np.ones(len(data_wide)).astype(int)
df_wide.head()


# ## Mixed Logit specification
# 
# We now make use of the developed formula interface to specify the utilities of the mixed logit model. 
# 
# We begin by defining the fixed effects parameters, the random effects parameters, and the observed variables. This creates instances of Python objects that can be put together to define the utility functions for the different alternatives.
# 
# Once the utilities are defined, we collect them in a Python dictionary mapping alternative names to their corresponding expressions.

# In[4]:


from core.dcm_interface import FixedEffect, RandomEffect, ObservedVariable

# define fixed effects parameters
B_PRICE = FixedEffect('B_PRICE')

# define random effects parameters
B_OperCost = RandomEffect('B_OperCost')
B_Range = RandomEffect('B_Range')
B_EV = RandomEffect('B_EV')
B_Hybrid = RandomEffect('B_Hybrid')
B_HighPerf = RandomEffect('B_HighPerf')
B_MedHighPerf = RandomEffect('B_MedHighPerf')

# define observed variables
for attr in df_wide.columns:
    exec("%s = ObservedVariable('%s')" % (attr,attr))

# define utility functions
V1 = B_PRICE*Price1 + B_OperCost*OperCost1 + B_Range*Range1 + B_EV*EV1 + B_Hybrid*Hybrid1 + B_HighPerf*HighPerf1 + B_MedHighPerf*MedHighPerf1
V2 = B_PRICE*Price2 + B_OperCost*OperCost2 + B_Range*Range2 + B_EV*EV2 + B_Hybrid*Hybrid2 + B_HighPerf*HighPerf2 + B_MedHighPerf*MedHighPerf2
V3 = B_PRICE*Price3 + B_OperCost*OperCost3 + B_Range*Range3 + B_EV*EV3 + B_Hybrid*Hybrid3 + B_HighPerf*HighPerf3 + B_MedHighPerf*MedHighPerf3

# associate utility functions with the names of the alternatives
utilities = {"ALT1": V1, "ALT2": V2, "ALT3": V3}


# We are now ready to create a Specification object containing the utilities that we have just defined. Note that we must also specify the type of choice model to be used - a mixed logit model (MXL) in this case.
# 
# Note that we can inspect the specification by printing the dcm_spec object.

# In[5]:


from core.dcm_interface import Specification

# create MXL specification object based on the utilities previously defined
dcm_spec = Specification('MXL', utilities)
print(dcm_spec)


# Once the Specification is defined, we need to define the DCM Dataset object that goes along with it. For this, we instantiate the Dataset class with the Pandas dataframe containing the data in the so-called "wide format", the name of column in the dataframe containing the observed choices and the dcm_spec that we have previously created.
# 
# Note that since this is panel data, we must also specify the name of the column in the dataframe that contains the ID of the respondent (this should be a integer ranging from 0 the num_resp-1).

# In[7]:


from core.dcm_interface import Dataset

# create DCM dataset object
dcm_dataset = Dataset(df_wide, 'Chosen', dcm_spec, resp_id_col='IndID')


# As with the specification, we can inspect the DCM dataset by printing the dcm_dataset object:

# In[8]:


print(dcm_dataset)


# ## Bayesian Mixed Logit Model in PyTorch
# 
# It is now time to perform approximate Bayesian inference on the mixed logit model that we have specified. The generative process of the MXL model that we will be using is the following:
# 
# 1. Draw fixed taste parameters $\boldsymbol\alpha \sim \mathcal{N}(\boldsymbol\lambda_0, \boldsymbol\Xi_0)$
# 2. Draw mean vector $\boldsymbol\zeta \sim \mathcal{N}(\boldsymbol\mu_0, \boldsymbol\Sigma_0)$
# 3. Draw scales vector $\boldsymbol\theta \sim \mbox{half-Cauchy}(\boldsymbol\sigma_0)$
# 4. Draw correlation matrix $\boldsymbol\Psi \sim \mbox{LKJ}(\nu)$
# 5. For each decision-maker $n \in \{1,\dots,N\}$
#     1. Draw random taste parameters $\boldsymbol\beta_n \sim \mathcal{N}(\boldsymbol\zeta,\boldsymbol\Omega)$
#     2. For each choice occasion $t \in \{1,\dots,T_n\}$
#         1. Draw observed choice $y_{nt} \sim \mbox{MNL}(\boldsymbol\alpha, \boldsymbol\beta_n, \textbf{X}_{nt})$
#         
# where $\boldsymbol\Omega = \mbox{diag}(\boldsymbol\theta) \times \boldsymbol\Psi \times  \mbox{diag}(\boldsymbol\theta)$.
# 
# We can instantiate this model from the TorchMXL using the following code. We can the run variational inference to approximate the posterior distribution of the latent variables in the model. 
# 
# Note that we are providing the "infer" method with the parameter estimates obtained by Biogeme in order to track the progress of VI and for comparison and debugging purposes. I.e., the Alpha RMSE and Beta RMSE values outputted correspond to the RMSE with respect to the results of Biogeme. 

# In[9]:


get_ipython().run_cell_magic('time', '', "\nfrom core.torch_mxl import TorchMXL\n\n# instantiate MXL model\nmxl = TorchMXL(dcm_dataset, batch_size=dcm_dataset.num_resp, use_inference_net=False, use_cuda=True)\n\n# we are using Biogeme's results as a reference results for comparison\nbiogeme_alpha = np.array([-0.5080])\nbiogeme_beta = np.array([-0.1355, 0.4759, -1.5995, 0.5256, 0.1116, 0.5333])\n\n# run Bayesian inference (variational inference)\nresults = mxl.infer(num_epochs=10000, true_alpha=biogeme_alpha, true_beta=biogeme_beta)")


# The "results" dictionary containts a summary of the results of variational inference, including means of the posterior approximations for the different parameters in the model:

# In[10]:


results

