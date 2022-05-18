#!/usr/bin/env python
# coding: utf-8

# # Mixed logit with simulated data
# 
# This is a demo of stochastic variational inference (SVI) using simulated data for 500 individuals

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(0, "/home/rodr/code/amortized-mxl-dev/release") 

import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fix random seed for reproducibility
np.random.seed(42)


# ## Generate simulated data
# 
# We predefine the fixed effects parameters (true_alpha) and random effects parameters (true_beta), as well as the covariance matrix (true_Omega), and sample simulated choice data for 500 respondents (num_resp), each with 5 choice situations (num_menus). The number of choice alternatives is set to 5. 

# In[2]:


from core.dcm_fakedata import generate_fake_data_wide

num_resp = 500
num_menus = 5
num_alternatives = 5

true_alpha = np.array([-0.8, 0.8, 1.2])
true_beta = np.array([-0.8, 0.8, 1.0, -0.8, 1.5])
# dynamic version of generating Omega
corr = 0.8
scale_factor = 1.0
true_Omega = corr*np.ones((len(true_beta),len(true_beta))) # off-diagonal values of cov matrix
true_Omega[np.arange(len(true_beta)), np.arange(len(true_beta))] = 1.0 # diagonal values of cov matrix
true_Omega *= scale_factor

df = generate_fake_data_wide(num_resp, num_menus, num_alternatives, true_alpha, true_beta, true_Omega)
df.head()


# ## Mixed Logit specification
# 
# We now make use of the developed formula interface to specify the utilities of the mixed logit model. 
# 
# We begin by defining the fixed effects parameters, the random effects parameters, and the observed variables. This creates instances of Python objects that can be put together to define the utility functions for the different alternatives.
# 
# Once the utilities are defined, we collect them in a Python dictionary mapping alternative names to their corresponding expressions.

# In[3]:


from core.dcm_interface import FixedEffect, RandomEffect, ObservedVariable

# define fixed effects parameters
B_XF1 = FixedEffect('BETA_XF1')
B_XF2 = FixedEffect('BETA_XF2')
B_XF3 = FixedEffect('BETA_XF3')

# define random effects parameters
B_XR1 = RandomEffect('BETA_XR1')
B_XR2 = RandomEffect('BETA_XR2')
B_XR3 = RandomEffect('BETA_XR3')
B_XR4 = RandomEffect('BETA_XR4')
B_XR5 = RandomEffect('BETA_XR5')

# define observed variables
for attr in df.columns:
    exec("%s = ObservedVariable('%s')" % (attr,attr))

# define utility functions
V1 = B_XF1*ALT1_XF1 + B_XF2*ALT1_XF2 + B_XF3*ALT1_XF3 + B_XR1*ALT1_XR1 + B_XR2*ALT1_XR2 + B_XR3*ALT1_XR3 + B_XR4*ALT1_XR4 + B_XR5*ALT1_XR5
V2 = B_XF1*ALT2_XF1 + B_XF2*ALT2_XF2 + B_XF3*ALT2_XF3 + B_XR1*ALT2_XR1 + B_XR2*ALT2_XR2 + B_XR3*ALT2_XR3 + B_XR4*ALT2_XR4 + B_XR5*ALT2_XR5
V3 = B_XF1*ALT3_XF1 + B_XF2*ALT3_XF2 + B_XF3*ALT3_XF3 + B_XR1*ALT3_XR1 + B_XR2*ALT3_XR2 + B_XR3*ALT3_XR3 + B_XR4*ALT3_XR4 + B_XR5*ALT3_XR5
V4 = B_XF1*ALT4_XF1 + B_XF2*ALT4_XF2 + B_XF3*ALT4_XF3 + B_XR1*ALT4_XR1 + B_XR2*ALT4_XR2 + B_XR3*ALT4_XR3 + B_XR4*ALT4_XR4 + B_XR5*ALT4_XR5
V5 = B_XF1*ALT5_XF1 + B_XF2*ALT5_XF2 + B_XF3*ALT5_XF3 + B_XR1*ALT5_XR1 + B_XR2*ALT5_XR2 + B_XR3*ALT5_XR3 + B_XR4*ALT5_XR4 + B_XR5*ALT5_XR5

# associate utility functions with the names of the alternatives
utilities = {"ALT1": V1, "ALT2": V2, "ALT3": V3, "ALT4": V4, "ALT5": V5}


# We are now ready to create a Specification object containing the utilities that we have just defined. Note that we must also specify the type of choice model to be used - a mixed logit model (MXL) in this case.
# 
# Note that we can inspect the specification by printing the dcm_spec object.

# In[4]:


from core.dcm_interface import Specification

# create MXL specification object based on the utilities previously defined
dcm_spec = Specification('MXL', utilities)
print(dcm_spec)


# Once the Specification is defined, we need to define the DCM Dataset object that goes along with it. For this, we instantiate the Dataset class with the Pandas dataframe containing the data in the so-called "wide format", the name of column in the dataframe containing the observed choices and the dcm_spec that we have previously created.
# 
# Note that since this is panel data, we must also specify the name of the column in the dataframe that contains the ID of the respondent (this should be a integer ranging from 0 the num_resp-1).

# In[5]:


from core.dcm_interface import Dataset

# create DCM dataset object
dcm_dataset = Dataset(df, 'choice', dcm_spec, resp_id_col='indID')


# As with the specification, we can inspect the DCM dataset by printing the dcm_dataset object:

# In[6]:


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
# We can instantiate this model from the TorchMXL using the following code. We can the run variational inference to approximate the posterior distribution of the latent variables in the model. Note that since in this case we know the true parameters that were used to generate the simualated choice data, we can pass them to the "infer" method in order to obtain additional information during the ELBO maximization (useful for tracking the progress of VI and for other debugging purposes). 

# In[7]:


get_ipython().run_cell_magic('time', '', '\nfrom core.torch_mxl import TorchMXL\n\n# instantiate MXL model\nmxl = TorchMXL(dcm_dataset, batch_size=num_resp, use_inference_net=False, use_cuda=True)\n\n# run Bayesian inference (variational inference)\nresults = mxl.infer(num_epochs=5000, true_alpha=true_alpha, true_beta=true_beta)')


# The "results" dictionary containts a summary of the results of variational inference, including means of the posterior approximations for the different parameters in the model:

# In[8]:


results


# This interface is currently being improved to include additional output information, but additional information can be obtained from the attributes of the "mxl" object for now. 
