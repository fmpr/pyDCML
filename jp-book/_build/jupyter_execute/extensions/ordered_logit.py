#!/usr/bin/env python
# coding: utf-8

# # Mixed ordered logit
# 
# We begin by performing the necessary imports:

# In[10]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(0, "/home/rodr/code/amortized-mxl-dev/release") 

import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Fix random seed for reproducibility
np.random.seed(42)


# ## Import previously simulated data
# 
# For this demo, we will use previously generated simulated data. Since this is artificial data, we actually know the parameter values that were used to generate the choices: [1, -1, 1, -1]. The (log) cutoff points used to generate the data were: [-0.7 -0.2  0.5]. Our goal is to see whether our model implementation can accurately recover the true parameters that were used to generate the data. 

# In[2]:


df = pd.read_csv('../data/fake_data_ordered.csv', index_col=0)
num_resp = len(df)
df['indID'] = np.arange(num_resp)
df.head()


# ## Mixed Logit specification
# 
# We now make use of the developed formula interface to specify the utilities of the mixed logit model. 
# 
# We begin by defining the fixed effects parameters, the random effects parameters, and the observed variables. This creates instances of Python objects that can be put together to define the utility functions for the different alternatives.
# 
# Once the utilities are defined, we collect them in a Python dictionary mapping alternative names to their corresponding expressions.
# 
# Note that, since this is an ordinal regression model, there is only one single utility. The observed values correspond to different "levels" of the response variable. 

# In[3]:


from core.dcm_interface import FixedEffect, RandomEffect, ObservedVariable
import torch.distributions as dists

# define fixed effects parameters
B_X0 = FixedEffect('BETA_X0')
B_X1 = FixedEffect('BETA_X1')

# define random effects parameters
B_X2 = RandomEffect('BETA_X2')
B_X3 = RandomEffect('BETA_X3')

# define observed variables
for attr in df.columns:
    exec("%s = ObservedVariable('%s')" % (attr,attr))

# define utility functions
V1 = B_X0*x0 + B_X1*x1 + B_X2*x2 + B_X3*x3

# associate utility functions with the names of the alternatives
utilities = {"ALT1": V1}


# We are now ready to create a Specification object containing the utilities that we have just defined. Note that we must also specify the type of choice model to be used - a mixed logit model (MXL) in this case.
# 
# Note that we can inspect the specification by printing the dcm_spec object.

# In[4]:


from core.dcm_interface import Specification

#Logit(choice, utilities, availability, df)
#Logit(choice_test, utilities, availability_test, df_test)

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


# ## Bayesian Mixed Ordered Logit Model in PyTorch
# 
# We will modify the generative process of the core MXL model by introducing a vector of $C-2$ cutoff points $\boldsymbol\kappa$, where $C$ denotes the number of levels in the response variable. The cutoffs will then be used by the ``OrderedLogit`` function to compute the probabilities of each level. 
# 
# The resulting generative process is the following (changes to the core MXL model are highlighted in red):
# 
# 1. <font color='red'> Draw cutoff parameters $\boldsymbol\kappa \sim \mathcal{N}(\boldsymbol\eta_0, \boldsymbol\Phi_0)$ </font>
# 2. Draw fixed taste parameters $\boldsymbol\alpha \sim \mathcal{N}(\boldsymbol\lambda_0, \boldsymbol\Xi_0)$
# 3. Draw mean vector $\boldsymbol\zeta \sim \mathcal{N}(\boldsymbol\mu_0, \boldsymbol\Sigma_0)$
# 4. Draw scales vector $\boldsymbol\theta \sim \mbox{half-Cauchy}(\boldsymbol\sigma_0)$
# 5. Draw correlation matrix $\boldsymbol\Psi \sim \mbox{LKJ}(\nu)$
# 6. For each decision-maker $n \in \{1,\dots,N\}$
#     1. Draw random taste parameters $\boldsymbol\beta_n \sim \mathcal{N}(\boldsymbol\zeta,\boldsymbol\Omega)$
#     2. For each choice occasion $t \in \{1,\dots,T_n\}$
#         1. <font color='red'> Draw observed ordered response $y_{nt} \sim \mbox{OrderedLogit}(\boldsymbol\alpha, \boldsymbol\beta_n, \boldsymbol\kappa, \textbf{X}_{nt})$ </font>
#         
# where $\boldsymbol\Omega = \mbox{diag}(\boldsymbol\theta) \times \boldsymbol\Psi \times  \mbox{diag}(\boldsymbol\theta)$.
# 
# For the ``OrderedLogit`` probability, we assume:
# 
# $$
# \begin{align}
# p(y_{nt} = j) &= p(\kappa_{i-1} < \boldsymbol\alpha^T \textbf{x}_{ntj,F} + \boldsymbol\beta_n^T \textbf{x}_{ntj,R} < \kappa_i) \\
# &= \frac{1}{1+\exp(-\kappa_i + \boldsymbol\alpha^T \textbf{x}_{ntj,F} + \boldsymbol\beta_n^T \textbf{x}_{ntj,R})} - \frac{1}{1+\exp(-\kappa_{i-1} + \boldsymbol\alpha^T \textbf{x}_{ntj,F} + \boldsymbol\beta_n^T \textbf{x}_{ntj,R})}
# \end{align}
# $$
# 
# where $\kappa_0$ is defined as $-\infty$ and $\kappa_C$ as $+\infty$ (hence, the number of $\kappa_i$ variables that we need to perform inference on is only $C-2$). Note that we use $\textbf{x}_{ntj,F}$ and $\textbf{x}_{ntj,R}$ to distinguish between fixed effects and random effects, respectively.
# 
# This model is already implemented in the class ``TorchMXL_Ordered``. At the end of this notebook, we provide an explanation of how this extension was implemented. 
# 
# We can instantiate this model from the ``TorchMXL_Ordered`` using the following code. We can the run variational inference to approximate the posterior distribution of the latent variables in the model. Note that since in this case we know the true parameters that were used to generate the simualated choice data, we can pass them to the "infer" method in order to obtain additional information during the ELBO maximization (useful for tracking the progress of VI and for other debugging purposes). 

# In[7]:


get_ipython().run_cell_magic('time', '', '\nfrom core.torch_mxl_ordered import TorchMXL_Ordered\n\n# instantiate MXL model\nnum_categories = 5\nmxl = TorchMXL_Ordered(dcm_dataset, num_categories, batch_size=num_resp, use_inference_net=False, use_cuda=True)\n\n# run Bayesian inference (variational inference)\nresults = mxl.infer(num_epochs=7000, true_alpha=np.array([1, -1]), true_beta=np.array([1, -1]))')


# In[30]:


# log(true_cutoffs)=[-0.7 -0.2  0.5]


# Lets now compare the inferred cutoffs with the true cutoffs:
# log(true_cutoffs) = [-0.7 -0.2  0.5]
# 
# The ones that we infered are:

# In[11]:


torch.log(mxl.softplus(mxl.kappa_mu))


# Quite similar values!

# The "results" dictionary containts a summary of the results of variational inference, including means of the posterior approximations for the different parameters in the model:

# In[33]:


results


# This interface is currently being improved to include additional output information, but additional information can be obtained from the attributes of the "mxl" object for now. 

# ## Implementation details

# TODO
