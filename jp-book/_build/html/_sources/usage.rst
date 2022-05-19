.. _usage:

Basic usage
===========================

Specifying choice models using the formula interface provided by PyDCML is very simple. We start by defining the fixed effects parameters and the random effects parameters using the ``FixedEffect`` and ``RandomEffect`` classes::

    # define fixed effects parameters
    B_XF1 = FixedEffect('BETA_XF1')
    B_XF2 = FixedEffect('BETA_XF2')
    B_XF3 = FixedEffect('BETA_XF3')

    # define random effects parameters
    B_XR1 = RandomEffect('BETA_XR1')
    B_XR2 = RandomEffect('BETA_XR2')

The only input that ``FixedEffect`` and ``RandomEffect`` take is the names of the parameters to be used, for example, for displaying the estimation results. 

We then need to define the observed variables. This can be done as follows::

    XF1 = ObservedVariable('XF1')
    XF2 = ObservedVariable('XF2')
    XF3 = ObservedVariable('XF3')
    
    XR1 = ObservedVariable('XR1')
    XR2 = ObservedVariable('XR2')
    
Note that the names provided as input to ``ObservedVariable()`` must correspond to the names of columns in the Pandas dataframe containing the data. 

Alternatively, we can compactly convert each column of a Pandas dataframe ``df`` into an ``ObservedVariable`` by using the following code::

    # define observed variables
    for attr in df.columns:
        exec("%s = ObservedVariable('%s')" % (attr,attr))
        
The blocks of code above create instances of Python objects that can be put together to define the utility functions for the different alternatives. For example, we can define the following utility functions corresponding to 3 different choice alternatives::

    V1 = B_XF1*ALT1_XF1 + B_XF2*ALT1_XF2 + B_XF3*ALT1_XF3 + B_XR1*ALT1_XR1 + B_XR2*ALT1_XR2
    V2 = B_XF1*ALT2_XF1 + B_XF2*ALT2_XF2 + B_XF3*ALT2_XF3 + B_XR1*ALT2_XR1 + B_XR2*ALT2_XR2
    V3 = B_XF1*ALT3_XF1 + B_XF2*ALT3_XF2 + B_XF3*ALT3_XF3 + B_XR1*ALT3_XR1 + B_XR2*ALT3_XR2

Once the expressions for the different choice alternatives are defined, we need to associate them with the correspoding alternative names. This achieved using a Python dictionary::

    utilities = {"ALT1": V1, "ALT2": V2, "ALT3": V3}
    
We are now ready to create a ``Specification`` object containing the utilities that we have just defined. Note that we must also specify the type of choice model to be used - a mixed logit model (MXL) in this case.::

    dcm_spec = Specification('MXL', utilities)

Note that we can inspect the specification by printing the ``dcm_spec`` object::

    print(dcm_spec)
    
Once the Specification is defined, we need to define the ``Dataset`` object that goes along with it. For this, we instantiate the ``Dataset`` class with the Pandas dataframe containing the data in the so-called "wide format". In the case of panel data, we must also specify the name of the column in the dataframe that contains the ID of the respondent (this should be a integer ranging from 0 the num_resp-1)::

    dcm_dataset = Dataset(df, 'choice', dcm_spec, resp_id_col='indID')
    
As with the specification, we can inspect the DCM dataset by printing the ``dcm_dataset`` object::

    print(dcm_dataset)

We are now ready to instante the choice model that we want to use. For example, we can use a Mixed Logit model as implemented by the class ``TorchMXL``. We do this by instatiating the class with the ``dcm_dataset`` object that we have previously created::

    mxl = TorchMXL(dcm_dataset, batch_size=num_resp, use_cuda=True, use_inference_net=False)

Note that we additionally provide ``TorchMXL`` with some extra information: ``batch_size`` indicates the size of the random subset of respondents whose data should be considered for computing gradients at each iteration of variational inference, ``use_cuda`` indicates whether or not to use GPU-accelaration to speed-up inference, and ``use_inference_net`` indicates whether or not to use an inference neural network to amortize the cost of variational inference as proposed in [Ref1]_.

We can now run variational inference on the model using::

    results = mxl.infer(num_epochs=5000)
    
Once inference is completed, PyDCML will output a brief summary of the results such as the means of the posterior approximations for the different parameters in the model. This information and additional one is also stored in the ``results`` dictionary outputted by the ``infer()`` method. 

.. [Ref1] Rodrigues, F. Scaling Bayesian inference of mixed multinomial logit models to large datasets. In Transportation Research Part B: Methodological, 2022.
