.. _extensions:

----------------------------------
Extensions
----------------------------------

This section describes several extensions of the core Mixed Logit that are already provided by PyDCML.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Neural networks in utility functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension adds support for the use of Neural Networks when specifying utility functions for Mixed Logit models. Neural Networks are powerful Machine Learning models that can be used as extremely flexible function approximators. The goal is then to allow for capturing complex non-linear interactions between different alternative attributes in the utility functions. This approach was originally proposed in [1]_ for simple Multinomial Logit models. 

.. [1] Sifringer, B., Lurkin, V. and Alahi, A. Enhancing discrete choice models with representation learning. Transportation Research Part B: Methodological, 2020.

The formula interface of PyDCML was also extended to simplify the use of Neural Networks as follows::

    V1 = BETA_COST*ALT1_COST + BETA_DUR*ALT1_DURATION + NNET(ALT1_DISCOUNT, INCOME, ...) + ...
    
Check out the corresponding subsection for details.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Mixed logit with Automatic Relevance Determination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension adds support for Automatic Relevance Determination (ARD). ARD is a popular Bayesian method in Machine Learning for automatic feature selection. In the context of choice models, ARD can therefore be used to assist choice modellers in specifying utility functions. The choice modeller just needs to over-specify the utilties and the model will "prune out" irrelevant terms through shrinkage. This approach was originally proposed in [2]_ for simple Multinomial Logit models. The implementation privded by PyDCML is a generalization to Mixed Logit models.

.. [2] Rodrigues, F., Ortelli, N., Bierlaire, M. and Pereira, F.C. Bayesian automatic relevance determination for utility function specification in discrete choice models. IEEE Transactions on Intelligent Transportation Systems, 2020.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Mixed ordered logit models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an extension of the core Mixed Logit model to ordinal response variables (choices), which can be useful in situation where there is a natural ordering of the choice set. 
