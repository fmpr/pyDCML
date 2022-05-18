Introduction to PyDCML
===========================

PyDCML is a `Python <https://www.python.org/>`_ library for **fast implementation and scalable inference of Bayesian Discrete Choice Models** that makes it easy to leverage flexible state-of-the-art modelling techniques from Machine Learning, while remaining interpretable and preserving the links with economic theories established by Daniel McFadden [Ref1]_.  

PyDCML uses `PyTorch <https://pytorch.org/>`_ on the backend in order to enable stochastic backpropagation, automatic differentiation and GPU-accelerated computation. In doing so, **PyDCML aims at enabling flexible and expressive Choice Modeling, unifying the best of modern Machine Learning and Bayesian modeling with Discrete Choice Theory**.

PyDCML provides a simple formula interface that allows users to define observed and latent variables (parameters), and use them to easily specify utility functions::

    V1 = BETA_COST*ALT1_COST + BETA_DUR*ALT1_DURATION + ...

Read the `documentation <https://fmpr.github.io/pyDCML/intro.html>`_ for additonal details, demos, model extensions, etc.

######################## 
Supported models
######################## 

Besides core implementations of Multinomial Logit (MNL) and Mixed Logit (MXL) models, PyDCML currently provides implementations of:

* Mixed Logit models with neural networks in the utilities (originally proposed in [Ref2]_ for MNL models)
* Mixed Logit models with Automatic Relevance Determination (originally proposed in [Ref3]_ for MNL models)
* Mixed Logit models with ordered responses

All these model extensions leverage PyDCML's flexibility and modular design. See :ref:`understanding` for a detailed explanation of how PyDCML is implemented, and check out :ref:`extending` for a tutorial on how to extend and implement new models in PyDCML. 

######################## 
Bayesian inference
######################## 

Inference of Discrete Choice Models in PyDCML is done using Stochastic Variational Inference (SVI). Thanks to the efficient PyTorch implementation that can leverage modern GPUs with `Cuda <https://developer.nvidia.com/cuda-toolkit/>`_ support, PyDCML is able to scale inference to large datasets. Additionally, PyDCML can make use of Neural Networks to amortize the cost of Variational Inference (Amortized VI), as proposed in [Ref4]_, thereby achieving **computational speedups of orders of magnitude when compared with traditional estimation methods** for Mixed Logit Models. 

.. figure:: docs/images/scalability2.png
    :width: 500px
    :align: center
    :alt: Scalability of Amortized VI for Mixed Logit Models
    :figclass: align-center

    Scalability of Amortized VI for Mixed Logit Models when compared with traditional estimation methods such as Maximum Simulated Likelihood Estimation (MSLE) and Gibbs sampling. See :ref:`demos`.
    
######################## 
References
######################## 

.. [Ref1] McFadden, D. Conditional logit analysis of qualitative choice behavior, 1973.

.. [Ref2] Sifringer, B., Lurkin, V. and Alahi, A. Enhancing discrete choice models with representation learning. Transportation Research Part B: Methodological, 2020.

.. [Ref3] Rodrigues, F., Ortelli, N., Bierlaire, M. and Pereira, F.C. Bayesian automatic relevance determination for utility function specification in discrete choice models. IEEE Transactions on Intelligent Transportation Systems, 2020.

.. [Ref4] Rodrigues, F. Scaling Bayesian inference of mixed multinomial logit models to large datasets. In Transportation Research Part B: Methodological, 2022.

