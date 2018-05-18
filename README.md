# tfdiv

A library for factorization machines in [TensorFlow](https://www.tensorflow.org/).  
The library provides standard ***Classifier*** and ***Regression*** modules,
that can be extended by defining a custom `loss_function`. 
It also provides a **Ranking** module for both classifiers and regressions.
We also provide the **Bayesian Personalized Ranking** **[2]**,
which is a pairwise learning-to-rank algorithm.

### What are Factorization Machines?

Factorization Machines (FMs) are a new model class devised by S. Rendle **[1]**, 
Similarly to Support Vector Machines (SVM), 
they are a general predictor working with any 
real-valued feature vector. However, FMs model all
interactions between variables using factorized parameters. 
Thus, they estimate high-order interactions even in 
problems with huge sparsity like _Recommender Systems_.

Factorized parameters estimation is a shared feature 
with many other factorization models like Matrix Factorization.
In contrast to those, FMs can handle the general prediction tasks 
whereas other factorization models work with specific input data. 

![dataset](./images/real-valued-feature-vectors.jpg "Real-Valued Feature Vectors")


<!---

#### FMs' Model Equation
 A linear model, given a vector `x` models its output `y` as

<p>
<a href="url"><img src="https://raw.githubusercontent.com/jmhessel/fmpytorch/master/images/linear_model.png" width="250" align="center"></a>
</p>

where `w` are the learnable weights of the model.

However, the interactions between the input variables `x_i` are purely additive. In some cases, it might be useful to model the interactions between your variables, e.g., `x_i * x_j`. You could add terms into your model like


<p>
<a href="url"><img src="https://raw.githubusercontent.com/jmhessel/fmpytorch/master/images/second_order.png" width="400" align="center"></a>
</p>

However, this introduces a large number of `w2` variables. Specifically, there are `O(n^2)` parameters introduced in this formulation, one for each interaction pair. A factorization machine approximates `w2` using low dimensional factors, i.e.,
<p>
<a href="url"><img src="https://raw.githubusercontent.com/jmhessel/fmpytorch/master/images/fm.png" width="400" align="center"></a>
</p>

where each `v_i` is a low-dimensional vector. This is the forward pass of a second order factorization machine. This low-rank re-formulation has reduced the number of additional parameters for the factorization machine to `O(k*n)`. Magically, the forward (and backward) pass can be reformulated so that it can be computed in `O(k*n)`, rather than the naive `O(k*n^2)` formulation above.


## Usage

The factorization machine layers in can be used just like any other built-in module. Here's a simple feed-forward model using a factorization machine that takes in a 50-D input, and models interactions using `k=5` factors.
See demo for fuller examples.
--->

## Installation

This package requires ```scikit-learn```, ```numpy```, ```scipy```, ```tensorflow```.

To install, you can run:

```
cd tfdiv
python setup.py install
```

<!---

## Currently supported features

Currently, only a second order factorization machine is supported. The
forward and backward passes are implemented in cython. Compared to the
autodiff solution, the cython passes run several orders of magnitude
faster. I've only tested it with python 2 at the moment.

## TODOs

0. Support for sparse tensors.
1. More interesting useage examples
2. More testing, e.g., with python 3, etc.
3. Make sure all of the code plays nice with torch-specific stuff, e.g., GPUs
4. Arbitrary order factorization machine support
5. Better organization/code cleaning
--->

## References 

1. Rendle, Steffen. "Factorization machines." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
2. Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.