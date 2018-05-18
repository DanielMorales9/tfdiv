# tfdiv

A library for factorization machines in [TensorFlow](https://www.tensorflow.org/).  
The library provides standard *Classifier* and *Regression* modules,
that can be extended by defining a custom `loss_function`.  
It also provides a *Ranking* module for several classifiers.
We also provide the *Bayesian Personalized Ranking* **[2]**,
which is a pairwise learning-to-rank algorithm.

### What are Factorization Machines?

Factorization Machines (FMs) are a new model class devised by S. Rendle **[1]**.    
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


#### FMs' Model Equation
In a linear model, given a vector `x` models its predicted output `y` is as follows:

![linear](http://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%28x%29%20%3A%3D%20w_0%20&plus;%20%5Csum_%7Bi%20%3D%201%7D%5En%20w_i%20x_i)  

where `w` are the estimated weights of the model.  
Here, the interactions between the input variables `x_i` 
are purely additive, whereas it might be useful to 
model the interactions between your variables, e.g., `x_i * x_j`.
Thus, such model class has additional parameters to estimate 
the interactions between variables, 
i.e. `V` whose dimension depends on th order of interactions.   
Therefore, the equation for a model that captures the pairwise interaction between variables looks like as follows.  

![equation](http://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%28%5Cmathbf%7Bx%7D%29%20%3A%3D%20w_0%20&plus;%20%5Csum_%7Bj%20%3D%201%7D%5En%20w_j%20x_j%20&plus;%20%5Csum_%7Bi%20%3D%201%7D%5En%20%5Csum_%7Bj%20%3D%20i&plus;1%7D%5En%20v_%7Bij%7D%7E%20x_i%20x_j)

However, in this formulation the number of parameters 
grows exponentially with the number of features in the feature vector, 
e.g. in the second order interaction model there are `O(n^2)` parameters introduced. 

Rendle mathematically demonstrated that factorization machine 
can reduce the number of parameters to estimate by factorizing them.
Thus, he reduced both memory and time complexity to `O(k*n)`, 
i.e. linear complexity rather than polynomial.  

Which translates to the following 2-way model equation:   
  
![equation](http://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%28%5Cmathbf%7Bx%7D%29%20%3A%3D%20w_0%20&plus;%20%5Csum_%7Bj%20%3D%201%7D%5En%20w_j%20x_j%20&plus;%20%5Csum_%7Bi%20%3D%201%7D%5En%20%5Csum_%7Bj%20%3D%20i&plus;1%7D%5En%20%5Cleft%20%5Clangle%20v_i%2C%20v_j%20%5Cright%20%5Crangle%20x_i%20x_j)


<!---
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