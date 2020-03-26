# SUPPORT VECTOR MACHINES

Support Vector Machines have been around for a long time now. First developed in the framework of statistical learning theory, and have been successfully used in ML thereafter. 

So, here I have:

  - Some theory behind the algorithm
  - An easy to follow example
  - Useful references if you want to read about them further

# Brief Theory

What we do in statistical learning theory:
The problem in SLT consists in finding a function f that minimizes the expectation of the error on new data, that is,
find a function f that minimizes the expected error:

![formula](https://render.githubusercontent.com/render/math?math=\int%20V(y,f(x))P(x,y)dxdy)

Since $P(x,y)$ in unknown, we need to use some induction principle in order to infer from the $l$ available training
examples a function that minimizes the expected error. The principle used is Empirical Risk Minimization (ERM)
over a set of possible functions, called hypothesis space. Formally this can be written as minimizing the empirical
error:

![formula](https://render.githubusercontent.com/render/math?math=\frac{1}{l}%20\sum%20V(y_{i},%20f(x_{i})))

with f being restricted to be in a space of functions - hypothesis space - say H.

The simplest formulation of SVM is the linear one,
where the hyperplane lies on the space of the input data x. In this case the hypothesis space is a subset of all
hyperplanes of the form:

![formula](https://render.githubusercontent.com/render/math?math=f(x)=w\cdot%20x+b)

SVM finds a hyperplane in a feature space induced by a kernel K (the kernel defines a dot product in that space.
Through the kernel K the hypothesis space is defined as a set of "hyperplanes" in the feature space induced by K.
This can also be seen as a set of functions in a Reproducing Kernel Hilbert Space (RKHS) defined by K. 
So to summarize, the hypothesis space used by SVM is a subset of the set of hyperplanes defined in some space -
an RKHS. This space can be formally written as

![formula](https://render.githubusercontent.com/render/math?math=\left%20\|%20f%20\right%20\|_{K}^{2}%20\leq%20A^{2})


where K is the kernel that defines the RKHS, and f K is the RKHS norm of the function. for some constant A

For binary SVM margin criterion is defined as the projections of the data on the hyperplane are such that the between-class variance of the projections is maximized, while the within-class variance is minimized

On a real dataset:

SVM model finds a hyperplane by using the optimal values w* (weights/normal) and b* (intercept) which define this hyperplane. The optimal values are found by minimizing a cost function. The SVM model with the optimal values is then defined like this:

![formula](https://render.githubusercontent.com/render/math?math=f(x)=sign(w^{*}\cdot%20x+b^{*}))

The cost function of an SVM looks like this:

![formula](https://render.githubusercontent.com/render/math?math=J(w)=\frac{1}{2}\left%20\|%20w%20\right%20\|^{2}+C\left%20[%20\frac{1}{N}\sum%20max(o,1%20-%20y_{i}*(w\cdot%20x_{i}+b))%20\right%20])

The common thing to do instead of calculating $$b$$ is to push be into the weights vector by adding 1 before all the rows.

To add SGD to the model we calculate the partial devivatives (the gradient vector) and then feed them into the  stochastic gradient descent algorithm. 

> Essentialy we calculate the derivatives, 
> update our weights with some learning rate,
> then we calculate the cost
> and, finally, check if it converges.


