# Citlalicue

### Oscar Barragán (April 2020)

*_Citlalicue_* is the name of the Aztec Goddess who created the stars. 
This small code allows you to mimic Citlalicue powers to create simulated stellar light curves.

The actual version of the code allows you to add transits and stellar variability. 
Transits are implemented using [PyTransit](https://github.com/hpparvi/PyTransit), 
while the stellar variability is added from samples of a Quasi-Periodic Kernel with covariance given by

<a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;\gamma(t_i,t_j)&space;=&space;A&space;\exp&space;\left[&space;-&space;\frac{\sin^2[\pi(t_i&space;-&space;t_j)/P_{\rm&space;GP}]}{2&space;\lambda_{\rm&space;P}^2}&space;-&space;\frac{(t_i&space;-&space;t_j)^2}{2\lambda_{\rm&space;e}^2}&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\LARGE&space;\gamma(t_i,t_j)&space;=&space;A&space;\exp&space;\left[&space;-&space;\frac{\sin^2[\pi(t_i&space;-&space;t_j)/P_{\rm&space;GP}]}{2&space;\lambda_{\rm&space;P}^2}&space;-&space;\frac{(t_i&space;-&space;t_j)^2}{2\lambda_{\rm&space;e}^2}&space;\right]" title="\LARGE \gamma(t_i,t_j) = A \exp \left[ - \frac{\sin^2[\pi(t_i - t_j)/P_{\rm GP}]}{2 \lambda_{\rm P}^2} - \frac{(t_i - t_j)^2}{2\lambda_{\rm e}^2} \right]" /></a> 

Dependencies:

* Numpy
* Matplotlib
* Scipy
* [PyTransit](https://github.com/hpparvi/PyTransit)