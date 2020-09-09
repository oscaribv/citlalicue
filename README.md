# Citlalicue

### Oscar Barragán (Sept 2020)

*_Citlalicue_* is the name of the Aztec Goddess who created the stars. 
This code allows you to mimic Citlalicue powers to create and manipulate stellar light curves.

The actual version of the code allows you to create synthetic stellar light curves (transits, stellar variability and white noise)
and detrend light curves using Gaussian Processes (GPs). 

##### Dependencies:

* Numpy
* Matplotlib
* Scipy
* [PyTransit](https://github.com/hpparvi/PyTransit)
* [george](https://github.com/dfm/george)


##### Try it now!

Install it by typing

```
pip install citlalicue
```

### Simulate light curves

Transits are implemented using [PyTransit](https://github.com/hpparvi/PyTransit), 
while the stellar variability is added from samples of a Quasi-Periodic Kernel with covariance given by

<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma(t_i,t_j)&space;=&space;A&space;\exp&space;\left[&space;-&space;\frac{\sin^2[\pi(t_i&space;-&space;t_j)/P_{\rm&space;GP}]}{2&space;\lambda_{\rm&space;P}^2}&space;-&space;\frac{(t_i&space;-&space;t_j)^2}{2\lambda_{\rm&space;e}^2}&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma(t_i,t_j)&space;=&space;A&space;\exp&space;\left[&space;-&space;\frac{\sin^2[\pi(t_i&space;-&space;t_j)/P_{\rm&space;GP}]}{2&space;\lambda_{\rm&space;P}^2}&space;-&space;\frac{(t_i&space;-&space;t_j)^2}{2\lambda_{\rm&space;e}^2}&space;\right]" title="\gamma(t_i,t_j) = A \exp \left[ - \frac{\sin^2[\pi(t_i - t_j)/P_{\rm GP}]}{2 \lambda_{\rm P}^2} - \frac{(t_i - t_j)^2}{2\lambda_{\rm e}^2} \right]" /></a>

Check the example of how to use Citlalicue to create light curves in the link
[example_light_curves.ipynb](https://github.com/oscaribv/citlalicue/blob/master/example_light_curves.ipynb).

### Detrend light curves

Check the example of how to use Citlalicue to detrend light curves in the link
[example_detrending.ipynb](https://github.com/oscaribv/citlalicue/blob/master/example_detrending.ipynb).
