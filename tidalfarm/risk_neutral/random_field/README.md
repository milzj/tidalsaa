# Random field

We use a truncated series expansion of the form

$$
  \kappa(x,\xi) = \kappa_0 + \sum_{k, j=1}^p  \lambda_k  \Big( \frac{1}{\sqrt{\ell_1\ell_2}2 \cos(\pi j x/\ell_1) \cos(\pi k y/\ell_2) \Big),
$$

where $\ell_1 = 1000$, $\ell_2 = 2000$. We have

[We have](https://www.wolframalpha.com/input?i=integrate+%282+cos%28pi+x%2F1000%29+cos%28pi+y%2F2000%29%2Fsqrt%281000*2000%29%29%5E2+for+x%3D0..1000+and+y%3D0..2000)

$$
  \int_{x\in [0,1000], y \in [0,2000]} (2 \cos(\pi x/1000) \cos(\pi y/2000)/\sqrt{1000*2000})^2 \mathrm{d}x \mathrm{d}y = 1.
$$




