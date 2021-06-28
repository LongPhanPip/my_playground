# Machine learning
## Common function
### Sigmod
> $\sigma (x) = \frac{1}{1 + e^{-x}}$
> $\sigma ^{'}(x) = \sigma (x)(1 - \sigma (x))$

## Lineaar model
### Linear regression
> $\frac{1}{2n_{sample}}min\left \| Xw - y \right \|_{2}^{2} + \alpha \rho \left \| w \right \|_{1} + \frac{\alpha(1 - \rho )}{2}\left \| w \right \|_{2}^{2}$ 
 with $\rho$ is percentage of $l_{1}$, 0 is  Ridge, 1 is Lasso
 Higher $\alpha$ less overfitting usually 1

### Logistic regression 

