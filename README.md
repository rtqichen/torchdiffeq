# Neural ODEs - Latent ODE Experiments

This is the code supporting Lynn Pepin and Nila Mandal's Spring 2019 CSE 5820 group project. This work was forked from <https://github.com/rtqichen/torchdiffeq/>. See that repository for `torchdiffeq` installation instructions and more examples.

## latent_ode.py experiments

The original experiments dealt with datasets of points drawn from one of two spirals. The goal was to produce a model that, given a set of noisy points drawn from one of the two spirals, learn to extrapolate forwards and backwards.

Those experiments show neural ODEs to be superior in comparison with RNNs, but the dataset is still extremely simple and limited. We take an approach here, augmenting the dataset with 10 classes of shapes plus transformations.

10 classes:
 * Sine curve ('snake')
 * Trefoil knot
 * Rose, k = 7
 * Rose, k = 2
 * Torus knot (p=1, q=6)
 * Torus knot (p=2, q=3)
 * Torus knot (p=3, q=2)
 * Archimedian spiral
 * Logarithmic spiral
 * Spirograph (a = 11, b =6)

 
Data augmentation: Rotation and translation
Evaluation: Evaluated on ability to reconstruct true curve.

## Basic usage

### References

---

If you found these experiments useful, please be sure to check and cite the original authors:
```
@article{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  journal={Advances in Neural Information Processing Systems},
  year={2018}
}
```
