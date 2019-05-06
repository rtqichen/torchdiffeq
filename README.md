# Neural ODEs - Latent ODE Experiments

This is the code supporting Lynn Pepin and Nila Mandal's Spring 2019 CSE 5820 group project. This work was forked from <https://github.com/rtqichen/torchdiffeq/>. See that repository for `torchdiffeq` installation instructions and more examples.

## latent_ode.py experiments

The original experiments dealt with datasets of points drawn from one of two spirals. The goal was to produce a model that, given a set of noisy points drawn from one of the two spirals, learns to interpolate the shape as well as extrapolate forwards and backwards.

Those experiments show neural ODEs to be superior in comparison with RNNs, but the dataset is still simple and limited. We produce a dataset of 10 trajectories of shapes, plus augmentation:

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

See `parametric_dataset.py`.

## Basic usage

To repeat the experiments as produced in the paper, simply run `latent_ode.py`, with flags as such:

 * `--exp1` : Experiment 1
 * `--exp1s` : Experiment 1, on the smaller dataset
 * `--exp2` : Experiment 2
 * `--exp3` : Experiment 3
 * `--adjoint` : Use the adjoint sensitivity method.
 * `--device [string]` : What device to run the learning on. (E.g. `cuda:0`, `cuda:1`, `cpu`, ...)

Be advised, if you run all the experiments at once (e.g. `python latent_ode.py --exp1 --exp1s --exp2 --exp3`) you could run out of memory!

If you don't run multiple GPUs, you probably don't need to set --device.

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
