
Code for the paper "Particle gradient descent model for point process generation".


**Requirements**:
- Pytorch (version <= 1.7.1)
- Numpy, scipy, matplotlib, tqdm, appdirs


**To run the demo**:
- For periodic case, Voronoi example:
```bash
python examples/pp_lbfgs2_par.py
```
- aperiodic case, or Turbulence example:
```bash
python examples/pp_lbfgs2_par.py --aperiodic --data turbulence
```

Other arguments, to specify the model, can be found in 'examples/pp_lbfgs2_par.py'.
