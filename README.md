# Conditional average treatment effect estimation with marginally constrained models

This repository contains the code reproduce the results of the paper

"Conditional average treatment effect estimation with marginally constrained models" by Wouter van Amsterdam and Rajesh Ranganath, published in the Journal of Causal Inference, 2023

[![DOI](https://zenodo.org/badge/666127129.svg)](https://zenodo.org/badge/latestdoi/666127129)

## installing

have python 3 and pip installed

clone this repository and cd to it

```
pip install .
```

## reproducing the experiments




```python
python binarymcm/experiments.py [experimentname]
```
where experiment name in [collapsibility, confounding, efficiency, fullgrid]



timings on M1 mac:

- collapsibility 15.29 seconds
- confounding 17.33 seconds
- efficiency 748 seconds (12.5 min)

linux machine with 64G of RAM
- fullgrid ... hours (overnight)

After each experiments, run the R-script in the corresponding results/[experimentname] directory, to reproduce the plots and statistical analyses of the results

## Troubleshooting

### Running out of memmory

If you run out of memmory for the efficiency or fullgrid experiments, go to the corresponding 'make\_grid\_[experiment]' function.
Partition the grid in smaller chunks and run the chunks sequentially.
The R-code will later bind the results together.

