# maximally_predictive_states
This repository contains the main scripts for the maximally predictive state space reconstruction and ensemble dynamics modelling presented in

Costa AC, Ahamed T, Jordan D, Stephens GJ "Maximally predictive states: from partial observations to long timescales" [*Chaos*](https://aip.scitation.org/doi/full/10.1063/5.0129398) 2023

The data for reproducing the figures can be found in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7130012.svg)](https://doi.org/10.5281/zenodo.7130012). We provide a run through all the steps on the analysis in two model systems in the folder ./ExamplePipeline

For a follow-up application in *C. elegans* postural time series, check this [repository](https://github.com/AntonioCCosta/markov_worm) and the corresponding [publication](https://www.pnas.org/doi/10.1073/pnas.2318805121).


Any comments or questions, contact antonioccosta.phys(at)gmail(dot)com. Also, suggestions to speed up the code are more than welcome!


-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------


The code is fully written in python3, and we use of the following packages:

- h5py '3.0.0'
- sklearn '0.23.1'
- matplotlib '3.3.4'
- msmtools '1.2.5'
- scipy '1.3.1'
- numpy '1.17.2'
- joblib '0.13.2'
- cython '0.29.23' 
- findiff '0.9.2'
