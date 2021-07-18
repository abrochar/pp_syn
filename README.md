export PATH="$PATH:~/anaconda3/bin"  
conda create --name pp python=3.6  
source activate pp  
install required packages:  
- conda install scipy  
- conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch  
- pip install matplotlib  
- pip install appdirs  
- pip install tqdm  
add path to directory in example/pp_lbfgs2_par.py (maybe do this automatically?)  

* to run the demos:
  * for periodic case, voronoi example:
```bash
python example/pp_pp_lbfgs2_par.py
```
  * aperiodic, or turbulence:
```bash
python example/pp_pp_lbfgs2_par.py --periodic False --data turbulence
```






