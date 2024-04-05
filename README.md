# A Supervised Variational Auto-Encoder for Human Motion Generation using Convolutional Neural Networks

This paper is accepted at the 2024 International Conference on Pattern Recognition and Artificial Intelligence ([ICPRAI2024](https://brain.korea.ac.kr/icprai2024/)).<br>
Authors: [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/), [Maxime Devanne](https://maxime-devanne.com/), [Stefano Berretti](http://www.micc.unifi.it/berretti/), [Jonathan Weber](https://www.jonathan-weber.eu) and [Germain Forestier](https://germain-forestier.info/).

## Requirements

```
tensorflow==2.10
numpy
pandas
sklearn
matplotlib
scipy
imageio
```
You will need to install as well ```fmpeg``` on your system.

## Train Classifier

In order to evaluate at the end the generative models, we will need to first train the GRU based classifier on the real data.

### In the case of no train/test split needed

```python3 main.py --output-directory results/ --train-on real --dataset HumanAct12 --runs 5 --split all --n-epochs 2000```

### In case of a train/test split needed

```python3 main.py --output-directory results/ --train-on real --dataset HumanAct12 --runs 5 --split train_test --n-epochs 2000```

And the code will use the 4 cross subject split used for the HumanAct12.

## Train VAE and SVAE Generators

### For VAE

Training on all the dataset:

```python3 train_vae.py --generative-model VAE --dataset HumanAct12 --output-directory results/ --runs 5 --split all --weight-rec 1.0 --weight-kl 1.0 --n-epochs 2000```
<br>
To train on the train test cross subject splits proposed in the paper, use the train_test value for the ```--split``` argument.

### For VAE

Training on all the dataset:

```python3 train_vae.py --generative-model SVAE --dataset HumanAct12 --output-directory results/ --runs 5 --split all --weight-rec 0.4995 --weight-kl 0.001 --weight-cls 0.4995 --n-epochs 2000```
<br>
To train on the train test cross subject splits proposed in the paper, use the train_test value for the ```--split``` argument.

## Generating with Pre-trained Generators

```python3 generate_samples.py --dataset HumanAct12 --generative-model SVAE --run 0 --plot-skeletons True --weight-rec 0.4995 --weight-kl 0.001 --weight-cls 0.4995 --save-skeletons True --class-generate 0 --n-samples 5 --best-predictions 3 --output-directory results/```

## Evaluation with FID and Diversity Metrics

### On generated Data by Pre-trained Generators

```python3 calculate_FID_Diversity.py --dataset HumanAct12 --output-directory results/ --generative-model SVAE --weight-rec 0.4995 --weight-kl 0.001 --weight-cls 0.4995 --on generated --n-generations 5 --n-factors 1```

### On Real Data

```python3 calculate_FID_Diversity.py --dataset HumanAct12 --output-directory results/ --on real```

## Cite This Work

If you use this work please cite the following:

```
@inproceedings{Ismail-Fawaz2024SVAE,
  author = {Ismail-Fawaz, A. and Devanne, M. and Berretti, S. and Weber, J. and Forestier, G.},
  title = {A Supervised Variational Auto-Encoder for Human Motion Generation using Convolutional Neural Networks},
  booktitle = {International Conference on Pattern Recognition and Artificial Intelligence (ICPRAI)},
  year = {2024}
}
```

## Acknowledgments

This work was supported by the ANR DELEGATION project (grant ANR-21-CE23-0014) of the French Agence Nationale de la Recherche. The authors would like to acknowledge the High Performance Computing Center of the University of Strasbourg for supporting this work by providing scientific support and access to computing resources. Part of the computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d’Avenir) and the CPER Alsacalcul/Big Data. The authors would also like to thank the creators and providers of the HumanAct12 dataset.