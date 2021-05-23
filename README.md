# latent-mix
latent-composition and augmix


original papers:
* [Augmix](https://openreview.net/pdf?id=S1gmrxHFvB)
* [Latent Composition](https://arxiv.org/pdf/2103.10426.pdf)

Create environment
```bash
conda env create -f environment.yml
```

Run
```bash
python cifar.py -m allconv --latent-composition --batch-size 1
```
