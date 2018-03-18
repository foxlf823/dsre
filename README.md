# Distantly-supervised Relation Classification

This is a research project for distantly-supervised relation classification. We refer to the following publications:

* [Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)
* [Neural Relation Extraction with Selective Attention over Instances](https://github.com/thunlp/NRE)

We also refer to the THUNLP implement on [github](https://github.com/thunlp/NRE). The dataset comes from the paper "Riedel et al., 2010. Modeling relations and their mentions without labeled text". The evaluation method follows "Surdeanu et al., 2012. Multi-instance multi-label learning for relation extraction". 

## Requirements

* Python 3.5
* PyTorch 0.3 

## Usage

To train the PCNN model, run 'dsre.py' with the following command, where 'vec.bin', 'relatoin2id.txt', 'train.txt' and 'test.txt' can be obtained from the THUNLP implement.

```
dsre.py -signature modelname -emb ./vec.bin -rel ./relation2id.txt -traindata ./train.txt -testdata ./test.txt -train
```

To test the trained PCNN model, run 'dsre.py' with the following command.

```
dsre.py -signature modelname -emb ./vec.bin -rel ./relation2id.txt -traindata ./train.txt -testdata ./test.txt
```

To use the PCNN_ATT model, add "-model 2" to the above commands.

To show the PR curves of all models in the output directory, run:

```
dsre.py -prcurve
```
