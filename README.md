# Distantly-supervised Relation Classification

This is a research project for distantly-supervised relation classification. We refer to the following publications:

* [Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)

We also refer to the THUNLP implement on [github](https://github.com/thunlp/NRE). The dataset comes from the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text."


## Requirements

* Python 3.5
* PyTorch 0.3 

## Usage

To train a model, run 'dsre.py' with the following command, where 'vec.bin', 'relatoin2id.txt', 'train.txt' and 'test.txt' can be obtained from the THUNLP implement.

```
dsre.py -emb ./vec.bin -rel ./relation2id.txt -traindata ./train.txt -testdata ./test.txt -train
```

To test using a existing model, run 'dsre.py' with the following command.
```
dsre.py -emb ./vec.bin -rel ./relation2id.txt -traindata ./train.txt -testdata ./test.txt -output modelpath
```
