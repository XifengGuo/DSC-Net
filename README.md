## Pytorch implementation of DSC-Net (Deep-subspace-clustering-networks)

A pytorch implementation of the following paper:
- Pan Ji*, Tong Zhang*, Hongdong Li, Mathieu Salzmann, Ian Reid. Deep Subspace Clustering Networks. in NIPS'17.

The original Implementation by Tensorflow can be found at 
[Orginal code](https://github.com/panji1990/Deep-subspace-clustering-networks).

So why re-implement it by Pytorch?
Just getting bored recently.

### Results

AS shown in the following table, this implementation has the same result with Tensorflow's.
The pretrained weights are converted from Tensorflow.
Due to the different padding behavior of Conv2d/ConvTranspose2d between Pytorch and Tensorflow,
I spent extra effort on implementing by Pytorch the "SAME" padding mode in Tensorflow.


| Dataset | ACC (This) | NMI (This) | ACC ([Original](https://github.com/panji1990/Deep-subspace-clustering-networks)) | NMI ([Original](https://github.com/panji1990/Deep-subspace-clustering-networks)) | ACC ([Paper](https://arxiv.org/abs/1709.02508))
| :---: | :---: | :---: | :---: | :---: | :---: | 
| YaleB | 0.9733 | 0.9634 | 0.9729 | 0.9629 | 0.9733 |
| ORL   | 0.8550 | 0.9195 | 0.8450 | 0.9177 | 0.8600 |
| COIL20| 0.9100 | 0.9587 | 0.8917 | 0.9547 | 0.9486 |

### Usage

Step 1: Prepare pytorch environment. See [Pytorch](https://pytorch.org/get-started/locally/).

Step 2: Prepare data. Download all data from https://github.com/panji1990/Deep-subspace-clustering-networks .
Then copy the .mat files in `Data` folder to `datasets` folder in this repo.

Step 3: Run on COIL20, COIL100, ORL by using the following commands.
```
python main.py --db coil20  # you should uncomment line 64 of post_clustering.py to get better result on this dataset.
python main.py --db coil100  # I didn't test on this dataset since my laptop is too old.
python main.py --db orl --show-freq 100
```

Step 4: Run on YaleB dataset with all 48 classes.
```
python yaleb.py 
```
If you want to get the result with different classes as shown in Table 2 in [Paper](https://arxiv.org/abs/1709.02508),
just change the line 220 in yaleb.py to something like
`all_subjects = [10, 15, 20, 25, 30, 35, 38]`

### Conclusion
- I successfully re-implemented the [Tensorflow code](https://github.com/panji1990/Deep-subspace-clustering-networks)
of DSC-Net by Pytorch.
- Two Padding modules were defined to implement the "SAME" padding mode of Tensorflow.  

### TODO

- I'm gonna complete the pretraining code and delete the "SAME" padding modules (which are essential for pretrained 
Tensorflow weights) to make the code more cleaner and easier to extend.
But for my own use.