# End-to-End Cross-Modality Retrieval with CCA Projections and Pairwise Ranking Loss
This repository contains the code for all three cross-modality retrieval methods
evaluated in our manuscript listed below:

>End-to-End Cross-Modality Retrieval with CCA Projections and Pairwise Ranking Loss.<br>
Dorfer M., Schlüter J., Vall A., Korzeniowski F., and Widmer G.<br>
Under Review for the International Journal of Multimedia Information Retrieval, 2018

This README explains how to set up the project,
how to get the data,
and how to train and evaluate the different retrieval models.

In particular this repository contains code for the following methods:

- *Deep Canonical Correlation Analysis (optimizing the Trace Norm Objective (TNO))*

- *A learned, linear embedding layer optimized with a pairwise ranking loss*

- **A Canonically Correlated Embedding Layer optimized with a pairwise ranking loss (our proposal)**

The main purpose of this repository is to make the methods evaluated in our article
easily applicable to new retrieval problems.
For details on the three retrieval paradigms we refer to the corresponding article.

# Table of Contents
  * [Requirements and Installation](#installation)
  * [Data Preparation](#data_prep)
  * [Model Training](#training)
  * [Model Evaluation](#evaluation)
  * [Applying the Models to New Retrieval Problems](#new_problems)
    * [Required Steps](#new_model_steps)
    * [Hyper-parameter Recommendations](#hyper_params)

# Requirements and Installation <a id="installation"></a>
This is the list of python packages required to run the code:
- scipy
- numpy
- matplotlib
- seaborn
- Theano [(installation instructions)](http://deeplearning.net/software/theano/install.html)
- lasagne [(installation instructions)](https://lasagne.readthedocs.io/en/latest/user/installation.html)

Once all requirements are available
we recommend to install the *cca_layer package* in develop mode using the following command:
```
python setup.py develop --user
```

# Data Preparation <a id="data_prep"></a>
We provide two diverse experimental data sets along with this repository.

##### Text-to-Image Retrieval
With the first data set we tackle a classic application,
namely *text-to-image retrieval*.<br>
In this setting we rely on pre-trained [ImageNet](http://www.image-net.org/) features
and precomputed [text features](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

##### Audio-to-Score Retrieval
For the second data set we learn retrieval embedding spaces for complex audio - score pairs.
This set of experiments is more interesting,
as we learn the embedding networks for both modalities entirely from scratch.
This will emphasize the differences between the three methods.

![Audio Score Pairs](audio_score.png?raw=true=100x)

To get the data you can run the python script listed below.
It should automatically download the data to the correct folder in this repository.
You can of course also move the data somewhere else
and change the data root path in our settings file (*<project_root>/cca_layer/config/settings.py*).

```
python prepare_data.py
```

Overall you should have a bit more than 314MB of disk space available.


# Model Training <a id="training"></a>
Once you have downloaded the data you can start training the models.

### Text-to-Image
To train the text-to-image models run the following options:

- Deep Canonical Correlation Analysis (TNO)
```
python run_train.py --model models/iapr_ccal_tno.py --data iapr
```

- Pairwise Ranking Loss (contrastive hinge loss)
```
python run_train.py --model models/iapr_learned_cont.py --data iapr
```

- **CCA-Layer optimized with Pairwise Ranking Loss**
```
python run_train.py --model models/iapr_ccal_cont.py --data iapr
```
### Audio-to-Score

To train the audio-score retrieval models run:
```
python run_train.py --model models/<model>.py --data audio_score
```
where *<model>* can be again one of the following options:<br>
(audio_score_ccal_tno, audio_score_learned_cont, audio_score_ccal_cont)


# Model Evaluation <a id="evaluation"></a>

## Visualization of Training Progress
To visualize the evolution of your models during training you can run the following command:
```
python plot_log.py model_params/audio_score_*/results.pkl --key "mrr_%s" --high_is_better
```
This will plot you the Mean Reciprocal Rank (MRR) of your models on train and validation set
over the training epochs. Below you see an exemplar plot for the audio-score data set and all three models:

![Audio Score Pairs](model_evolution_audio_score.png?raw=true)

## Evaluating on the Test Set
To test the performance of a model on the test set you can run  the following command
```
python run_eval.py --model models/iapr_ccal_cont.py --data iapr
```

Running the command above should give you something like this:

```
Loading data...
Train: 17000
Valid: 1000
Test: 2000

Compiling prediction functions...
Evaluating on test set...
Computing embedding ...
lv1_latent.shape: (2000, 128)
lv2_latent.shape: (2000, 128)

Hit Rates:
Top 01: 31.050 (621) 31.050
Top 05: 58.250 (1165) 11.650
Top 10: 69.700 (1394) 6.970
Top 25: 81.700 (1634) 3.268

Median Rank: 3.00 (2000)
Mean Rank  : 33.99 (2000)
Mean Dist  : 0.58933 
MRR        : 0.440 
Min Dist   : 0.18588 
Max Dist   : 1.16693 
Med Dist   : 0.59117
```

If you would like to change the retrieval direction simply add this flag to the evaluation command:
```
--V2_to_V1
```

# Applying the Models to New Retrieval Problems <a id="new_problems"></a>

## Required Steps <a id="new_model_steps"></a>
If you would like to test the models on other retrieval problems,
these steps are required:
- Implement your data loading function in *cca_layer/utils/data.py*
- Add this function to *select_data()* in *cca_layer/utils/run_train.py*
- Create the model definition files in *cca_layer/models* as we did for our applications.
- Train and evaluate your models as described above.
- Tweak hyper-parameters

## Hyper-parameter Recommendations <a id="hyper_params"></a>
Depending on your problem you might need different hyper parameter settings
to get to the best retrieval performance out of your models.
Here are just a few practical recommendations where you can start to tweaking:

```
DIM_LATENT = 32     # dimensionality of retrieval space
BATCH_SIZE = 100    # batch-size used for training
```
Depending on the problem we set the dimensionality of the latent space (*DIM_LATENT*)
to values such as [32, 48, 64, 128].
Keep in mind that the CCA-Layer maintains statistics (e.g. Covariance Matrices)
of the mini-batches used for training
as done for example in [batch normalization](https://arxiv.org/abs/1502.03167).
To get stable covariance estimates we need to set the *BATCH_SIZE*
to a value at least as large as *DIM_LATENT*.
For the two examples we use the following combinations that worked well for us:
(DIM_LATENT=32, BATCH_SIZE=100) and (DIM_LATENT=128, BATCH_SIZE=1000).
In general if you encounter numerical instabilities while training your models,
simply increase your BATCH_SIZE or reduce the dimensionality of your retrieval space.

```
INI_LEARNING_RATE = 0.001   # initial learning rate (0.001 or 0.002)
MAX_EPOCHS = 1000           # limits the maximum number of training epochs
PATIENCE = 30               # number of epochs without improvment before the learning rate get s reducede
REFINEMENT_STEPS = 3        # once patience expiers we reduce the learning rate.
                            # this parmeter controls how often this procedure is repeated.
LR_MULTIPLIER = 0.5         # this is the factor by which the learning rate gets multiplied (0.1, 0.5)
```
These are the parameters to control your learning rate as well as your learning rate schedule.
If you get the initial learning rate right, the rest is not too crucial.

```
L2 = 0.00001        # degree of L2 regularization (weight decay)
r1 = r2 = 1e-3      # Tikhonov regularization of covariance matrices (e.g. C + rI)        
rT = 1e-3
```
These are regularization parameters.
The *r* parameters are used to [regularize the covariance matrices.](https://en.wikipedia.org/wiki/Tikhonov_regularization),
which is common for Deep Canonical Correlation Analysis.

```
ALPHA = 1.0         # maintain exponential running average of batch statistics
                    # ]0.0, 1.0] (if 1.0 we only take the most recent batch into account)
WEIGHT_TNO = 1.0    # (range: 0 to 1) controls the influence of the Trace Norm Objective (TNO)
                    # if 1.0 only TNO. if 0.0 only pairwise ranking loss
USE_CCAL = True     # if True we make use of the proposed CCA-Layer embedding layer
                    # if False we learn an embedding layer from scratch
GAMMA = 0.5         # margin parameter of pairwise ranking loss (0.5, 0.7)
```