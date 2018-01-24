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

# Requirements
This is the list of python packages required to run the code:
- scipy
- numpy
- matplotlib
- seaborn
- Theano [(installation instructions)](http://deeplearning.net/software/theano/install.html)
- lasagne [(installation instructions)](https://lasagne.readthedocs.io/en/latest/user/installation.html)

# Data Preparation
We provide two diverse experimental data sets along with this repository.

##### Text-to-Image Retrieval
With the first data set we tackle a classic application,
namely *text-to-image retrieval*.
In this setting we rely on pre-trained [ImageNet](http://www.image-net.org/) features
and precomputed [text features](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

##### Audio-to-Score Retrieval
For the second data set we learn retrieval embedding spaces for complex audio - score pairs.
This experiment is more interesting
as we learn the embedding networks for both modalities completely from scratch.
This will also emphasize the differences between the three methods.

![Audio Score Pairs](audio_score.png?raw=true=100x)

To get the data you can run the python script listed below.
It should automatically download the data to the correct folder in this repository.
You can of course also move the data somewhere else
and change the data root path in our settings file (*<project_root>/cca_layer/config/settings.py*).

```
python prepare_data.py
```

Overall you should have a bit more than 314MB of disk space available.


## Model Training
Once you have downloaded the data you can start training the models.

#### Text-to-Image
To train the text-to-image models run the following options:

- Deep Canonical Correlation Analysis (TNO)
```
python run_train.py --model models/iapr_ccal_tno.py --data iapr
```

- Pairwise Ranking Loss (contrastive hinge loss)
```
python run_train.py --model models/iapr_learned_cont.py --data iapr
```

- CCA-Layer optimized with Pairwise Ranking Loss
```
python run_train.py --model models/iapr_ccal_cont.py --data iapr
```
#### Audio-to-Score

To train the audio-score retrieval models run:
```
python run_train.py --model models/<model>.py --data iapr
```
where *<model>* can be again one of the following options:<br>
(audio_score_ccal_tno, audio_score_learned_cont, audio_score_ccal_cont)


## Model Evaluation

#### Visualization of Training Progress
To visualize the evolution of your models during training you can run the following command:
```
python plot_log.py model_params/audio_score_*/results.pkl --key "mrr_%s" --high_is_better
```
This will plot you the Mean Reciprocal Rank (MRR) of your models on train and validation set
over the training epochs. Below you see an exemplar plot for the audio-score data set and all three models:

![Audio Score Pairs](model_evolution_audio_score.png?raw=true)

#### Evaluating on the Test Set
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

## Applying the Models to New Retrieval Problems
If you would like to test the models on other retrieval problems,
these steps are required:
- Implement your data loading function in *cca_layer/utils/data.py*
- Add this function to *select_data()* in *cca_layer/utils/run_train.py*
- Create the model definition files in *cca_layer/models* as we did for our applications.
- Train and evaluate your models as described above.