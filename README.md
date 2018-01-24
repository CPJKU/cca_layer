# End-to-End Cross-Modality Retrieval with CCA Projections and Pairwise Ranking Loss
This repository contains code for all of the three cross-modality retrieval methods
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

- *A Canonically Correlated Embedding Layer optimized with a pairwise ranking loss (**our proposal**)*

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

##### Audio-to-Sheet Music Retrieval
For the second data set we learn retrieval embedding spaces for complex audio - sheet music pairs.
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

##### Text-to-Image
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
##### Audio-to-Score

To train the audio-score retrieval models run:
```
python run_train.py --model models/<model>.py --data iapr
```
where *<model>* can be again one of the following options:<br>
(audio_score_ccal_tno, audio_score_learned_cont, audio_score_ccal_cont)


## Model Evaluation

##### Visualization of Training Progress
To visualize the evolution of your models during training you can run the following command:
```
python plot_log.py model_params/audio_score_*/results.pkl --key "mrr_%s" --high_is_better
```
This will plot you the Mean Reciprocal Rank (MRR) of your models on train and validation set
over the training epochs. Below you see an exemplar plot for the audio-score data set and all three models:

![Audio Score Pairs](Model_Evolution_Audio_Score.png?raw=true)

##### Evaluating on the Test Set
To test the performance of a model on the test set you can run  the follwing command
```
python run_eval.py --model models/iapr_ccal_cont.py --data iapr
```
to change the retrieval direction simply add the flag *--V2_to_V1*.