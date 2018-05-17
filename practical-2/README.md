# README - ULL: Practical 1
By, Samarth Bhargav and Tharangni Harsha Sivaji

Requires the setup of the requirements present in requirements.txt file present in the root of the repository, which can be done by executing the following command (assumes Python3.5x and pip is installed):
```pip install -r requirements.txt```


## How to run

We've already run the models and extracted the `.out` files required for evaluation. To just compute the numbers (without training the models) skip to the `Evaluation` section.  

### Training
To train a model, go to the corresponding file and change parameters in the `params` dictionary. Once done, execute it

```
python skipgram.py
python bayesian_skipgram.py
python embed_align.py
```


## Evaluation
This will create the models. For evaluation, first create the corresponding `.out` by first changing params in each `evaluate_*` script, and then executing:

```
python evaluate_skipgram.py
python evaluate_bayesian_skipgram.py
python evaluate_embed_align.py
```
The last command also creates `.naacl` files to test AER.
Then to evaluate it on the LS task (and word alignment task), execute the following commands:

```
## LST
# Skipgram
python lst_gap.py lst_test.gold skipgram_lst.out out no-mwe
# BSG
python lst_gap.py lst_test.gold bs_mu_lst.out out no-mwe
python lst_gap.py lst_test.gold bs_kl_post_lst.out out no-mwe
python lst_gap.py lst_test.gold bs_kl_prior_lst_mu.out out no-mwe
# Embed Align
python lst_gap.py lst_test.gold eam.out out no-mwe


# For AER
python aer.py dev.naacl <created dev.naacl>
python aer.py test.naacl <created test.naacl>
```
