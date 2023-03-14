# Improving Robustness to Model Inversion Attacks via Mutual Information Regularization

This work is appeared in AAAI'21. Paper preprint is available at https://arxiv.org/abs/2009.05241. 

## Requirements (not too important as long as you can run)
torch==1.0

## Example Transcript
`python train_inv.py` used to train MI attack models.
`python train_vib.py` and `python train_dp` are used to train MID and DP models, respectively. 

For Facescrub dataset, we provide a very small subset for it. To download the full dataset, please refer to http://vintage.winklerbros.net/facescrub.html.

## GMI

`python gmi/target/train_vib.py` and `python gmi/target/train_dp.py` are used to train MID and DP models. `python gmi/attack/attack.py` is used to attack the trained model.

## Update_leaks

`python update_leaks/train_vib.py` and `python update_leaks/train_dp.py` are used to train MID and DP models. `python update_leaks/single_attack.py` is used to perform the attack.