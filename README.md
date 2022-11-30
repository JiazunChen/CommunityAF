# CommunityAF

code for submission "CommunityAF: An Example-based Community Search Method via Autoregressive Flow"

## Datasets

We provide five pre-processed datasets, including `facebook, dblp, amazon, twitter, youtube`.  The codes for the preprocessing  can be found in [SEAL](https://github.com/FDUDSDE/SEAL).

## Dependencies

 python == 3.7
 
 torch == 1.5.0

nextworkx == 2.6.3

scipy == 1.7.1

sklearn == 0.24.0

## Train


 

We provide the default hyperparameters for easy running.

Run the algorithm:
```
python main.py --dataset facebook --default_parameters
```
Or you can try different hyperparameters:
```
python main.py --dataset facebook --cnt_rank 0 --num_flow_layers 4 ... 
```
There are some problems with multiprocessing when running CSAF on a Windows environment. You need to use the parameter multiprocessing to turn it off:
```
python main.py --dataset facebook --default_parameters --multiprocessing
```

## Evaluation

By default, the best models are saved in the `./result/{time}/checkpoint{best_epoch}`

Run the algorithm:

```
python main.py --eval_path ./result/{time}/checkpoint{best_epoch}
```
 


