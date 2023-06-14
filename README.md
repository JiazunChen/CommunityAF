# CommunityAF

code for vldb23 "CommunityAF: An Example-based Community Search Method via Autoregressive Flow"

## Datasets


For a fair comparison, for each dataset we perform the same preprocessing as the previous work. The codes for the preprocessing  can be found in [SEAL](https://github.com/FDUDSDE/SEAL):

(1) We remove outliers. Specifically, we neglect communities whose sizes are beyond the $90$-th percentile and those whose sizes are less than $3$. For example, for DBLP, the maximum community size is close to $8000$, and the average size of the last $90$-th is only $8$.

(2) We remove redundant nodes that belong neither to communities nor to community neighbors to simplify the computation. This is because ground truth communities are sparsely distributed in the dataset, and we focus on the interaction between communities and their neighbor nodes. We also test methods on the dataset without removing redundant nodes, which better fits the real world scenario.

(3) For datasets with attributes, we use PCA to reduce the dimension in advance to speed up the subsequent computation and save memory.

We provide five pre-processed datasets, including `facebook, dblp, amazon, twitter, youtube`. `orkut` is out of size limit and cannot be uploaded, raw data can be  downloaded from [SNAP](https://snap.stanford.edu/data/com-Orkut.html).
## Dependencies

python == 3.7
 
torch == 1.5.0

nextworkx == 2.6.3

scipy == 1.7.1

sklearn == 0.24.0

tqdm == 4.55.1

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
 


