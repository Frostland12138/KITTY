# Code for: Contrastive Learning Unlocks Geometric Insights for Dataset Pruning

## Abstract 

Dataset pruning aims at selecting a subset of the data so that the model trained on the subset performs comparably to the one trained on the full dataset. In the era of big data, unsupervised pruning of the dataset can alleviate the issue of expensive labeling process from the beginning. Existing methods sort and select instances by well-designed importance metrics, while the unsupervised ones commonly regard representation learning as a black box employed to get embeddings, with its properties remaining insufficiently explored for the dataset pruning task. In this study, we revisit self-supervised Contrastive Learning by observing the learned embedding manifold, introducing Curvature Estimation to characterize the geometrical properties of the manifold. The statistical results reveal the embedding distribution of instances on manifold surfaces is not uniform. Based on this observation, we propose an unsupervised dataset pruning strategy by performing a downsample in geometric areas with high instance density, namely KITTY sampling. Extensive experiments demonstrate that our proposed methods have achieved leading performances on CV dataset pruning compared to the baselines.

<div align=center><img src="./illus.png" width="800"/></div>

*Fig.1 (A) The embedding of instances and their augmentations are assumed to be neighboring scatters on a semantic manifold, and we propose to estimate the local curvature from them. (B) A statistical histogram of the estimated local curvature for each instance, where the height of each bar represents the number of samples within that curvature range. (C) Toy example of an ellipsoid manifold with our observed distribution, and our proposed KITTY sampling: downsample those high-instance-density regions of estimated curvature while maintaining dataset diversity, figuratively making the distribution more uniform.*

## main package version

``` text
torch==1.12.1+cu116
torchvision==0.13.1+cu116
sklearn==1.2.0
```

## How to run
### Requirements:
```shell
python -m pip install -r requirements.txt
```
### Training SimCLR:
```shell
python main.py --experiment='some experiment' --gpu='0' --dataset='CIFAR-10' --batch_size=512 --sampling='KITTY' --alpha=1.0
```
#### you will get:
  - ./save/$experiment name$+xxxepoch.csv
    - Recorded curvature of each sample on epoch xxx.
  
| Estimated curvature      | Index |
| :-----------: | :-----------: |
|   1.0    |    1    |
|  ...  |     ...    |
  - ./save/checkpointxxx.tar
#### After pretraining, you can use './save/$experiment name$+xxxepoch.csv' to perform KITTY sampling and retrain SimCLR. 
#### We have provided the pretrained csvs in 
```shell
./save/stable/
```
### Evaluating:
```shell
python linear-evaluation.py --model_path='./save/' --experiment='some experiment' --gpu='0' --dataset='CIFAR-10' --batch_size=256 --reload_model='checkpoint_dim256_25_600.tar'
```
## Core Functions:

### DGC calculation

```python
x_i = x_i.to(device)
x_j = x_j.to(device)
x_k = x_k.to(device)
x_l = x_l.to(device)
z_i, z_j, z_k, z_l = model(x_i, x_j, x_k, x_l)
#without normalization
vec1=z_i-z_j
vec2=z_i-z_k
vec3=z_i-z_l
cos_sim12 = F.cosine_similarity(vec1, vec2)
cos_sim13 = F.cosine_similarity(vec1, vec3)
cos_sim23 = F.cosine_similarity(vec2, vec3)
curv= 2*np.pi-torch.acos(cos_sim12)-torch.acos(cos_sim13)-torch.acos(cos_sim23)
```

### Data loder
```python
class MyDataSet(Dataset):-
    def __init__(self, data_set_input):
        self.data_input = data_set_input
        #Random Sampling
        all_nodes=[x for x in range(self.length)]
        self.new_samples=random.choices(all_nodes,k=37761)
        #Other Sampling
        self.new_samples=Other_Sampling()

        self.length=len(self.new_samples)
        

    def __getitem__(self, mask):
        new_mask=int(self.new_samples[mask])
        data,label=self.data_input[new_mask]
        return data,label,new_mask

    def __len__(self):
        return self.length
```
### KITTY Sampling
```python
def gauss(_mean,_sigma,x):
    u1=_mean
    sigma1=_sigma
    return np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))

table=np.array(read_in_csv('./save/stable/stable-cifar-10/CIFAR_10_pretrained.csv'))
samples=table[:,1]
indexs=table[:,2]
sigma=np.array([x*x for x in samples]).mean()-np.array(samples).mean()*np.array(samples).mean()
sigma=np.sqrt(sigma)
mean=samples.mean()

alpha=0.5
self.new_samples=[]
for index,item in enumerate(samples):
    prob=1-alpha*gauss(mean,sigma,item)
    if random.random()<prob:
        self.new_samples.append(indexs[index])
print(len(self.new_samples))
```

### GraNd
#### Firstly, remove curvature calculation from loss
#### Secondly:
```python
x_i = x_i.to(device)
x_j = x_j.to(device)

z_i, z_j = model(x_i, x_j)

z_i.requires_grad_(True)
z_j.requires_grad_(True)

......

loss.backward()

gz_i=z_i.retain_grad()
gz_j=z_j.retain_grad()

GraNd_Score=l2_norm(gz_i)+l2_norm(gz_j)

```

