# Multi-resolution Time-Series Transformer for Long-term Forecasting (MTST). [arxiv](https://arxiv.org/pdf/2311.04147.pdf)
This paper has been accepted at AISTATS 2024.


## Getting Started


1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

### Multi-resolution Time Series Transformer (MTST) 

1. Training. All the scripts are in the directory ```./scripts/MTST```. For example,
```
sh ./scripts/MTST/etth2.sh
```


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

