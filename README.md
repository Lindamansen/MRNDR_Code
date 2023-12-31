# MRNDR model
MRNDR(Multi-head attention-based Recommendation Network for Drug Repurposing)

# Requirements and Installation
* [PyTorch](http://pytorch.org/) version == 1.9.1
* Python version == 3.7.11
* pyqt5==5.15.4

|Name|URL|
|----|----|
|BioRE dataset|[link](https://github.com/Lindamansen/BioRE_dataset)|
|MRNDR model|[link](https://github.com/Lindamansen/MRNDR_Code)|

* BioRE
```
data is the BioRE data set, which includes Drug-Disease, Drug-Target, Target-Disease, and Target-Pathway.
```

* Train is the code for training the model, and you can directly run the train.py in it,
* The recommended potential drug system is Recommended system.py,
* The front end is front end.py, which displays recommended drugs and diseases.

``` 
Data processing for dataset.py
MRNDR_model of model.py
Crawling the number of documents related to drug diseases for Literature crawler.py
Parper_exam carries the public evaluation indicators of MRR and Hits@N
In addition to data, the data folder also contains code solutions for rare recommended methods.
Word2Vec.py encodes Word2Vec
```

# Email
For other questions, please leave a message to the email address��[mafeng21331@foxmail.com](mafeng21331@foxmail.com)
