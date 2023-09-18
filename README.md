MRNDR_Code   
MRNDR(Multi-head attention-based Recommendation Network for Drug Repurposing)       
environment    
###python==3.7.11    
###torch==1.9.1   
###pyqt5==5.15.4   
data is the BioRE data set, which includes Drug-Disease, Drug-Target, Target-Disease, and Target-Pathway.   
BioRE address is packaged to: [https://github.com/Lindamansen/BioRE_dataset](https://github.com/Lindamansen/BioRE_dataset)    
The code has been packaged to:[https://github.com/Lindamansen/MRNDR_Code](https://github.com/Lindamansen/MRNDR_Code)     
1.Train is the code for training the model, and you can directly run the train.py in it        
2. The recommended potential drug system is Recommended system.py      
3. The front end is front end.py, which displays recommended drugs and diseases     
###Word2Vec.py encodes Word2Vec          
###Data processing for dataset.py   
###MRNDR_model of model.py   
###Crawling the number of documents related to drug diseases for Literature crawler.py   
###Parper_exam carries the public evaluation indicators of MRR and Hits@N   
###In addition to data, the data folder also contains code solutions for rare recommended methods     
For other questions, please leave a message to the email address£º[mafeng21331@foxmail.com](mafeng21331@foxmail.com)
