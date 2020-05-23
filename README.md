# NLP
This repo is for the project in the Obdelava naravnega jezika class

**Data parsing** directory contains two python files. One parses the data and the other prints some basic statistics of the dataset.

**Attribute retrieval** directory contains all the code used to construct attributes for the data. Before running any other code the files from this directory have to be run in this order:
  - props1.py
  - props2.py
  - props3.py
  - props4.py
  - get_znacilke.py
  - order_data_for_learning.py

**Neural network** directory contains the code used for training and testing the BERT model. Before you are able to run the code you need to install the packages listed in requirements.txt (pip3 install -r requirements.txt) and download the saved trained model from: [ner_bert_pt2.pt](https://drive.google.com/drive/folders/1X3Iec_dC1bTq5Wb0giivvE_6BgQIUN2p?usp=sharing) (use your email from e-classroom) as well as the pretrained model directory from: [slo-hr-en-bert-pytorch](https://unilj-my.sharepoint.com/personal/slavkozitnik_fri1_uni-lj_si/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fslavkozitnik%5Ffri1%5Funi%2Dlj%5Fsi%2FDocuments%2Ftemp&originalPath=aHR0cHM6Ly91bmlsai1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9zbGF2a296aXRuaWtfZnJpMV91bmktbGpfc2kvRXVVOGV1OUpIclpFdGtKb01odnpMNW9COFBqVURQMERteXdUeFI1cjNsOGRRQT9ydGltZT1oZUxUUjBQLTEwZw"). If there are any problems with downloading the required models please contact _dp2576@student.uni-lj.si_. The code will have to load a bit and then perform some iterations which all together will take a minute or two. After that it will print out the evaluation of the model.

**CRF** directory contains code used for data preparation (5 distinct experiments), hyper-parameter 
tuning and cross validation of the CRF model. Read CRF/README.md for instructions on how to use it.
