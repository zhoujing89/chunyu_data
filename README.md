# chunyu_data

This dataset is associated with our paper, titled:
"A Longitudinal Physician Dataset with Multimodal Consultation-derived Features from a Chinese Online Healthcare Platform".

## 1.Introduction
In this work, we contribute the following resources:
(1)A longitudinal physician dataset collected from Chunyu Doctor, a leading online healthcare platform in China.The dataset integrates physician profile features, patient feedback measures, and consultation-derived features characterizing interaction patterns, textual communication, and vocal behavior. 
(2)Python code for constructing this dataset.

## 2.Data
The dataset is organized into three directories:
### Physician Profile Feature
The Physician Profile Feature dataset comprises a longitudinal physician–week panel, where each row represents a physician–week observation uniquely identified by the combination of *doc_id* and *week*.

### Derived Variables
The Derived Variables dataset encompasses feedback metrics and multimodal consultation-derived features. Specifically, the feedback metrics are uniquely identified by the combination of *doc_id* and *week*, whereas the consultation-derived features are provided at two levels of granularity: consultation-level and weekly-level.

### Supplementary Data
This directory contains supporting resources required to interpret or reproduce the released variables.

## 3.Data processing & technical verification
The code directory contains Python scripts utilized for dataset construction and technical verification. Detailed instructions are provided in the ReadMe.txt file within the same directory.

## 4.License
All resources are licensed under the MIT license.
