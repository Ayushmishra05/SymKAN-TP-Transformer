# Decoder KAN (Kolmogorov-Arnold Networks)

## Task 1 :  Extracting the Data From the sources 
Starting off with the Data Extraction part, The Data was collected from the listed source, the data was in the Raw Text format, the first task was to convert this Raw text data into a CSV formatted dataset
* I have used re (python module), which is python module for regular expressions 
* The Dataset was converted into individual variables, and then was embedded into the dataset 
* Vertex was not consistent across the data, this might be the case because, some of the particle collisions doesn't happen between vertices but it only depends on the single vertex, that's why the case 


## Task 2 : Tokenization  
The Tokenization technique used here was 
   **Mathematical Aware Tokenization Method** 
   However there were various tokenization methods, that was explored, like **Character Wise Encoding, Byte Pair Encoding**, out of this **Mathematical Aware tokenization gave the good results**, The Working of this Tokenization Method is explained in this Paper <a href="https://cdn.iiit.ac.in/cdn/web2py.iiit.ac.in/research_centres/publications/download/inproceedings.pdf.867521e9a9170b72.312e393738313631313937373137322e33332e706466.pdf" style="color:blue;">Paper Link</a> (This paper shows, ho Performance increases significantly in math and physics related tasks, using this tokenization method). 


## Task 3 : Transformer 
the Transformer model was trained on the dataset, initially the plan was to use the Decoder Architecture, with the KAN layers, but because Decoders only works in language tasks, i cancelled the plan of training Decoder 

* This Transformer architecture is the same as, which is provided in the paper, Now for comparison a basic Transformer was built to train on the dataset, the dataset consisted of 15K Rows, which was provided, by the org
* After training the Transformers for 10 epochs, the Accuracy came out to be 99.5%, Now the task was to improve it and also bring the interpretability here 

## Approach - SymKAN-TPT (Tensor-Product Transformer combined with KAN and Sympy)
 ### Why this Approach 
* The Last Approach for the Same Problem Introduced 