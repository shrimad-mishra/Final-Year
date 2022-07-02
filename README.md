# Prediction of Replacement and Diagnosis of Osteoarthritis using Deep Learning: Data from the Osteoarthritis Initiative/ GM Hospital

## Introduction
This repo contains implementation of the deep learning-based outcome prediction model used for osteoarthritis research as described in our paper: [........................................................................................................................] . By using this implementation, you can either train new models using nested cross-validation or obtain TKR outcome and KL grade redictions by using our pretrained models.[>>>> Model link will be given soon <<<<] 


## Instructions
1. Please refer to **requirements.txt** to install all dependencies for this project. 
2. Check **(https://drive.google.com/file/d/1rIuwM_nGRbmkuiYQW_a6bkt8pP40pKFK/view?usp=sharing)** 

## Repo Structure
* Repo includes different folders and files.

* Data folder contains --:--
    * test
    * train
    * Test.csv
    * Train.csv

* Resources folder contains --:--
    * Dataset Link
    * Extra
    * Papers

* Methodology diagram

* Main python notebook file which make all the predictions and calculate results.

* Requirements.txt which contains all the required python library.

## Training a DL model
You can use this repo to train models for predicting KR outcome and KL grade. So, you directly run the following script to train DL models with nested cross-validation. 


## Approach

* We have used Inception and Xception models to bring all the result and used the model which have high accuracy.

* Here is the architecture of the Xception Model:-
* ![Xception](https://user-images.githubusercontent.com/42738198/176997269-149e025e-67e2-4a54-b0ef-805bcc248d08.png)

* Here is the architecture of the Inception Model:-
* ![Inception drawio](https://user-images.githubusercontent.com/42738198/176997277-fe91d21d-7ae8-4f11-9b11-920db6ef88bd.png)


## Inference

* ![FLow Diagram](https://user-images.githubusercontent.com/42738198/176733893-a7cec182-a06a-4be0-9b32-5e8d68897c13.png)


--- Inference Results ---
* Total Knee Replacement (TKR): 0.86
    * Minimal  0: 0.47
    * Healthy  1: 0.41
    * Moderate 2: 0.40
    * Doubtful 3: 0.78
    * Severe   4: 0.83

## Reference
If you found this code useful, please cite our paper:

*Prediction of Total Knee Replacement and Diagnosis of Osteoarthritis using Deep Learning*
Shrimad Mishra, Sindhu Kotegar, S.G Anushka, Sneha Nanda

#### Emails 
* mshrimad@gmail.com          --- LinkedIn --- (https://www.linkedin.com/in/shrimad-mishra-45929a1a1/)
* sindhukotegar@gmail.com     --- LinkedIn --- (https://www.linkedin.com/in/sindhu-n-b7948516b/)
*
*
