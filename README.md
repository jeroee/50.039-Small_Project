# 50.039-Small_Project

50.039 Deep Learning Small Project Report

Jeremy Ng 1003565

Zhou Yutong 1003704

A project on building a custom deep learning model where the model will assist with the diagnosis of pneumonia, for COVID and non-COVID cases, by using X-ray images of patients.



Instructions: 
1.  Clone directory
2.  Refer to instructions in https://drive.google.com/drive/folders/1jBfdrcVBr_W9W_9y2-VTWXpgx7ctx2Z5?usp=sharing 
3.  customCnn_1.py & customCnn_2.py: custom CNN models implemented for the small project. 
    customDataset_1.py & customDataset_2.py: custom datasets implemented for the small project.
    utils.py: common useful functions which are shared between both models.
    Graph folder: stores accuracy and loss graphs of the models
    Models folder: stores the models 
    Checkpointing.ipynb: Demonstration of checkpoint saving and reloading of model to resume training.
    Final_confusion_matrix.png: confusion matrix of output after undergoing two models
4.  To run the full code on training the models and displaying final results, open Small_project_sequential.ipynb and run all cells. 
*Important Note: The weights of the models we trained were saved under their respective folders in the “models” directory. Re-running the notebook will overwrite the saved weights.

5.  To run the demonstration of how our code implements checkpoint saving upon premature termination during training, reloading the checkpoint model to resume training, open checkpointing.ipynb and run all cells. When the model is being trained half way, interrupt the process. Run the cells below to load and continue training the model.
