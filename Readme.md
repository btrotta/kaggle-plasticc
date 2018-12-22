# kaggle-plasticc

Code for the 14th place solution in the Kaggle PLAsTiCC competition. 

See `Modelling_approach.pdf` for a detailed discussion of the modelling approach.

#### Quick-start guide to running the code

Total runtime is around 5.5 hours on a 24 Gb laptop.

- Download the code. Create a subfolder called `data` and save the csv files there.

- To reproduce the results exactly, create an environment with the specific
     package versions I used. (If you already have numpy, pandas, scikit-learn
     and lightgbm you can skip this 
     step, but the results may differ slightly if you have different versions.) If you have conda, the 
     easiest option is to
     build a conda environment using this command:
     ```
     conda env create environment.yml
     ```
     This will create an environment called `plasticc-bt`.
     The `requirements.txt` file is provided as well if you want to build an environment with pip.

- Run `split_test.py` to split the test data into 100 hdf5 files. They will 
 be saved in an automatically created subfolder `split_100` of the `data` folder. Takes around 15 minutes.

- Run `calculate_features.py` to calculate the features. This will generate 3 files in a folder called  
`features` (the folder is created automatically). Takes around 3.5 hours.

- Run `predict.py` to train the model and make predictions on the test set. Takes around 1.5 hours.

- Run `scale.py` to apply regularisation to the class 99 predictions and generate the final submission file. 
  Takes a couple of minutes.