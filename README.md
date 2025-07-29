# Joint Health Level Classification
Classify trapeziometacarpal osteoarthritis from multi-modal data, including CT images, kinematic data, and mechanical/geometry measurements. MRI images will be included in the future.
### Dependencies:
    torch >= 1.13.1
    torchinfo >= 1.8.0
    torchvision >= 0.14.1
    xgboost >= 1.6.2
    imagecodecs
    imageio 
    jupyter
    ipykernel 
    matplotlib  
    matplotlib_venn
    numpy 
    opencv-contrib-python  
    pandas
    Pillow  
    scikit-image 
    scikit-learn 
    scipy
    seaborn 
    yaml
    glob
    
  
### Folders:
    image_model: scripts for training and testing models using CT images.
                  usage: python run.py
    kinematic_model: scripts for training and testing models using motion tracking data.
                  usage: python run.py 
    configs are saved in config.yaml
### Notebooks:
    tmc_data.ipynb: pipeline for pre-processing data, including resaving data into pkl files and creating config.yaml for model training.
    tmc_data_plot.ipynb: scripts for visualizing the dataset.
    tmc_xgboost.ipynb: pipeline for training & testing xgboost models
    tmc_results.ipynb: scripts for summarizing testing results from all models.
    tmc_model_debug.ipynb: scratch working space for debugging.
      
