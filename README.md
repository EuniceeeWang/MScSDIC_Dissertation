# MScSDIC_Dissertation: Airway Segmentation from lung CTs based on UNet andGraph Neural Network method

All functions, modules, classes are defined in files in src folder. Those code are developed with comment.

Raw developing notebooks are in the notebook folder. Training scripts are in the script folder. Those code are not with comment.

./
    -3dtestoutput: 3d output of validation data 
        -cases: folders are named with the 'name' in labeling.csv
            -grount_truth.nii.gz: grount truth
            -gnn_unet_pass.nii.gz: pass connect gnn unet output
            -unet.nii.gz: unet output
            -gnn_unet_linear.nii.gz: linear connect gnn unet output
            
    -data: the raw data folder (now it is empty)
        -scans: download from https://www.dropbox.com/s/9xi6cy7v19mtdgi/scans.zip?dl=0
        -consensusAnnotations, 
        -fullAnnotations, 
        -referenceAnnotations: download from https://www.dropbox.com/sh/72a6ezpgdvox1aq/AAB4rvKS7CbylXXPFZsQX73La?dl=0
        
    -model_weights: trained model weights;file with  '_weight' in name is the params_dict file. Otherwise, the model files
    
    -runs: tensorboard runs folder. To see result, shell type "tensorboard --logdir runs"
    
    -train: trianing data. Now empty, if download data, use create_dataset.py to create train data
    
    -val: validation data. Now empty, if download data, use create_dataset.py to create val data
    
    -src: All modules and functions used in this projects
        -data.py: return the pytorch train and val dataset
        
        -generate_3d.py: contains functions used for 3d prediction
        
        -Graph_GNN.py: contains graph_gnn_linear and graph_gnn_pass models
        
        -test_models.py: contains functions used to get statitsics for models
        
        -unet_gnn_trainer.py: as title (linear)
        
        -unet_gnn_trainer_pass.py: as title
        
        -unet_trainer.py: as title
        
        -utils.py: contains functions to calculate metrics, also get the edges information
   
    -notebooks: NOTICE!!!! notebooks are developed in the ./ local. If you wanna succesful run the code, move the notebook you wanna run out of this foloder
    
        -compare.ipynb: compare each model performance

        -examing_model.ipynb: Predict who validation data and store stat csv files for each model

        -prepare_data: eda

        -save_3d_test_output.ipynb: use this to save whole 3d output

        -saving_model_weights.ipynb: use this to transform model pt to weights pt and save

        -unet-gnn-linear.ipynb: Unet-gnn linear model develop notebook

        -unet-gnn-pure.ipynb: Unet-gnn pass model develop notebook

        -unet.ipynb: unet model develop notebook
        
    -scripts: NOTICE!!!! scripts are developed in the ./ local. If you wanna succesful run the code, move the script you wanna run out of this foloder
    
        -create_dataset.py: script for creating train and val 

        -examing_model.py: script version of examing_model.ipynb

        -run_unet.py: train the unet

        -save_3d_test_output.py: save 3d test output

        -unet_sageconv_linear.py: train the unet-gnn-linear model

        -unet_sageconv_pass.py: train the unet-gnn-pass model
    
    -unet_stat.csv: unet model stats file
    
    -gnn_linear_stat.csv: unet-gnn linear model's stats

    -gnn_pass_stat.csv: unet-gnn pass model's stats
    
    -labeling.csv: for creating the data

