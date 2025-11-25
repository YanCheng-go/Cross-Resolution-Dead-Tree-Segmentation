# Cross-Scale-Dead-Tree-Segmentation

Cross-resolution segmentation of individual dead trees from aerial images

==============================

This framework integrates multiple pipelines, including training sample preprocessing, model training, prediction, and evaluation.

Project Organization
------------
 
    ├── deic_init.sh <- a bash script for initializing the environment on DEIC HPD
    ├── train <- training scripts
    │   ├── ordinal_watershed.py
    │   ├── treehealth_ordinal_watershed.py <- building blocks for the training script for dead tree segmentation
    │   ├── treehealth_ordinal_watershed_5c.py <- training script for dead tree segmentation
    │   ├── segmentation.py
    │   ├── watershed.py
    │   ├── base.py
    │   └── __init__.py
    ├── config <- configuration folder
    │   ├── treehealth.py <- configurations for tree mortality mapping
    │   ├── watershed.py <- configurations for watershed segmentation
    │   ├── treehealth_5c.py <- configurations for tree mortality mapping
    │   └── base.py <- base configurations for segmentation task
    ├── src <- source code
    │   ├── modelling
    │   │   ├── models <- model architectures
    │   │   │   ├── smp.py <- segmentation model
    │   │   │   ├── backbones_conf.py
    │   │   │   ├── modules
    │   │   │   ├── util.py
    │   │   │   ├── map_prediction.py 
    │   │   │   ├── __init__.py
    │   │   │   └── unet_with_scalar.py <- segmentation model with resolution information
    │   │   ├── metric.py <- metric functions
    │   │   ├── helper.py 
    │   │   ├── model_wrapper.py 
    │   │   ├── losses <- loss functions
    │   │   │   ├── focal.py 
    │   │   │   ├── tversky.py
    │   │   │   └── base.py
    │   │   └──  __init__.py
    │   ├── utils <- utility functions
    │   │   ├── data_utils.py
    │   │   ├── config_parser.py
    │   │   ├── __init__.py
    │   │   ├── jaccard.py
    │   │   └── log_metrics.py 
    │   ├── data <- data processing
    │   │   ├── collate.py
    │   │   ├── base_dataset.py 
    │   │   ├── image_matching.py <- match image patches
    │   │   ├── image_extractor.py <- extract image patches
    │   │   ├── image_table.py <- build image table
    │   │   ├── raster_prediction_writer.py <- write predictions to raster
    │   │   ├── base_segmentation_dataset.py 
    │   │   ├── preprocess_labels.py <- pre-process shared labels from others
    │   │   ├── watershed_dataset.py <- generate watershed energy layer from labels
    │   │   └── __init__.py
    │   ├── visualization <- visualization functions 
    │   │   ├── visualize.py
    │   │   └──  __init__.py
    │   └── __init__.py
    ├── unit_test
    │   ├── test_build_image_table.py
    │   ├── test_watershed.py
    │   ├── test_confusion_metric.py
    │   ├── test_segmentation.py
    │   ├── test_models.py
    │   ├── sample_labels
    │   │   ├── sample_tree_species.gpkg
    │   │   ├── training_areas_example.gpkg
    │   │   └── training_polygons_example.gpkg
    │   ├── test_size_weights.py
    │   ├── test_polygon_transformation.py
    │   ├── test_cutlines_scoring.py
    │   ├── test_patch_count_calculator.py
    │   └── test_ordinal_watershed.py
    ├── predict
    │   ├── ordinal_watershed.py 
    │   ├── treehealth_ordinal_watershed_5c.py  <- prediction script for tree mortality mapping
    │   ├── segmentation.py
    │   ├── watershed.py
    │   └── __init__.py
    ├── postprocessing
    │   ├── extract_mean_std_all.py  <- extract mean and std of all images
    │   ├── evaluation_plots.py  <- plot evaluation results
    │   ├── ordinal_watershed_evaluation.py <- evaluate the model performance
    │   ├── treehealth_owatershed_evaluation_5c.py  <- run file of the evaluation step
    │   ├── util.py  <- utility functions
    │   ├── organize_training_configs.py  <- summarize training and evaluation configurations
    │   ├── prep_ortho_germany.py  <- prepare ortho images for Germany
    │   ├── external_evaluation.py  <- evaluate the model performance with external datasets
    │   └── __init__.py
    ├── env_cuda12.yml <- conda environment file (compatible with cuda12)
    ├── test_environment.py
    ├── config.json <- example configuration file, change configuration here or in the treehealther_5c.py in the config folder
    └──README.md




Prerequisites
------------
To be able to run through this project, please make sure that you have a GPU, conda, and CUDA 12 installed.
This project has been tested on the GPU RTX3090, H100 on both the HPC and the local Linux system.


Set up the environment
------------
In the terminal, please run the following code and wait for about 5 minutes to install the environment.
```conda env create -f env_cuda12.yml --name ssl4eo_delfors```

Activate the environment
------------
In the terminal, please run the following code to activate the environment and change the directory to the project folder.
```
conda activate ssl4eo_delfors
cd Cross-Scale-Dead-Tree-Segmentation
```

Configure parameters
------------
In ./config/treehealth_5c.py, you can modify parameters such as number of epochs, patch size, batch size, learning rate, number of workers, and so on. The list of customizable parameters can be found in ./config/base.py. Alternatively, you can assign values to specific parameters using flags in the command in the terminal.
There is an example configuration file named config.json in the project folder. You can change the parameters in this file and use it in the command line.

Process training samples
------------
In the terminal, please run

```python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "process-data" --epochs 1 --config-file "./config.json" --processed-dir "./processed_dir/process_3bands" --data-dir "./training_dataset/5c_20240717" --allow_partial_patches True --patch-size 256 --log-dir <LOG DIR>```


Train models
------------
To train a model, please run the following command and change the parameters accordingly or go to ./config/treehealth_5c.py or config.json and update the parameters in the train class named TreeHealthSegmentationConfig.

```python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "train-model" --config-file "./config.json" --processed-dir <FOLDER TO SAVE PROCESSED TRAINING DATASETS> --epochs 500 --lr 3e-5 --log-dir <LOG DIR>```

To continue a paused training, please run the following command

```python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "train-model" --config-file "./config.json" --processed-dir <FOLDER TO SAVE PROCESSED TRAINING DATASETS> --epochs 500 --lr 3e-5 --reset-head False --load <CHANGE TO THE MODEL PATH> --log-dir <LOG DIR>```

Evaluate models and calculate AP metrics
------------
To Evaluate the model performance and calculate IoU, F1, (r)MAE, bias, and AP3070, please run the following command and change the parameters accordingly
    
```python -m evaluate.treehealth_ordinal_watershed_5c --load <CHANGE TO THE MODEL PATH> --config-file <MODEL CONFIG PATH>  --log-dir <LOG DIR>```

Prediction
--------
For predictions, please run the following command and change the parameters accordingly, or go to ./config/treehealth_5c.py and update the parameters in the prediction class named TreeHealthSegmentationPredictionConfig.

```python -m predict.treehealth_ordinal_watershed_5c --load <MODEL PATH> --config-file <MODEL CONFIG PATH> --out-prediction-folder <FOLDER TO SAVE PREDICTIONS> --image-src <IMAGE SOURCE PARAMETER> --log-dir <LOG DIR>```

Example of image source parameter, note the reference data name should stay the same as the config file of the trained model.
```
--image-srcs '{
    "germany20cm_2022": {
      "base_path": ".",
      "image_file_type": ".tif",
      "image_file_prefix": "",
      "image_file_postfix": "",
      "filelist_path": "./deadtrees_images_to_predict.txt"
    }
  }'
  ```

Evaluate models with external datasets
------------
Use the code in postprocessing/external_evaluation.py to evaluate the model performance with external datasets.

Pretrained models
------------
[Link](https://drive.google.com/drive/folders/1ycIdRRyQGY35-awr2pdAsKZAIALFJVju) to the pretrained model. Download the entire folder and change the <MODEL PATH> and <MODEL CONFIG PATH> in the command lines above for prediction or continuous training. 

Instructions on fine-tuning are on the way
------------
Please contact chengyan2017@gmail.com for urgent needs in such cases.

Citation
------------
Cheng, Y., & Oehmcke, S. (2025). YanCheng-go/Cross-Resolution-Dead-Tree-Segmentation: v0.0.1 (v0.0.1). Zenodo. https://doi.org/10.5281/zenodo.17234915
