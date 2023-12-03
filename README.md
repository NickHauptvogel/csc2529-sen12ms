# Final Project CSC2529 - Fall term 2023
Nick Hauptvogel (1010801624)

This is the code for the final project for CSC2529. The code is based on the code from the paper 
> Schmitt M, Hughes LH, Qiu C, Zhu XX (2019) SEN12MS - a curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. In: ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences IV-2/W7: 153-160

for which the original README (repository_info.md) and the license is provided. The repository URL is https://github.com/schmitt-muc/SEN12MS.

To replicate the project results, please follow the instructions below.

1. Download and unzip only the land cover maps (suffix "_lc") from the SEN12MS dataset. The data can be downloaded from https://mediatum.ub.tum.de/1474004.
2. Download and unzip the Sentinel-1, Sentinel-2 and S2 cloudy images from the SEN12MS-CR dataset from https://mediatum.ub.tum.de/1554803.
3. After unzipping, the data should look like
```
.
├── data
│   ├── ROIs1158_spring_lc
│   │   ├── lc_1
│   │   │   ├── ROIs1158_spring_lc_1_p30.tif
│   ├── ROIs1158_spring_s1
│   │   ├── s1_1
│   │   │   ├── ROIs1158_spring_s1_1_p30.tif
│   ├── ROIs1158_spring_s2
│   │   ├── s2_1
│   │   │   ├── ROIs1158_spring_s2_1_p30.tif
│   ├── ROIs1158_spring_s2_cloudy
│   │   ├── s2_1
│   │   │   ├── ROIs1158_spring_s2_1_p30.tif
```

4. Create the cloud-removed patches with UnCRtainTS as found in https://github.com/PatrickTUM/UnCRtainTS. This is done via the Jupyter Notebook in UnCRtainTS/ModelLoader.ipynb. As the data structure is assumed differently, minor adaptations to the repository were made. You will find the files in the folder UnCRtainTS and can overwrite the specific sections in the cloned repository (this is only concercing data tree structure and using .tif images to store the predictions, so that they equal the original cloud-free structure in subsequent use). The result of this step should be cloud removed S2 patches in the UnCRtainTS folder
5. Run the model training of the ResNet classifier. This can be done using the Jupyter Notebook in classification/ClassificationTrainTest.ipynb. In this notebook, all test scores are also created (for cloud-free, cloud-removed and cloudy data)
6. Finally, to run the analysis on the scores, execute the file classification/analysis.py. You will need a virtual environment with the dependencies in requirements.txt
7. To run the GradCAM feature attribution, execute the file classification/run_and_plot_grad_cam.py, which was taken and adapted from https://github.com/JakobCode/explaining_cloud_effects/. For this, you'll need a second virtual environment with the dependencies listed in requirements_gradcam.txt
8. Using the available test_scores from training, executing the analysis steps will work out of the box.
