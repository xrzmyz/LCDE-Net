# LCDE-Net
An end-to-end point cloud semantic segmentation network.  
The network consists of two modules, one is a locally enhanced global feature extraction module, which mainly extracts local features from the input point cloud data. On the basis of local feature extraction, a global feature extraction module is used to extract global features from point clouds with local information, so as to obtain point cloud features with both local and global information; Another module is the feature channel modulation module, which adaptively eliminates redundant noise information generated during feature dimensionality enhancement. To avoid local information loss caused by the global information extraction process, the extracted features are fused with the original local features again, so that the local features are fully fused with the global features. The network adopts a traditional U-shaped structure for encoding and decoder construction, during which four downsampling operations are performed to extract richer semantic information through local and global feature fusion at different scales.
# Usage
* Download the file directly and extract it;
* Installation instructions:
  * Windows 10
  * Make sure CUDA and cuDNN are installed. One configuration has been tested:
    * `PyTorch 1.8.1`, `CUDA 11.1` and `cuDNN 8.4.1`
  * Install the other dependencies with pip:
    * numpy
    * scikit-learn
  * Compile the C++ extension modules for python located in cpp_wrappers. You just have to execute two .bat files:
    * ```cpp_wrappers/cpp_neighbors/build.bat```
    * ```cpp_wrappers/cpp_subsampling/build.bat```
   
Then, you can do the following to train the dataset.  
* Create a folder `Data\S3DIS` or `Data\Semantic3D`, convert the point cloud file into a ply file, for example, `area1. ply`, place the point cloud file in the corresponding folder, and then run `train_S3DIS.py` or `train_Semantic3D.py` from the two main folders to train;  
* During testing, run the `test_models.py` file in the corresponding folder to obtain the test results;  
* If you want to train other datasets, you can refer to the `S3DIS.py` or `Semantic3D.py` file for corresponding modifications.
