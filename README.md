# LCDE-Net
An end-to-end point cloud semantic segmentation network
The network consists of two modules, one is a locally enhanced global feature extraction module, which mainly extracts local features from the input point cloud data. On the basis of local feature extraction, a global feature extraction module is used to extract global features from point clouds with local information, so as to obtain point cloud features with both local and global information; Another module is the feature channel modulation module, which adaptively eliminates redundant noise information generated during feature dimensionality enhancement. To avoid local information loss caused by the global information extraction process, the extracted features are fused with the original local features again, so that the local features are fully fused with the global features. The network adopts a traditional U-shaped structure for encoding and decoder construction, during which four downsampling operations are performed to extract richer semantic information through local and global feature fusion at different scales.
