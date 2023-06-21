# Existing Implementation  
-----------

# Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

https://github.com/xingyizhou/pytorch-pose-hg-3d

All the dependencies and other datasets are same as stated in the above link ( the main repository)
This repository is a fork of [pytorch implementation](https://github.com/xingyizhou/pytorch-pose-hg-3d) 


This repository is the PyTorch implementation for the network presented in:
> Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei, 
> **Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach**
> ICCV 2017 ([arXiv:1704.02447](https://arxiv.org/abs/1704.02447))
<p align="center"> 
<img src="teaser.png" width="350"/>
</p>
**<span style="color:red">Note: </span>** This repository has been updated and is different from the method discribed in the paper. To fully reproduce the results in the paper, please checkout the original [torch implementation](https://github.com/xingyizhou/pose-hg-3d) or our [pytorch re-implementation branch](https://github.com/xingyizhou/pytorch-pose-hg-3d/tree/hg3d) (slightly worse than torch). 
We also provide a clean [2D hourglass network branch](https://github.com/xingyizhou/pytorch-pose-hg-3d/tree/2D).
The updates include:
- Change network backbone to ResNet50 with deconvolution layers ([Xiao et al. ECCV2018](https://github.com/Microsoft/human-pose-estimation.pytorch)). Training is now about 3x faster than the original hourglass net backbone (but no significant performance improvement). 
- Change the depth regression sub-network to a one-layer depth map (described in our [StarMap project](https://github.com/xingyizhou/StarMap)).
- Change the Human3.6M dataset to official release in [ECCV18 challenge](http://vision.imar.ro/human3.6m/challenge_open.php). 
- Update from python 2.7 and pytorch 0.1.12 to python 3.6 and pytorch 0.4.1.
# Integrating the 3DPW Dataset
______________________________

## Downloading data(3DPW dataset):
```
https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html
```

## Structure of 3dpw dataset in the data folder
```
 |------3dpw
        |------imageFiles
               |-----courtyard_arguing_00
               |     |-----image_0000_1.jpg, image_0000_1.jpg, etc.
               |-----courtyard_backpacking_00
               |     |-----image_0000_1.jpg, image_0000_1.jpg, etc.
               |... etc
        |------sequenceFiles
               |-----train
               |     |-----downtown_arguing_00.pkl, downtown_cafe_00.pkl, etc.
               |-----test
               |     |-----downtown_arguing_00.pkl, downtown_cafe_00.pkl, etc.
               |-----validation
               |     |-----downtown_arguing_00.pkl, downtown_cafe_00.pkl, etc.
```
## DEMO of existing model
   ```
   python demo.py --demo ../images --gpus -1 --load_model ../models/fusion_3d_var.pth
   ```

## Changes:
1. Added threedpw.py script in src/lib/datasets
2. Added the dataset in main.py, opts.py
3. Train the model:
   Change the current directory to src folder
   ```
   python main.py --exp_id threedpw
   ```

4. Run DEMO for 3dpw dataset
   ```
   python demo.py --demo ../data/3dpw/imageFiles --dataset threedpw --gpus -1 --load_model ../models/fusion_3d_var.pth
   ```

## Result Comparison

#### Existing model i.e. mpii and h36m datasets
<!images/ExistingResult.png>

#### 3DPW dataset
<!images/Newresult.png>

## Conclusion
The paper effectively uses convolutional neural networks for pose estimation. The inclusion of the 3DPW dataset not only allows the model to predict poses on a new dataset but also potentially improves the generalizability of the model. The results after integrating the 3DPW dataset were not initially accurate, which may be due to the necessity for further training, optimization, or potential issues with the dataset integration. Nonetheless, with a few minor changes and potential future improvements, the project demonstrates the application of deep learning for 2D and 3D pose estimation tasks.

THANKYOU! 

## Citation

    @InProceedings{Zhou_2017_ICCV,
    author = {Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen},
    title = {Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }

    3dpw dataset: https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html
    
    @inproceedings{vonMarcard2018,
    title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
    author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018},
    month = {sep}
    }
