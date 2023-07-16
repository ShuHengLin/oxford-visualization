# visualization_Oxford_Radar_RobotCar_Dataset

## Prepare Data
1) Download the [Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-10-11-46-21-radar-oxford-10k).
   * Currently, only the vehicles in the first data record (Date: 10/01/2019, Time: 11:46:21 GMT) are labeled.
2) Clone this repository.
3) Clone the [MVDNet](https://github.com/qiank10/MVDNet) repository.
4) Clone the [RobotCar Dataset SDK](https://github.com/ori-mrg/robotcar-dataset-sdk) repository.
5) After unzipping the files, the directory should look like this:
```
# Oxford Radar RobotCar Data Record
|-- DATA_PATH
    |-- MVDNet
    |-- oxford-visualization
    |-- robotcar-dataset-sdk
    |-- 2019-01-10-11-46-21-radar-oxford-10k
        |-- gt
        |-- radar
        |-- stereo
        |-- velodyne_left
        |-- velodyne_right
        |-- vo
        |-- radar.timestamps
        |-- stereo.timestamps
        |-- velodyne_left.timestamps
        |-- velodyne_right.timestamps
        |-- ...
```


## Usage
* AAA
```
```


## References
1) [The Oxford Radar RobotCar Dataset: A Radar Extension to the Oxford RobotCar Dataset](https://arxiv.org/abs/1909.01300)
2) [Github - RobotCar Dataset SDK](https://github.com/ori-mrg/robotcar-dataset-sdk)
3) [Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_Robust_Multimodal_Vehicle_Detection_in_Foggy_Weather_Using_Complementary_Lidar_CVPR_2021_paper.pdf)
4) [Github - MVDNet](https://github.com/qiank10/MVDNet)
