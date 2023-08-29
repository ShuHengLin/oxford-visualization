# Visualization Oxford Radar RobotCar Dataset & MVDNet label

## Prepare Data
1) Download the [Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-10-11-46-21-radar-oxford-10k).
   * Currently, only the vehicles in the first data record (Date: 10/01/2019, Time: 11:46:21 GMT) are labeled.
2) Clone this repository.
3) Clone the [MVDNet](https://github.com/qiank10/MVDNet) repository.
4) Clone the [RobotCar Dataset SDK](https://github.com/ori-mrg/robotcar-dataset-sdk) repository.
5) After unzipping the files, the directory should look like this:
```
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

* Prepare the processed radar & lidar data using MVDNet:
```
cd /DATA_PATH/MVDNet
```
```
python -B data/sdk/prepare_radar_data.py \
--data_path /DATA_PATH/2019-01-10-11-46-21-radar-oxford-10k --image_size 320 --resolution 0.2
```
```
python -B data/sdk/prepare_lidar_data.py \
--data_path /DATA_PATH/2019-01-10-11-46-21-radar-oxford-10k
```
Files will be generated in the **/DATA_PATH/2019-01-10-11-46-21-radar-oxford-10k/processed** folder.
  
* Calculate and generate lidar → radar transform:
```
cd /DATA_PATH/oxford-visualization
```
```
python -B caculate_transform.py
```
8,866 `.txt` files will be generated in the **/DATA_PATH/oxford-visualization/calib** folder.


## Visualize lidar pointcloud
* Using rviz to visualize:
```
roscore
```
```
rosrun rviz rviz -d rviz_config.rviz
```
```
python -B vis_lidar.py
```
```
python -B vis_lidar_processed.py
```
The first code will visualize the raw point cloud file **velodyne_right** and provide options to:
1) Project bounding boxes onto the lidar frame.
2) Project the lidar pointcloud onto the radar frame.

The second code will visualize the processed point cloud file **processed/lidar** and the bounding boxes.


## Visualize camera image
```
python -B vis_camera.py
```
```
python -B vis_camera_lidar.py
```
The first code will perform **radar → camera** projection.  
The second code will perform **radar → lidar → camera** projection.


## Visualize radar image
```
python -B vis_radar.py
```
Will visualize radar image and the 2d bounding boxes.


## Visualize all
* Using rviz to visualize:
```
roscore
```
```
rosrun rviz rviz -d rviz_config.rviz
```
```
python -B vis_all.py
```


## Video
[![](https://img.youtube.com/vi/wvTzqsMHO6o/0.jpg)](https://youtu.be/wvTzqsMHO6o)


## References
1) [The Oxford Radar RobotCar Dataset: A Radar Extension to the Oxford RobotCar Dataset](https://arxiv.org/abs/1909.01300)
2) [Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_Robust_Multimodal_Vehicle_Detection_in_Foggy_Weather_Using_Complementary_Lidar_CVPR_2021_paper.pdf)
3) [Spatial Transformation Matrices](https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html)
