lidar_processor allows loading upsampling and viewing of lidar data, It is specifically built for scans where both Lidar and XRF data is acquired
some files for transformations and trimming may be assumed otherwise, the default trimming parameters are also quite narrow, if you would like to view a full cloud 
change the window_size parameter in construction to 1600, this may overload memory for a number of functions, depending on availably VRAM/Shared GPU memory but simply viewing the point cloud is fine.

only tested on python 3.10.11 
