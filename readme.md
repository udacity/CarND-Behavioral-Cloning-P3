# 注意事项
环境安装
- ffmepg：不能直接使用apt-get 或者pip直接安装
  - 原因：会出现找不到libx264的错误
  - 解决方法: conda install -c conda-forge ffmpeg

- socketio:当提醒缺少这个库的时候，不能直接使用pip安装
  - 原因：直接安装会强制降级setuptools的版本，造成环境的 tensorflow导入失败
  - 解决方法: 使用pip安装 flask-socketio


[参考](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/environment-gpu.yml)

修改如下dependencies:
    - numpy
    - matplotlib
    - jupyter
    - pillow
    - scikit-learn
    - scikit-image
    - scipy
    - h5py
    - eventlet
    - flask-socketio
    - seaborn
    - pandas
    - imageio
    - pyqt
    - pip:
        - moviepy
        - opencv-python
        - requests
