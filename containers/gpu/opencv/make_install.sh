wget https://github.com/opencv/opencv/archive/3.3.1.tar.gz
tar -xvf 3.3.1.tar.gz
cd opencv-3.3.1
mkdir build
cd build
cmake -D CUDA_ARCH_BIN=3.7 \
    -D CUDA_ARCH_PTX="" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D WITH_V4L=ON \
    -D BUILD_TIFF=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON ..
make -j4
make install
sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
ldconfig
