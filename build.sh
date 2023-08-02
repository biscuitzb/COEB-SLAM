cd build
###
 # @Author: biscuitzb 903428724@qq.com
 # @Date: 2022-08-13 16:55:01
 # @LastEditors: biscuitzb 903428724@qq.com
 # @LastEditTime: 2023-02-15 21:27:12
 # @FilePath: /ORB_SLAM2/build.sh
 # @Description: 
 # 
 # Copyright (c) 2022 by biscuitzb 903428724@qq.com, All Rights Reserved. 
### 
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
