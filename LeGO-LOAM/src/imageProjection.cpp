// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"

class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;     //* 订阅原始点云
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;  

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; //* 距离图像         
    cv::Mat labelMat; //* 分割标签矩阵
    cv::Mat groundMat; //* 地图标记矩阵
    int labelCount;

    float startOrientation;  //* 当前帧点云扫描的起始角度
    float endOrientation;    //* 当前帧点云扫描的结束角度

    // cloud_msgs::cloud_info消息类型
    // # Header 信息
    // std_msgs/Header header

    // # 起始和结束方位角
    // float32 startOrientation        # 点云扫描的起始角度
    // float32 endOrientation         # 点云扫描的结束角度
    // float32 orientationDiff        # 角度差值

    // # 每个扫描线的起始和结束索引
    // int32[] startRingIndex         # 每条扫描线的起始点索引
    // int32[] endRingIndex          # 每条扫描线的结束点索引

    // # 分割点云的信息
    // bool[]    segmentedCloudGroundFlag    # 标记点是否为地面点
    // int32[]   segmentedCloudColInd        # 点的列索引
    // float32[] segmentedCloudRange         # 点到激光雷达的距离
    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection(): 
        //* nh("~") 表示使用私有的命名空间
        nh("~"){
        
        //* 队列大小为1 只保留最新一条消息   回调函数：&ImageProjection::cloudHandler：指向类成员函数的指针     this:this 指向当前类实例
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);   

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        //* 点云初始化 quiet_NaN()是不会触发浮点异常的NaN值  intensity = -1 表示无效点
        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());       //输入点云
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());  //带环号信息的点云

        //* 完整点云
        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        //* 带地面信息的点云  地面点，分割后点云 纯分割点云 离群点云
        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        //* 设置完整点云和信息点云大小 N_SCAN是线数  Horizon_SCAN是一线有多少点
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        // 初始化起始和结束索引数组
        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        // 初始化分割标志和信息数组
        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        // 定义四个搜索方向：
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        // 分配用于跟踪分割点的数组
        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        // 分配用于BFS搜索的队列数组
        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        // 距离矩阵：初始化为最大浮点数
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        // 地面标记矩阵：初始化为0
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        // 标签矩阵: 初始化为0
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        // 使用nanPoint填充fullInfoCloud
        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);

        // indices用来存储有效点的索引
        std::vector<int> indices;
        // 函数原型：
        //* template<typename PointT> void removeNaNFromPointCloud(
        //*     const pcl::PointCloud<PointT> &cloud_in,  // 输入点云
        //*     pcl::PointCloud<PointT> &cloud_out,       // 输出点云
        //*     std::vector<int> &index                   // 有效点的索引
        //* )
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);


        //* 转换出带行号的点云
        if (useCloudRing == true){
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false) {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }  
        }
        //! 同一个输入源sensor_msgs::PointCloud2ConstPtr& laserCloudMsg，为什么既能够转换laserCloudIn类型 又能够转换为laserCloudInRing类型，且laserCloudInRing还是utility自定义的一个类型
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        //* ROS点云转化成pcl格式的点云，*laserCloudIn,*laserCloudInRing
        copyPointCloud(laserCloudMsg);

        //* 查询第一个点和最后一个点的角度差
        findStartEndAngle();

        //* 点云到图像的投影转换(计算距离图)，同时填充fullCloud和fullInfoCloud信息
        projectPointCloud();

        //* 标记地面点云
        groundRemoval();

        //* 点云分割
        cloudSegmentation();
        // 6. Publish all clouds
        publishCloud();
        // 7. Reset parameters for next iteration 下一帧数据到来之前重置变量
        resetParameters();
    }

    void findStartEndAngle(){
        //* 计算点云第一个点和最后一个点的方位角
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;

        //* 处理角度跨越情况，如果角度差大于3Π，减去2Π；如果小于Π，加上2Π
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        
        //* 计算总的角度差
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    void projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;           //* 垂直角，水平角，距离
        size_t rowIdn, columnIdn, index, cloudSize;         //* 行号，列号，索引，点云大小
        PointType thisPoint;                                //* 当前处理的点

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // find the row and column index in the iamge for this point
            if (useCloudRing == true){
                rowIdn = laserCloudInRing->points[i].ring;    //* 直接使用点云中的ring来作为图像的行号
            }
            else{
                //* 计算俯仰角 之后计算行索引   这样来看行号的0应该是在最下面
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            if (rowIdn < 0 || rowIdn >= N_SCAN)   //* 对于行号不在范围内的点不进行处理
                continue;

            //? 注意这里使用的是 x/y 那估计这里的雷达向前表示x方向  即 向上为x正方向，向左为y正方向  有待考证
            //* 将弧度转化为角度 角度范围是(-180°，180°]
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            //* 角度偏移90°，并且将索引调整到图像中心，我的理解 此时对于一个x轴朝右的正方向坐标系来说 正中心就是x轴
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
            //* 处理列索引越界
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            //* 提出距离过近的点
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;
            
            //* 存储距离值到距离矩阵
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            //* 设置点的强度(编码行列信息)
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            //* 计算点云的一维索引
            index = columnIdn  + rowIdn * Horizon_SCAN;

            //* 存储点云信息(使用距离作为强度值)
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
        }
    }


    void groundRemoval(){
        size_t lowerInd, upperInd;          // 上下点索引
        float diffX, diffY, diffZ, angle;   // 差值和角度 
        // groundMat                                                 groundMat三种转台
        // -1, no valid info to check if ground of not               -1：无效点，无法判断是否为地面点
        //  0, initial value, after validation, means not ground      0：初始值，同时是经过验证后表示为非地面点
        //  1, ground                                                 1：地面点

        //* 主要分析相邻扫描线之间的几何关系，来实现地面点检测
        for (size_t j = 0; j < Horizon_SCAN; ++j){                //* 遍历每一列
            for (size_t i = 0; i < groundScanInd; ++i){           //* 遍历底部扫描线

                lowerInd = j + ( i )*Horizon_SCAN;    //* 当前扫描线上的点   Horizon_SCAN=1800
                upperInd = j + (i+1)*Horizon_SCAN;    //* 上一扫描线上的点

                //* 两点中如果任何一个点被表示为无效点，则无法判断地面特征    注意初始值为0
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                //* 计算差值向量与水平方向的夹角 
                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                //* 如果两个点均为地面点，则两个点的向量组成的连线应该基本与水平面平行，即接近0°，sensorMountAngle应该是雷达安装的俯仰角
                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;                //* 将两个点都设置为平面点
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        //* 标记后续不需要参与后续分割处理的点，主要包括地面点和无效点(即距离仍然为初始值浮点数最大值)
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;    //* 标签设置为无效点
                }
            }
        }

        //*if (pubGroundCloud.getNumSubscribers() != 0) 检查是否有节点订阅地面点云，属于一个优化处理，只有在订阅者订阅时，才执行点云提取
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)      //* 如果是地面点
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);    //* 加入地面点云groundCloud
                }
            }
        }
    }

    //* 点云分割处理函数
    void cloudSegmentation(){
        // segmentation process
        //* 对所有未标记的点进行分割处理
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)           
                    labelComponents(i, j);    //* 点云区域生长分割

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {
            //* 记录每条扫描线起始索引（偏移5个点）
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                //* 处理有效分割点或地面点
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    //* 对于离群点
                    if (labelMat.at<int>(i,j) == 999999){
                        //* 非地面区域 每5个点保留一个 放入离群点中
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    //* 地面点抽稀 ，每五个点保留一个
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }
                    // mark ground points so they will not be considered as edge features later
                    //* 标记是否为地面点
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    //* 列索引
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    //* 距离值
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    //* 保存到点云分割
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    //* 更新分割点云的计数
                    ++sizeOfSegCloud;
                }
            }

            //* 记录每条扫描线结束索引（偏移5个点）
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        //* 提取可视化点云
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){         //* labelMat记录了第几个分割类的标签
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);   //* segmentedCloudPure的YXZ记录点的坐标
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);    //* segmentedCloudPure的intensity记录点的类别
                    }
                }
            }
        }
    }

    //* 区域生长分割算法，主要通过BFS(广度优先搜索)对点云进行分割
    void labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;                      //* 用于集合特征计算
        int fromIndX, fromIndY, thisIndX, thisIndY;      //* 索引变量
        bool lineCountFlag[N_SCAN] = {false};            //* 记录每条扫描线是否被访问

        //* 初始化队列  (使用数组实现队列以提高效率)
        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        //* 记录所有访问过的点
        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;

            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            //* 遍历邻域点
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // 在距离图中他的邻域点
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                //* 在Y方向就越界丢弃
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                //* 在x方向 就循环处理 ，因为距离图是一个环
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                //* 避免重复处理
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                //* 取当前邻域点和中心点的距离较大值和较小值
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                //* 选择角分辨率
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;   //* 邻域点左右偏移 即在距离图中的y轴方向的偏移，也就是水平方向的分辨率
                else
                    alpha = segmentAlphaY;   //* 同理可得
                //! 平滑度计算的物理具象是怎么样的 angle反应两点之间的平滑程度
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){         //* 平滑度满足要求
                    //* 加入队列
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;
                    
                    //* 标记点
                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;
                    //* 记录访问过的点
                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        //* 分割有效性验证
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)     //* 点数足够多，大物体分割
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){        //* 点数大于阈值 计算下分割跨越了多少条扫描线
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        //* 分割有效，则郑家标签类别数目
        if (feasibleSegment == true){
            ++labelCount;
        //* 无效分割标记为离群点 999999
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        //* pubOutlierCloud离群点  pubSegmentedCloud带少数地面的分割点云  pubFullCloud完整点云(有订阅者)  pubGroundCloud地面点云(有订阅者)
        //* pubSegmentedCloudPure纯分割聚类点云  含有完整信息的点云
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
