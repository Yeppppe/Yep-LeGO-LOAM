utility.h
    extern const string pointCloudTopic; extern关键字表示声明一个变量，其真正的定义应该是在某个.cpp文件中
    extern使得这个变量可以在多个源文件间共享
    其他源文件通过包含这个头文件就能访问到这个变量

    cloud_msgs::cloud_info消息类型
    # Header 信息
    std_msgs/Header header
    # 起始和结束方位角
    float32 startOrientation        # 点云扫描的起始角度
    float32 endOrientation         # 点云扫描的结束角度
    float32 orientationDiff        # 角度差值
    # 每个扫描线的起始和结束索引
    int32[] startRingIndex         # 每条扫描线的起始点索引
    int32[] endRingIndex          # 每条扫描线的结束点索引
    # 分割点云的信息
    bool[]    segmentedCloudGroundFlag    # 标记点是否为地面点
    int32[]   segmentedCloudColInd        # 点的列索引
    float32[] segmentedCloudRange         # 点到激光雷达的距离