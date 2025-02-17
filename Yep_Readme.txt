utility.h
    extern const string pointCloudTopic; extern关键字表示声明一个变量，其真正的定义应该是在某个.cpp文件中
    extern使得这个变量可以在多个源文件间共享
    其他源文件通过包含这个头文件就能访问到这个变量