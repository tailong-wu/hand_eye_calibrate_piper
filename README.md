# hand_eye_calibrate_piper

该项目可以进行PiPER机器人的手眼标定（Eye in Hand）。 


手眼标定的原理可参考：[机械臂手眼标定方法详解](https://blog.csdn.net/leo0308/article/details/141498200)


data_collect.py 为数据采集脚本， 可直接运行。 注意需要将机械臂位姿获取根据自己实际使用的机械臂进行修改。 


hand_eye_calibrate.py 为计算程序， 可根据采集的数据， 计算得到手眼转换关系矩阵，并将结果保存为json文件。（普通相机坐标系）

hand_eye_calibrate_realsense.py 为计算程序， 可根据采集的数据， 计算得到手眼转换关系矩阵, 并将结果保存为json文件。 （realsense相机坐标系）

interactive_visualization.py 为可视化程序， 可根据采集的数据， 可视化手眼转换关系。

collect_data目录为示例数据， 可直接运行计算程序理解代码逻辑。 

collect_data_piper目录为PiPER机器人采集的数据， 可直接运行计算程序理解代码逻辑。 