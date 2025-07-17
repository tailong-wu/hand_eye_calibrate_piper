"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例， 其他相机数据读取可能需要对应修改。 """

import cv2
import numpy as np
import pyrealsense2 as rs
from piper_sdk import C_PiperInterface_V2

count = 0

# 初始化相机
pipeline = rs.pipeline()
config = rs.config()
# 指定相机序列号
config.enable_device('243222071317')
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 初始化piper机械臂
try:
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    print("已连接到piper机械臂")
except Exception as e:
    print(f"连接piper机械臂失败: {e}")
    print("请检查机械臂连接和CAN接口")
    exit(1)

image_save_path = "./collect_data/"

# 确保保存目录存在
import os
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)
    print(f"创建保存目录: {image_save_path}")


def data_collect():
    global count
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                               cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

            k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            if k == ord('s'):  # 键盘按一下s, 保存当前照片和机械臂位姿
                try:
                    print(f"采集第{count}组数据...")
                    # 获取当前机械臂位姿
                    arm_end_pose_msgs = piper.GetArmEndPoseMsgs()
                    pose = [
                        arm_end_pose_msgs.end_pose.X_axis / 1000000.0,  # 从0.001mm转换为米
                        arm_end_pose_msgs.end_pose.Y_axis / 1000000.0,
                        arm_end_pose_msgs.end_pose.Z_axis / 1000000.0,
                        arm_end_pose_msgs.end_pose.RX_axis / 1000.0,     # 从0.001度转换为度
                        arm_end_pose_msgs.end_pose.RY_axis / 1000.0,
                        arm_end_pose_msgs.end_pose.RZ_axis / 1000.0
                    ]
                    print(f"机械臂pose:{pose}")

                    with open(f'{image_save_path}poses.txt', 'a+') as f:
                        # 将列表中的元素用空格连接成一行
                        pose_ = [str(i) for i in pose]
                        new_line = f'{",".join(pose_)}\n'
                        # 将新行附加到文件的末尾
                        f.write(new_line)

                    cv2.imwrite(image_save_path + f'images{count}.jpg', color_image)
                    count += 1
                    print(f"数据保存成功，当前已采集{count}组数据")
                except Exception as e:
                    print(f"获取机械臂位姿失败: {e}")
            elif k == ord('q') or k == 27:  # 按q或ESC退出
                print("退出数据采集")
                break
    except KeyboardInterrupt:
        print("\n用户中断采集")
    except Exception as e:
        print(f"数据采集过程中发生错误: {e}")
    finally:
        # 清理资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("资源清理完成")


if __name__ == "__main__":
    print("=== Piper机械臂手眼标定数据采集程序 ===")
    print("操作说明:")
    print("- 按 's' 键保存当前图像和机械臂位姿")
    print("- 按 'q' 键或 ESC 键退出程序")
    print("- 按 Ctrl+C 强制退出")
    print("请确保机械臂已正确连接并处于安全位置")
    print("开始数据采集...\n")
    
    data_collect()
