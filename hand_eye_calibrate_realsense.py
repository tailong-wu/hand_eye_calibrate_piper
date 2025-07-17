# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os

import cv2
import numpy as np

import json

np.set_printoptions(precision=8, suppress=True)

images_path = "./collect_data_piper"  # 手眼标定采集的标定版图片所在路径
arm_pose_file = "./collect_data_piper/poses.txt"  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应
R_co = np.array([
    [0, 0, 1],   # 第一列：新 x 轴 [-e_y] = [0, -1, 0]
    [-1, 0, 0],  # 第二列：新 y 轴 [-e_z] = [0, 0, -1]
    [0, -1, 0]   # 第三列：新 z 轴 [e_x] = [1, 0, 0]
], dtype=float)


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    return R, t


def camera_calibrate(images_path):
    print("++++++++++开始相机标定++++++++++++++")
    # 角点的个数以及棋盘格间距
    XX = 10  # 标定板的中长度对应的角点的个数
    YY = 7  # 标定板的中宽度对应的角点的个数
    L = 0.015  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    valid_indices = []  # 存储成功检测到棋盘格的图片索引
    size = None  # 初始化 size 变量

    for i in range(0, 20):  # 标定好的图片在iamges_path路径下，从0.jpg到x.jpg   一般采集20张左右就够，实际情况可修改

        image = f"{images_path}/images{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):

            img = cv2.imread(image)
            print(f"图像大小： {img.shape}")
            # h_init, width_init = img.shape[:2]
            # img = cv2.resize(src=img, dsize=(width_init // 2, h_init // 2))
            # print(f"图像大小(resize)： {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            
            if ret:
                # print(corners)
                print(f"左上角点：{corners[0, 0]}")
                print(f"右下角点：{corners[-1, -1]}")

                # 绘制角点并显示图像
                cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
                # cv2.imshow('Chessboard', img)  # 注释掉显示功能，避免在无头环境中出错

                # cv2.waitKey(3000)  ## 停留1s, 观察找到的角点是否正确

                obj_points.append(objp)
                valid_indices.append(i)  # 记录成功检测到棋盘格的图片索引

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
            else:
                print(f"未能在图片 {image} 中找到棋盘格角点")

    N = len(img_points)
    
    if size is None:
        raise ValueError("未找到有效的标定图片，无法进行相机标定")

    # 标定得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # print("ret:", ret)
    print("内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs, valid_indices


def process_arm_pose(arm_pose_file, valid_indices=None):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    
    if valid_indices is None:
        # 如果没有指定有效索引，处理所有行
        for line in all_lines:
            pose = [float(v) for v in line.split(',')]
            R, t = pose_to_homogeneous_matrix(pose=pose)
            R_arm.append(R)
            t_arm.append(t)
    else:
        # 只处理有效索引对应的行
        for i in valid_indices:
            if i < len(all_lines):
                line = all_lines[i]
                pose = [float(v) for v in line.split(',')]
                R, t = pose_to_homogeneous_matrix(pose=pose)
                R_arm.append(R)
                t_arm.append(t)
    return R_arm, t_arm


def hand_eye_calibrate():
    rvecs, tvecs, valid_indices = camera_calibrate(images_path=images_path)
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file, valid_indices=valid_indices)
    
    print(f"成功处理了 {len(valid_indices)} 张图片")
    print(f"有效图片索引: {valid_indices}")

    R, t = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    R = np.dot(R_co,R)
    print("+++++++++++手眼标定完成+++++++++++++++")
    return R, t

def convert_result_to_json(R, t, save_path=None):
    """将旋转矩阵和平移向量转换为指定格式的json"""
    
    # 将numpy数组转换为嵌套列表
    rotation_matrix = R.tolist()
    translation_vector = t.tolist()
    
    result = {
        "rotation_matrix": rotation_matrix,
        "translation_vector": translation_vector
    }
    
    json_str = json.dumps(result, indent=4)
    
    # 如果指定了保存路径，则保存到文件
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"结果已保存到: {save_path}")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
    
    return json_str

if __name__ == "__main__":
    R, t = hand_eye_calibrate()

    print("旋转矩阵：")
    print(R)
    print("平移向量：")
    print(t)
    
    # 转换并打印json格式的结果
    json_result = convert_result_to_json(R, t, save_path="./hand_eye_result_realsense.json")
    print("\nJSON格式的结果：")
    print(json_result)
