# coding=utf-8
"""
手眼标定结果交互式可视化
使用plotly创建交互式3D可视化
"""

import numpy as np
import json
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True

def create_coordinate_frame(R, t, name, scale=0.03, colors=['red', 'green', 'blue']):
    """
    创建坐标系的3D箭头
    
    Args:
        R: 3x3旋转矩阵
        t: 3x1平移向量
        name: 坐标系名称
        scale: 坐标轴长度缩放因子
        colors: 坐标轴颜色 [X轴, Y轴, Z轴]
    
    Returns:
        traces: plotly图形对象列表
    """
    traces = []
    origin = t.flatten()
    
    # 坐标轴方向向量
    axes = [R[:, 0] * scale, R[:, 1] * scale, R[:, 2] * scale]
    axis_names = ['X', 'Y', 'Z']
    
    for i, (axis, axis_name, color) in enumerate(zip(axes, axis_names, colors)):
        # 箭头线段
        trace_line = go.Scatter3d(
            x=[origin[0], origin[0] + axis[0]],
            y=[origin[1], origin[1] + axis[1]],
            z=[origin[2], origin[2] + axis[2]],
            mode='lines',
            line=dict(color=color, width=8),
            name=f'{name}-{axis_name}',
            showlegend=True
        )
        traces.append(trace_line)
        
        # 箭头头部（使用cone）
        trace_cone = go.Cone(
            x=[origin[0] + axis[0]],
            y=[origin[1] + axis[1]],
            z=[origin[2] + axis[2]],
            u=[axis[0] * 0.3],
            v=[axis[1] * 0.3],
            w=[axis[2] * 0.3],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode='absolute',
            sizeref=scale * 0.3,
            name=f'{name}-{axis_name}-head',
            showlegend=False
        )
        traces.append(trace_cone)
    
    # 坐标系原点标记
    trace_origin = go.Scatter3d(
        x=[origin[0]],
        y=[origin[1]],
        z=[origin[2]],
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=[name],
        textposition='top center',
        name=f'{name} Origin',
        showlegend=True
    )
    traces.append(trace_origin)
    
    return traces

def load_hand_eye_result(json_file):
    """
    从JSON文件加载手眼标定结果
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    R = np.array(data['rotation_matrix'])
    t = np.array(data['translation_vector'])
    
    return R, t

def create_interactive_visualization(json_file='hand_eye_result.json'):
    """
    创建交互式手眼标定结果可视化
    """
    # 加载手眼标定结果
    R_cam2gripper, t_cam2gripper = load_hand_eye_result(json_file)
    
    # 机械臂末端坐标系（参考坐标系）
    R_gripper = np.eye(3)
    t_gripper = np.zeros((3, 1))
    
    # 相机坐标系
    R_camera = R_cam2gripper
    t_camera = t_cam2gripper
    
    # 创建图形
    fig = go.Figure()
    
    # 添加机械臂末端坐标系
    gripper_traces = create_coordinate_frame(
        R_gripper, t_gripper, 'Gripper', 
        scale=0.03, colors=['darkred', 'darkgreen', 'darkblue']
    )
    for trace in gripper_traces:
        fig.add_trace(trace)
    
    # 添加相机坐标系
    camera_traces = create_coordinate_frame(
        R_camera, t_camera, 'Camera', 
        scale=0.03, colors=['red', 'green', 'blue']
    )
    for trace in camera_traces:
        fig.add_trace(trace)
    
    # 添加连接线
    connection_trace = go.Scatter3d(
        x=[t_gripper[0, 0], t_camera[0, 0]],
        y=[t_gripper[1, 0], t_camera[1, 0]],
        z=[t_gripper[2, 0], t_camera[2, 0]],
        mode='lines',
        line=dict(color='black', width=4, dash='dash'),
        name='Transformation',
        showlegend=True
    )
    fig.add_trace(connection_trace)
    
    # 设置布局
    max_range = max(np.abs(t_camera).max(), 0.05)
    
    fig.update_layout(
        title={
            'text': 'Hand-Eye Calibration Result - Interactive 3D Visualization<br>Camera Frame relative to Gripper Frame',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis=dict(title='X (m)', range=[-max_range, max_range]),
            yaxis=dict(title='Y (m)', range=[-max_range, max_range]),
            zaxis=dict(title='Z (m)', range=[-max_range, max_range]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # 添加注释
    annotation_text = f"""
    <b>Transformation Parameters:</b><br>
    Translation (m): X={t_camera[0,0]:.4f}, Y={t_camera[1,0]:.4f}, Z={t_camera[2,0]:.4f}<br>
    Distance: {np.linalg.norm(t_camera):.4f} m<br>
    <br>
    <b>Rotation Angles (degrees):</b><br>
    Roll: {np.arcsin(R_camera[2,1]) * 180 / np.pi:.2f}°<br>
    Pitch: {np.arcsin(-R_camera[2,0]) * 180 / np.pi:.2f}°<br>
    Yaw: {np.arctan2(R_camera[1,0], R_camera[0,0]) * 180 / np.pi:.2f}°
    """
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def save_html_visualization(json_file='hand_eye_result.json', output_file='hand_eye_interactive.html'):
    """
    保存交互式可视化为HTML文件
    """
    fig = create_interactive_visualization(json_file)
    fig.write_html(output_file)
    print(f"Interactive visualization saved as: {output_file}")
    return fig

def show_visualization(json_file='hand_eye_result.json'):
    """
    显示交互式可视化
    """
    fig = create_interactive_visualization(json_file)
    fig.show()
    return fig

if __name__ == "__main__":
    print("Creating interactive 3D visualization...")
    
    # 保存HTML文件
    fig = save_html_visualization('hand_eye_result_realsense.json')
    # fig = save_html_visualization()
    
    # 尝试在浏览器中显示
    try:
        fig.show()
    except Exception as e:
        print(f"Cannot display in browser: {e}")
        print("Please open 'hand_eye_interactive.html' in your web browser to view the interactive visualization.")