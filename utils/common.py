"""
工具函数模块 - 提供各种通用工具方法

此模块包含文件处理、格式转换等通用功能，被其他模块调用。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

def sanitize_filename(filename):
    """
    清理文件名，移除或替换不安全的字符
    
    参数:
        filename (str): 原始文件名
        
    返回:
        str: 清理后的文件名
    """
    # 替换不安全的字符
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '【', '】', '｜', ' ', '：']
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # 移除前导和尾随空格
    filename = filename.strip()
    
    # 确保文件名不为空
    if not filename:
        filename = "audio_file"
    
    return filename

def format_timestamp(seconds, format_type="srt"):
    """
    将秒数格式化为时间戳格式
    
    参数:
        seconds (float): 秒数
        format_type (str): 格式类型，可选 "srt" 或 "vtt"
        
    返回:
        str: 格式化的时间戳
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_value = seconds % 60
    
    if format_type.lower() == "srt":
        # SRT格式: HH:MM:SS,mmm
        milliseconds = int((seconds_value - int(seconds_value)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_value):02d},{milliseconds:03d}"
    elif format_type.lower() == "vtt":
        # WebVTT格式: HH:MM:SS.mmm
        milliseconds = int((seconds_value - int(seconds_value)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_value):02d}.{milliseconds:03d}"
    elif format_type.lower() == "ass":
        # ASS格式: H:MM:SS.cc
        centiseconds = int((seconds_value % 1) * 100)
        return f"{hours}:{minutes:02d}:{int(seconds_value):02d}.{centiseconds:02d}"
    else:
        raise ValueError(f"不支持的时间戳格式: {format_type}")

def ensure_directory(directory_path):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory_path (str): 目录路径
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def get_timestamp_str():
    """
    获取当前时间戳字符串，用于文件命名
    
    返回:
        str: 格式化的时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def check_ffmpeg():
    """
    检查系统中是否安装了FFmpeg
    
    返回:
        bool: 如果FFmpeg可用返回True，否则返回False
    """
    try:
        # 尝试运行ffmpeg命令
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False

def find_ffmpeg_path():
    """
    查找系统中FFmpeg的路径
    
    返回:
        str: FFmpeg可执行文件的路径，如果未找到则返回"ffmpeg"
    """
    ffmpeg_path = "ffmpeg"  # 默认命令名
    
    try:
        # 尝试使用which/where命令查找ffmpeg路径
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                ffmpeg_path = result.stdout.strip().split('\n')[0]
        else:  # Unix/Linux/Mac
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                ffmpeg_path = result.stdout.strip()
        
        # 常见的ffmpeg安装路径
        possible_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
        ]
        
        # 检查可能的路径
        for path in possible_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break
                
        return ffmpeg_path
    except Exception as e:
        print(f"查找ffmpeg路径失败: {str(e)}")
        return ffmpeg_path

def save_text_to_file(text, output_path):
    """
    将文本保存到文件
    
    参数:
        text (str): 要保存的文本
        output_path (str): 输出文件路径
        
    返回:
        str: 保存的文件路径
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        ensure_directory(output_dir)
        
        # 保存文本
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        return output_path
    except Exception as e:
        print(f"保存文件失败: {str(e)}")
        return None

def read_text_from_file(file_path):
    """
    从文件读取文本
    
    参数:
        file_path (str): 文件路径
        
    返回:
        str: 文件内容
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            with open(file_path, "r", encoding="gbk") as f:
                return f.read()
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            return None
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        return None

def create_error_report(error_message, context_info, output_dir):
    """
    创建错误报告文件
    
    参数:
        error_message (str): 错误信息
        context_info (dict): 上下文信息
        output_dir (str): 输出目录
        
    返回:
        str: 错误报告文件路径
    """
    import torch
    
    # 创建输出目录
    ensure_directory(output_dir)
    
    # 生成错误报告文件路径
    error_report_path = os.path.join(output_dir, f"error_report_{get_timestamp_str()}.txt")
    
    try:
        with open(error_report_path, "w", encoding="utf-8") as f:
            f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 写入上下文信息
            for key, value in context_info.items():
                f.write(f"{key}: {value}\n")
                
            # 写入错误信息
            f.write(f"\n错误信息: {error_message}\n")
            
            # 添加系统信息
            f.write("\n系统信息:\n")
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"PyTorch版本: {torch.__version__}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA版本: {torch.version.cuda}\n")
                f.write(f"CUDA设备: {torch.cuda.get_device_name(0)}\n")
            else:
                f.write("CUDA: 不可用\n")
        
        print(f"错误报告已保存到: {error_report_path}")
        return error_report_path
    except Exception as report_error:
        print(f"保存错误报告失败: {str(report_error)}")
        return None
