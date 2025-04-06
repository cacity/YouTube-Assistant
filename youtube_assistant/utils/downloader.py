"""
下载器模块 - 负责从YouTube等平台下载视频和音频

此模块使用yt-dlp库从YouTube下载视频或音频，并提供相关处理功能。
"""

import os
import yt_dlp
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from youtube_assistant.utils.common import sanitize_filename, ensure_directory, check_ffmpeg, find_ffmpeg_path
from youtube_assistant import config

def download_youtube_video(youtube_url, output_dir=None, audio_only=True):
    """
    从YouTube下载视频或音频
    
    参数:
        youtube_url (str): YouTube视频链接
        output_dir (str): 输出目录，如果为None，则根据audio_only自动选择目录
        audio_only (bool): 是否只下载音频，如果为False则下载视频
        
    返回:
        str: 下载文件的完整路径
    """
    # 根据下载类型选择输出目录
    if output_dir is None:
        if audio_only:
            output_dir = config.DOWNLOADS_DIR
        else:
            output_dir = config.VIDEOS_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 获取视频信息
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
        # 获取视频标题并清理文件名
        video_title = info.get("title", "video")
        safe_title = sanitize_filename(video_title)
        
        # 当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置下载选项
        if audio_only:
            # 音频下载选项
            output_file = os.path.join(output_dir, f"{safe_title}_{timestamp}.mp3")
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_file,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': False,
                'no_warnings': False
            }
        else:
            # 视频下载选项
            output_file = os.path.join(output_dir, f"{safe_title}_{timestamp}.mp4")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_file,
                'quiet': False,
                'no_warnings': False
            }
        
        # 下载视频或音频
        print(f"开始下载{'音频' if audio_only else '视频'}: {video_title}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # 检查文件是否存在
        if not os.path.exists(output_file):
            # 尝试查找可能的输出文件
            base_path = os.path.splitext(output_file)[0]
            if audio_only:
                possible_files = [f"{base_path}.mp3", f"{base_path}.m4a", f"{base_path}.webm"]
            else:
                possible_files = [f"{base_path}.mp4", f"{base_path}.mkv", f"{base_path}.webm"]
                
            for possible_file in possible_files:
                if os.path.exists(possible_file):
                    output_file = possible_file
                    break
            else:
                # 如果仍然找不到文件，尝试查找目录中最新的文件
                files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
                if files:
                    newest_file = max(files, key=os.path.getctime)
                    output_file = newest_file
        
        print(f"下载完成: {output_file}")
        return output_file
    except Exception as e:
        print(f"下载失败: {str(e)}")
        raise

def extract_audio_from_video(video_path, output_dir=None):
    """
    从视频文件中提取音频
    
    参数:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录，如果为None，则使用默认目录
        
    返回:
        str: 提取的音频文件路径
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = config.EXTRACTED_AUDIO_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 检查FFmpeg是否可用
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg未安装或不可用，无法提取音频")
        
        # 获取视频文件名（不含路径和扩展名）
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # 当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置输出音频文件路径
        output_audio_path = os.path.join(output_dir, f"{video_name}_{timestamp}.mp3")
        
        # 获取FFmpeg路径
        ffmpeg_path = find_ffmpeg_path()
        
        # 使用FFmpeg提取音频
        print(f"开始从视频提取音频: {video_path}")
        
        # 构建FFmpeg命令
        command = [
            ffmpeg_path,
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-vn",
            output_audio_path
        ]
        
        # 执行命令
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # 检查是否成功
        if process.returncode != 0:
            print(f"FFmpeg错误: {process.stderr}")
            raise RuntimeError(f"音频提取失败: {process.stderr}")
        
        # 检查输出文件是否存在
        if not os.path.exists(output_audio_path):
            raise FileNotFoundError(f"提取的音频文件不存在: {output_audio_path}")
        
        print(f"音频提取完成: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"音频提取失败: {str(e)}")
        raise

def download_youtube_batch(urls, output_dir=None, audio_only=True):
    """
    批量下载YouTube视频或音频
    
    参数:
        urls (list): YouTube视频链接列表
        output_dir (str): 输出目录
        audio_only (bool): 是否只下载音频
        
    返回:
        list: 下载文件的完整路径列表
    """
    downloaded_files = []
    
    for i, url in enumerate(urls):
        try:
            print(f"处理第 {i+1}/{len(urls)} 个链接: {url}")
            file_path = download_youtube_video(url, output_dir, audio_only)
            downloaded_files.append(file_path)
        except Exception as e:
            print(f"处理链接 {url} 时出错: {str(e)}")
            continue
    
    return downloaded_files
