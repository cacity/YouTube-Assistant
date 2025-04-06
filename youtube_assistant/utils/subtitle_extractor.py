"""
字幕提取器模块 - 负责音频转录和字幕生成

此模块使用Whisper模型将音频转录为文本，并生成SRT格式字幕文件。
"""

import os
import whisper
import torch
from pathlib import Path
import json
import subprocess

from youtube_assistant.utils.common import format_timestamp, ensure_directory, sanitize_filename, find_ffmpeg_path, check_ffmpeg
from youtube_assistant import config

def transcribe_audio_to_text(audio_path, output_dir=None, model_size="medium"):
    """
    使用Whisper将音频转录为文本
    
    参数:
        audio_path (str): 音频文件路径
        output_dir (str): 输出目录，如果为None，则使用默认目录
        model_size (str): Whisper模型大小，可选值: tiny, base, small, medium, large
        
    返回:
        str: 转录文本的文件路径
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = config.TRANSCRIPTS_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 首先验证音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        print(f"准备转录音频: {audio_path}")
        
        # 检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载模型
        print(f"加载 {model_size} 模型...")
        try:
            # 设置环境变量，避免某些CUDA相关问题
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 尝试加载模型
            model = whisper.load_model(model_size, device=device)
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
        
        # 转录音频
        print("开始转录音频...")
        result = model.transcribe(
            audio_path,
            fp16=False,
            verbose=True,
        )
        
        # 获取音频文件名（不含路径和扩展名）
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        
        # 保存转录文本
        transcript_path = os.path.join(output_dir, f"{audio_name}_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # 保存详细转录结果（包含时间戳等信息）
        json_path = os.path.join(output_dir, f"{audio_name}_transcript.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"音频转录完成，结果保存到: {transcript_path}")
        return transcript_path
    except Exception as e:
        print(f"音频转录失败: {str(e)}")
        raise

def create_subtitles(audio_path, output_dir=None, model_size="tiny", format_type="srt"):
    """
    从音频创建字幕文件
    
    参数:
        audio_path (str): 音频文件路径
        output_dir (str): 输出目录，如果为None，则使用默认目录
        model_size (str): Whisper模型大小，可选值: tiny, base, small, medium, large
        format_type (str): 字幕格式，可选值: srt, vtt, ass
        
    返回:
        str: 字幕文件路径
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = config.SUBTITLES_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 首先验证音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        print(f"准备从音频创建字幕: {audio_path}")
        
        # 检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载模型
        print(f"加载 {model_size} 模型...")
        try:
            # 设置环境变量，避免某些CUDA相关问题
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 尝试加载模型
            model = whisper.load_model(model_size, device=device)
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
        
        # 转录音频
        print("开始转录音频并生成字幕...")
        result = model.transcribe(
            audio_path,
            fp16=False,
            verbose=True,
        )
        
        # 获取音频文件名（不含路径和扩展名）
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        
        # 根据格式类型设置文件扩展名
        if format_type.lower() == "srt":
            extension = ".srt"
        elif format_type.lower() == "vtt":
            extension = ".vtt"
        elif format_type.lower() == "ass":
            extension = ".ass"
        else:
            raise ValueError(f"不支持的字幕格式: {format_type}")
        
        # 设置字幕文件路径
        subtitle_path = os.path.join(output_dir, f"{audio_name}{extension}")
        
        # 生成字幕文件
        if format_type.lower() == "srt":
            # 生成SRT格式字幕
            with open(subtitle_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"]):
                    # 段落编号
                    f.write(f"{i+1}\n")
                    
                    # 时间戳
                    start_time = format_timestamp(segment["start"], "srt")
                    end_time = format_timestamp(segment["end"], "srt")
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # 文本内容
                    f.write(f"{segment['text'].strip()}\n\n")
        
        elif format_type.lower() == "vtt":
            # 生成WebVTT格式字幕
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                
                for i, segment in enumerate(result["segments"]):
                    # 时间戳
                    start_time = format_timestamp(segment["start"], "vtt")
                    end_time = format_timestamp(segment["end"], "vtt")
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # 文本内容
                    f.write(f"{segment['text'].strip()}\n\n")
        
        elif format_type.lower() == "ass":
            # 生成ASS格式字幕
            with open(subtitle_path, "w", encoding="utf-8") as f:
                # 写入ASS头部
                f.write("[Script Info]\n")
                f.write("Title: Auto-generated subtitle\n")
                f.write("ScriptType: v4.00+\n")
                f.write("WrapStyle: 0\n")
                f.write("ScaledBorderAndShadow: yes\n")
                f.write("PlayResX: 1920\n")
                f.write("PlayResY: 1080\n\n")
                
                # 写入样式
                f.write("[V4+ Styles]\n")
                f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
                
                # 写入事件
                f.write("[Events]\n")
                f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
                
                for i, segment in enumerate(result["segments"]):
                    # 时间戳
                    start_time = format_timestamp(segment["start"], "ass")
                    end_time = format_timestamp(segment["end"], "ass")
                    
                    # 文本内容
                    text = segment['text'].strip().replace("\n", "\\N")
                    
                    # 写入对话行
                    f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
        
        print(f"字幕生成完成，保存到: {subtitle_path}")
        return subtitle_path
    except Exception as e:
        print(f"字幕生成失败: {str(e)}")
        raise

def create_bilingual_subtitles(audio_path, output_dir=None, model_size="tiny", translate_to_chinese=True):
    """
    创建双语字幕文件
    
    参数:
        audio_path (str): 音频文件路径
        output_dir (str): 输出目录，如果为None，则使用默认目录
        model_size (str): Whisper模型大小，可选值: tiny, base, small, medium, large
        translate_to_chinese (bool): 是否翻译成中文
        
    返回:
        str: 字幕文件路径
    """
    # 导入翻译模块（避免循环导入）
    from utils.translator import translate_text
    
    # 设置输出目录
    if output_dir is None:
        output_dir = config.SUBTITLES_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 首先验证音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        print(f"准备从音频创建双语字幕: {audio_path}")
        
        # 检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载模型
        print(f"加载 {model_size} 模型...")
        try:
            # 设置环境变量，避免某些CUDA相关问题
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 尝试加载模型
            model = whisper.load_model(model_size, device=device)
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
        
        # 转录音频
        print("开始转录音频并生成字幕...")
        result = model.transcribe(
            audio_path,
            fp16=False,
            verbose=True,
        )
        
        # 获取音频文件名（不含路径和扩展名）
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        
        # 设置字幕文件路径
        subtitle_path = os.path.join(output_dir, f"{audio_name}_bilingual.srt")
        
        # 生成双语字幕文件
        with open(subtitle_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                # 获取原始文本
                original_text = segment['text'].strip()
                
                # 如果需要翻译，则翻译文本
                if translate_to_chinese:
                    translated_text = translate_text(original_text, target_language='zh-CN')
                    subtitle_text = f"{original_text}\n{translated_text}"
                else:
                    subtitle_text = original_text
                
                # 段落编号
                f.write(f"{i+1}\n")
                
                # 时间戳
                start_time = format_timestamp(segment["start"], "srt")
                end_time = format_timestamp(segment["end"], "srt")
                f.write(f"{start_time} --> {end_time}\n")
                
                # 文本内容
                f.write(f"{subtitle_text}\n\n")
        
        print(f"双语字幕生成完成，保存到: {subtitle_path}")
        return subtitle_path
    except Exception as e:
        print(f"双语字幕生成失败: {str(e)}")
        raise

def embed_subtitles_to_video(video_path, subtitle_path, output_dir=None):
    """
    将字幕嵌入到视频中
    
    参数:
        video_path (str): 视频文件路径
        subtitle_path (str): 字幕文件路径
        output_dir (str): 输出目录，如果为None，则使用默认目录
        
    返回:
        str: 输出视频路径
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = config.VIDEOS_WITH_SUBTITLES_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 检查FFmpeg是否可用
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg未安装或不可用，无法嵌入字幕")
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        if not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
        
        # 获取视频文件名（不含路径和扩展名）
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # 设置输出视频文件路径
        output_video_path = os.path.join(output_dir, f"{video_name}_with_subtitles.mp4")
        
        # 获取FFmpeg路径
        ffmpeg_path = find_ffmpeg_path()
        
        # 使用FFmpeg嵌入字幕
        print(f"开始将字幕嵌入到视频: {video_path}")
        
        # 构建FFmpeg命令
        command = [
            ffmpeg_path,
            "-i", video_path,
            "-i", subtitle_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            output_video_path
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
            raise RuntimeError(f"字幕嵌入失败: {process.stderr}")
        
        # 检查输出文件是否存在
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"输出视频文件不存在: {output_video_path}")
        
        print(f"字幕嵌入完成，输出视频: {output_video_path}")
        return output_video_path
    except Exception as e:
        print(f"字幕嵌入失败: {str(e)}")
        raise
