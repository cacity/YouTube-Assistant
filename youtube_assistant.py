import yt_dlp
import whisper
import torch
from pathlib import Path
import openai
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import shutil
import requests
import html
import subprocess

# Load environment variables from .env file
load_dotenv()

# 创建模板目录
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# 默认模板
DEFAULT_TEMPLATE = """请将以下文本改写成一篇完整、连贯、专业的文章。

要求：
1. 你是一名资深科技领域编辑，同时具备优秀的文笔，文本转为一篇文章，确保段落清晰，文字连贯，可读性强，必要修改调整段落结构，确保内容具备良好的逻辑性。
2. 添加适当的小标题来组织内容
3. 以markdown格式输出，充分利用标题、列表、引用等格式元素
4. 如果原文有技术内容，确保准确表达并提供必要的解释

原文内容：
{content}
"""

# 创建默认模板文件
DEFAULT_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "default.txt")
if not os.path.exists(DEFAULT_TEMPLATE_PATH):
    with open(DEFAULT_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        f.write(DEFAULT_TEMPLATE)

def load_template(template_path=None):
    """
    加载模板文件
    :param template_path: 模板文件路径，如果为None则使用默认模板
    :return: 模板内容
    """
    if template_path is None:
        # 使用默认模板
        template_path = DEFAULT_TEMPLATE_PATH
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"加载模板文件失败: {str(e)}")
        print(f"使用内置默认模板")
        return DEFAULT_TEMPLATE

def sanitize_filename(filename):
    """
    清理文件名，移除或替换不安全的字符
    :param filename: 原始文件名
    :return: 清理后的文件名
    """
    # 替换不安全的字符
    # 扩展不安全字符列表，包含更多特殊符号
    unsafe_chars = [
        '<', '>', ':', '"', '/', '\\', '|', '?', '*',  # 基本不安全字符
        '【', '】', '｜', '：',  # 中文特殊字符
        '!', '@', '#', '$', '%', '^', '&', '(', ')', '+', '=',  # 其他特殊符号
        '[', ']', '{', '}', ';', "'", ',', '.', '`', '~',  # 更多特殊符号
        '—', '–', '…', '“', '”', '‘', '’',  # 破折号、引号等
        '©', '®', '™',  # 版权符号、商标符号
    ]
    
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # 替换空格为下划线
    filename = filename.replace(' ', '_')
    
    # 处理多个连续的下划线
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # 移除前导和尾随的下划线和空格
    filename = filename.strip('_').strip()
    
    # 确保文件名不为空
    if not filename:
        filename = "video_file"
    
    # 限制文件名长度，避免路径过长
    if len(filename) > 100:
        filename = filename[:97] + '...'
    
    return filename

def translate_text(text, target_language='zh-CN', source_language='auto'):
    """
    使用Google翻译API翻译文本
    :param text: 要翻译的文本
    :param target_language: 目标语言代码，默认为中文
    :param source_language: 源语言代码，默认为自动检测
    :return: 翻译后的文本
    """
    try:
        # Google翻译API的URL
        url = "https://translate.googleapis.com/translate_a/single"
        
        # 请求参数
        params = {
            "client": "gtx",
            "sl": source_language,
            "tl": target_language,
            "dt": "t",
            "q": text
        }
        
        # 发送请求
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # 解析响应
            result = response.json()
            
            # 提取翻译文本
            translated_text = ""
            for sentence in result[0]:
                if sentence[0]:
                    translated_text += sentence[0]
            
            return html.unescape(translated_text)
        else:
            print(f"翻译请求失败: {response.status_code}")
            return text
    except Exception as e:
        print(f"翻译过程中出错: {str(e)}")
        return text

def format_timestamp(seconds):
    """
    将秒数格式化为SRT时间戳格式 (HH:MM:SS,mmm)
    :param seconds: 秒数
    :return: 格式化的时间戳
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def download_youtube_video(youtube_url, output_dir=None, audio_only=True):
    """
    从YouTube下载视频或音频
    :param youtube_url: YouTube视频链接
    :param output_dir: 输出目录，如果为None，则根据audio_only自动选择目录
    :param audio_only: 是否只下载音频，如果为False则下载视频
    :return: 下载文件的完整路径
    """
    # 根据下载类型选择默认输出目录
    if output_dir is None:
        output_dir = "downloads" if audio_only else "videos"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 设置yt-dlp的选项
    if audio_only:
        # 音频下载选项
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',  # 优先选择m4a格式的音频
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': False,  # 显示下载进度和错误信息
            'ignoreerrors': True,  # 忽略部分错误，尝试继续下载
            'noplaylist': True  # 确保只下载单个视频的音频而不是整个播放列表
        }
        expected_ext = "mp3"
    else:
        # 视频下载选项（最佳画质）
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',  # 使用更可靠的格式组合
            'merge_output_format': 'mp4',  # 确保输出为mp4
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': False,  # 显示下载进度和错误信息
            'ignoreerrors': True,  # 忽略部分错误，尝试继续下载
            'noplaylist': True  # 确保只下载单个视频而不是整个播放列表
        }
        expected_ext = "mp4"
    
    try:
        print(f"开始{'音频' if audio_only else '视频'}下载: {youtube_url}")
        print(f"下载选项: {'仅音频' if audio_only else '完整视频'}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 获取视频信息
            print(f"正在获取视频信息...")
            info = ydl.extract_info(youtube_url, download=True)
            
            # 获取原始文件名并清理
            original_title = info['title']
            print(f"原始视频标题: {original_title}")
            sanitized_title = sanitize_filename(original_title)
            print(f"清理后的标题: {sanitized_title}")
            
            # 构建文件路径
            original_path = os.path.join(output_dir, f"{original_title}.{expected_ext}")
            sanitized_path = os.path.join(output_dir, f"{sanitized_title}.{expected_ext}")
            
            print(f"原始文件路径: {original_path}")
            print(f"清理后的文件路径: {sanitized_path}")
            
            # 如果文件名被清理了，需要重命名文件
            if original_path != sanitized_path and os.path.exists(original_path):
                try:
                    os.rename(original_path, sanitized_path)
                    print(f"文件已重命名: {original_title} -> {sanitized_title}")
                except Exception as e:
                    print(f"重命名文件失败: {str(e)}")
            
            # 检查文件是否存在
            if os.path.exists(sanitized_path):
                print(f"文件下载成功: {sanitized_path}")
                return sanitized_path
            elif os.path.exists(original_path):
                print(f"文件下载成功但未重命名: {original_path}")
                return original_path
            else:
                # 尝试查找可能的文件
                possible_files = list(Path(output_dir).glob(f"*.{expected_ext}"))
                if possible_files:
                    newest_file = max(possible_files, key=os.path.getctime)
                    print(f"找到可能的文件: {newest_file}")
                    return str(newest_file)
                
                # 如果找不到预期扩展名的文件，尝试查找任何新文件
                all_files = list(Path(output_dir).glob("*.*"))
                if all_files:
                    newest_file = max(all_files, key=os.path.getctime)
                    print(f"找到可能的文件（不同扩展名）: {newest_file}")
                    return str(newest_file)
                
                raise Exception(f"下载成功但找不到文件，请检查 {output_dir} 目录")
    except yt_dlp.utils.DownloadError as e:
        print(f"下载失败详细信息: {str(e)}")
        raise Exception(f"下载失败: {str(e)}")
    except Exception as e:
        print(f"下载失败详细信息: {str(e)}")
        raise Exception(f"下载失败: {str(e)}")

def download_youtube_audio(youtube_url, output_dir="downloads"):
    """
    从YouTube视频中下载音频
    :param youtube_url: YouTube视频链接
    :param output_dir: 输出目录
    :return: 音频文件的完整路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置yt-dlp的选项
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 获取视频信息
            info = ydl.extract_info(youtube_url, download=True)
            
            # 获取原始文件名并清理
            original_title = info['title']
            sanitized_title = sanitize_filename(original_title)
            
            # 如果文件名被清理了，需要重命名文件
            original_path = os.path.join(output_dir, f"{original_title}.mp3")
            sanitized_path = os.path.join(output_dir, f"{sanitized_title}.mp3")
            
            if original_path != sanitized_path and os.path.exists(original_path):
                try:
                    os.rename(original_path, sanitized_path)
                    print(f"文件已重命名: {original_title} -> {sanitized_title}")
                except Exception as e:
                    print(f"重命名文件失败: {str(e)}")
            
            # 返回清理后的文件路径
            return sanitized_path
    except Exception as e:
        raise Exception(f"下载音频失败: {str(e)}")

def extract_audio_from_video(video_path, output_dir="downloads"):
    """
    从视频文件中提取音频
    :param video_path: 视频文件路径
    :param output_dir: 输出目录，默认为downloads
    :return: 提取的音频文件路径
    """
    try:
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取视频文件名（不含扩展名）
        video_name = Path(video_path).stem
        sanitized_name = sanitize_filename(video_name)
        
        # 设置输出音频路径
        audio_path = os.path.join(output_dir, f"{sanitized_name}.mp3")
        
        print(f"正在从视频提取音频: {video_path} -> {audio_path}")
        
        # 检查ffmpeg是否可用
        try:
            import subprocess
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("警告: ffmpeg命令不可用。请确保已安装ffmpeg并添加到系统PATH中。")
                print("您可以从 https://ffmpeg.org/download.html 下载ffmpeg。")
                raise Exception("ffmpeg命令不可用")
        except FileNotFoundError:
            print("错误: 找不到ffmpeg命令。请确保已安装ffmpeg并添加到系统PATH中。")
            print("您可以从 https://ffmpeg.org/download.html 下载ffmpeg。")
            raise Exception("找不到ffmpeg命令")
        
        # 首先检查视频文件是否包含音频流
        import subprocess
        probe_cmd = ["ffmpeg", "-i", video_path, "-hide_banner"]
        probe_process = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = probe_process.communicate()
        stderr_text = stderr.decode('utf-8', errors='ignore')
        
        # 检查输出中是否包含音频流信息
        if "Stream" in stderr_text and "Audio" not in stderr_text:
            print("警告: 视频文件不包含音频流")
            print("错误详情:")
            print(stderr_text)
            raise Exception("视频文件不包含音频流，无法提取音频")
        
        # 使用ffmpeg-python库提取音频
        try:
            import ffmpeg
            # 使用ffmpeg-python库
            try:
                # 先获取视频信息
                probe = ffmpeg.probe(video_path)
                # 检查是否有音频流
                audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                if not audio_streams:
                    raise Exception("视频文件不包含音频流，无法提取音频")
                
                # 有音频流，继续处理
                (
                    ffmpeg
                    .input(video_path)
                    .output(audio_path, acodec='libmp3lame', q=0)
                    .run(quiet=False, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                print(f"音频提取完成: {audio_path}")
            except ffmpeg._run.Error as e:
                print(f"ffmpeg错误: {str(e)}")
                print("尝试使用subprocess直接调用ffmpeg...")
                raise Exception("ffmpeg-python库调用失败，尝试使用subprocess")
        except (ImportError, Exception) as e:
            # 如果ffmpeg-python库不可用或调用失败，回退到subprocess
            print(f"使用subprocess调用ffmpeg: {str(e)}")
            import subprocess
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-q:a", "0",
                "-vn",
                "-y",  # 覆盖输出文件
                audio_path
            ]
            
            # 执行命令，捕获输出
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                print(f"ffmpeg命令执行失败，返回代码: {process.returncode}")
                print(f"错误输出: {stderr_text}")
                
                # 检查是否是因为没有音频流
                if "Stream map 'a' matches no streams" in stderr_text or "does not contain any stream" in stderr_text:
                    raise Exception("视频文件不包含音频流，无法提取音频")
                else:
                    raise Exception(f"ffmpeg命令执行失败: {stderr_text}")
            
            print(f"音频提取完成: {audio_path}")
        
        # 检查生成的音频文件是否存在
        if not os.path.exists(audio_path):
            raise Exception(f"音频文件未生成: {audio_path}")
        
        # 检查音频文件大小
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise Exception(f"生成的音频文件大小为0: {audio_path}")
        
        print(f"音频文件大小: {file_size} 字节")
        return audio_path
    except Exception as e:
        error_msg = f"从视频提取音频失败: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def transcribe_audio_to_text(audio_path, output_dir="transcripts", model_size="small"):
    """
    使用Whisper将音频转换为文本
    :param audio_path: 音频文件路径
    :param output_dir: 输出目录
    :param model_size: 模型大小，可选 "tiny", "base", "small", "medium", "large"
    :return: 文本文件的路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 确保CUDA可用时正确设置
        if device == "cuda":
            print(f"CUDA是否可用: {torch.cuda.is_available()}")
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        
        # 加载模型
        print(f"加载 {model_size} 模型...")
        model = whisper.load_model(model_size, device=device)
        
        # 转录音频
        print("开始转录音频...")
        result = model.transcribe(audio_path)
        print("转录完成!")
        
        # 生成输出文件路径
        base_name = Path(audio_path).stem
        sanitized_base_name = sanitize_filename(base_name)
        output_path = os.path.join(output_dir, f"{sanitized_base_name}_transcript.txt")
        
        # 保存转录文本
        with open(output_path, "w", encoding="utf-8") as f:
            # 如果result包含segments，按段落保存
            if 'segments' in result:
                for segment in result['segments']:
                    f.write(f"{segment['text'].strip()}\n\n")
            else:
                f.write(result['text'])
        
        return output_path
    except Exception as e:
        raise Exception(f"音频转文字失败: {str(e)}")

def transcribe_only(audio_path, whisper_model_size="medium", output_dir="transcripts"):
    """
    仅将音频转换为文本，不进行摘要生成
    
    参数:
        audio_path (str): 音频文件路径
        whisper_model_size (str): Whisper模型大小
        output_dir (str): 转录文本保存目录
    
    返回:
        str: 转录文本文件路径
    """
    print(f"正在将音频转换为文本: {audio_path}")
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 文件 {audio_path} 不存在")
        return None
    
    # 转录音频
    text_path = transcribe_audio_to_text(audio_path, output_dir=output_dir, model_size=whisper_model_size)
    
    print(f"音频转文本完成，文本已保存至: {text_path}")
    return text_path

def create_bilingual_subtitles(audio_path, output_dir="subtitles", model_size="tiny", translate_to_chinese=True):
    """
    创建双语字幕文件
    :param audio_path: 音频文件路径
    :param output_dir: 输出目录
    :param model_size: Whisper模型大小
    :param translate_to_chinese: 是否翻译成中文
    :return: 字幕文件路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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
            task="transcribe"  # 使用转录任务
        )
        
        # 获取源语言
        source_language = result.get("language", "en")
        print(f"检测到的语言: {source_language}")
        
        # 生成输出文件路径
        base_name = Path(audio_path).stem
        sanitized_name = sanitize_filename(base_name)
        srt_path = os.path.join(output_dir, f"{sanitized_name}_bilingual.srt")
        vtt_path = os.path.join(output_dir, f"{sanitized_name}_bilingual.vtt")
        ass_path = os.path.join(output_dir, f"{sanitized_name}_bilingual.ass")
        
        # 创建SRT字幕文件
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            # 写入SRT文件
            for i, segment in enumerate(result["segments"]):
                # 获取时间戳
                start_time = segment["start"]
                end_time = segment["end"]
                
                # 获取文本
                original_text = segment["text"].strip()
                
                # 如果需要翻译且源语言不是中文
                translated_text = ""
                if translate_to_chinese and source_language != "zh" and source_language != "chi":
                    translated_text = translate_text(original_text, target_language="zh-CN", source_language=source_language)
                    print(f"翻译: {original_text} -> {translated_text}")
                
                # 写入字幕索引
                srt_file.write(f"{i+1}\n")
                
                # 写入时间戳
                srt_file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                
                # 写入原文
                srt_file.write(f"{original_text}\n")
                
                # 如果有翻译，写入翻译
                if translated_text:
                    srt_file.write(f"{translated_text}\n")
                
                # 空行分隔
                srt_file.write("\n")
        
        # 创建WebVTT字幕文件
        with open(vtt_path, "w", encoding="utf-8") as vtt_file:
            # 写入WebVTT头
            vtt_file.write("WEBVTT\n\n")
            
            # 写入字幕
            for i, segment in enumerate(result["segments"]):
                # 获取时间戳
                start_time = segment["start"]
                end_time = segment["end"]
                
                # 格式化WebVTT时间戳 (HH:MM:SS.mmm)
                start_formatted = str(timedelta(seconds=start_time)).rjust(8, '0').replace(',', '.')
                end_formatted = str(timedelta(seconds=end_time)).rjust(8, '0').replace(',', '.')
                
                # 获取文本
                original_text = segment["text"].strip()
                
                # 如果需要翻译且源语言不是中文
                translated_text = ""
                if translate_to_chinese and source_language != "zh" and source_language != "chi":
                    # 使用缓存的翻译结果，避免重复翻译
                    if not hasattr(create_bilingual_subtitles, 'translation_cache'):
                        create_bilingual_subtitles.translation_cache = {}
                    
                    if original_text in create_bilingual_subtitles.translation_cache:
                        translated_text = create_bilingual_subtitles.translation_cache[original_text]
                    else:
                        translated_text = translate_text(original_text, target_language="zh-CN", source_language=source_language)
                        create_bilingual_subtitles.translation_cache[original_text] = translated_text
                
                # 写入时间戳
                vtt_file.write(f"{start_formatted} --> {end_formatted}\n")
                
                # 写入原文
                vtt_file.write(f"{original_text}\n")
                
                # 如果有翻译，写入翻译
                if translated_text:
                    vtt_file.write(f"{translated_text}\n")
                
                # 空行分隔
                vtt_file.write("\n")
        
        # 创建ASS字幕文件（高级字幕格式，支持更多样式）
        with open(ass_path, "w", encoding="utf-8") as ass_file:
            # 写入ASS头部
            ass_file.write("[Script Info]\n")
            ass_file.write("Title: 双语字幕\n")
            ass_file.write("ScriptType: v4.00+\n")
            ass_file.write("WrapStyle: 0\n")
            ass_file.write("ScaledBorderAndShadow: yes\n")
            ass_file.write("YCbCr Matrix: TV.601\n")
            ass_file.write("PlayResX: 1920\n")
            ass_file.write("PlayResY: 1080\n\n")
            
            # 写入样式
            ass_file.write("[V4+ Styles]\n")
            ass_file.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            ass_file.write("Style: Original,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n")
            ass_file.write("Style: Translation,Arial,28,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
            
            # 写入事件
            ass_file.write("[Events]\n")
            ass_file.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            # 写入字幕
            for i, segment in enumerate(result["segments"]):
                # 获取时间戳
                start_time = segment["start"]
                end_time = segment["end"]
                
                # 格式化ASS时间戳 (H:MM:SS.cc)
                start_h = int(start_time / 3600)
                start_m = int((start_time % 3600) / 60)
                start_s = int(start_time % 60)
                start_cs = int((start_time % 1) * 100)
                start_formatted = f"{start_h}:{start_m:02d}:{start_s:02d}.{start_cs:02d}"
                
                end_h = int(end_time / 3600)
                end_m = int((end_time % 3600) / 60)
                end_s = int(end_time % 60)
                end_cs = int((end_time % 1) * 100)
                end_formatted = f"{end_h}:{end_m:02d}:{end_s:02d}.{end_cs:02d}"
                
                # 获取文本
                original_text = segment["text"].strip()
                
                # 写入原文
                ass_file.write(f"Dialogue: 0,{start_formatted},{end_formatted},Original,,0,0,0,,{original_text}\n")
                
                # 如果需要翻译且源语言不是中文
                if translate_to_chinese and source_language != "zh" and source_language != "chi":
                    # 使用缓存的翻译结果
                    if not hasattr(create_bilingual_subtitles, 'translation_cache'):
                        create_bilingual_subtitles.translation_cache = {}
                    
                    if original_text in create_bilingual_subtitles.translation_cache:
                        translated_text = create_bilingual_subtitles.translation_cache[original_text]
                    else:
                        translated_text = translate_text(original_text, target_language="zh-CN", source_language=source_language)
                        create_bilingual_subtitles.translation_cache[original_text] = translated_text
                    
                    # 写入翻译
                    ass_file.write(f"Dialogue: 0,{start_formatted},{end_formatted},Translation,,0,0,0,,{translated_text}\n")
        
        print(f"字幕文件已创建: \nSRT: {srt_path}\nVTT: {vtt_path}\nASS: {ass_path}")
        
        # 返回SRT文件路径作为默认字幕文件
        return srt_path
    
    except Exception as e:
        print(f"创建字幕文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def embed_subtitles_to_video(video_path, subtitle_path, output_dir="videos_with_subtitles"):
    """
    将字幕嵌入到视频中
    
    Args:
        video_path: 视频文件路径
        subtitle_path: 字幕文件路径
        output_dir: 输出目录
        
    Returns:
        输出视频路径
    """
    try:
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取视频文件名（不含扩展名）
        video_name = Path(video_path).stem
        video_ext = Path(video_path).suffix
        
        # 生成输出文件路径
        output_path = os.path.join(output_dir, f"{video_name}_with_subtitles{video_ext}")
        
        # 获取字幕文件扩展名
        subtitle_ext = Path(subtitle_path).suffix
        
        # 如果是SRT字幕，优先使用ASS格式（如果存在）
        if subtitle_ext == '.srt' and os.path.exists(subtitle_path.replace('.srt', '.ass')):
            subtitle_path = subtitle_path.replace('.srt', '.ass')
            print(f"找到ASS格式字幕，使用: {subtitle_path}")
        
        print(f"正在将字幕嵌入视频: {video_path}")
        print(f"使用字幕文件: {subtitle_path}")
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
        
        # 创建临时字幕文件，避免路径问题
        temp_dir = os.path.join(os.path.dirname(output_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 使用简单的文件名，避免路径问题
        temp_subtitle = os.path.join(temp_dir, f"temp_subtitle{Path(subtitle_path).suffix}")
        
        # 复制字幕文件到临时位置
        import shutil
        shutil.copy2(subtitle_path, temp_subtitle)
        
        # 检查临时字幕文件是否存在
        if not os.path.exists(temp_subtitle):
            raise FileNotFoundError(f"临时字幕文件不存在: {temp_subtitle}")
        
        # 获取绝对路径
        video_path_abs = os.path.abspath(video_path)
        temp_subtitle_abs = os.path.abspath(temp_subtitle)
        output_path_abs = os.path.abspath(output_path)
        
        # 输出调试信息
        print(f"视频绝对路径: {video_path_abs}")
        print(f"临时字幕绝对路径: {temp_subtitle_abs}")
        print(f"输出视频绝对路径: {output_path_abs}")
        
        try:
            # 尝试使用简单的FFmpeg命令，使用escape=1参数
            # 首先尝试查找ffmpeg的路径
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
                
                print(f"找到ffmpeg路径: {ffmpeg_path}")
            except Exception as e:
                print(f"查找ffmpeg路径失败，使用默认命令: {str(e)}")
            
            # 方法1: 使用修改后的命令行参数
            if os.name == 'nt':  # Windows
                # 将路径中的反斜杠替换为正斜杠
                temp_subtitle_path = temp_subtitle_abs.replace('\\', '/')
                # 简化字幕参数，避免引号嵌套问题
                subtitles_param = f'subtitles={temp_subtitle_path}:force_style=FontName\\=Cascadia\\ Mono,FontSize\\=10'
            else:  # Unix/Linux/Mac
                subtitles_param = f"subtitles='{temp_subtitle_abs}':force_style='FontName=Cascadia Mono,FontSize=10'"
            
            cmd = [
                ffmpeg_path,
                "-i", video_path_abs,
                "-vf", subtitles_param,
                "-c:a", "copy",
                "-c:v", "libx264",
                "-crf", "20",
                "-vsync", "cfr",
                "-y",
                output_path_abs
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 在Windows下设置控制台编码为UTF-8
            if os.name == 'nt':
                os.system('chcp 65001 > nul')
                
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"FFmpeg错误输出: {stderr}")
                raise Exception(f"ffmpeg命令执行失败，返回代码: {process.returncode}")
            
            # 清理临时文件
            try:
                os.remove(temp_subtitle)
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"清理临时文件失败: {str(e)}")
                
            print(f"字幕嵌入完成: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"嵌入字幕失败: {str(e)}")
            
            # 尝试使用替代方法
            print("尝试使用替代方法...")
            
            # 使用ffmpeg命令行方式，避免路径问题
            # 创建一个简单的批处理文件来执行命令
            batch_file = os.path.join(temp_dir, "embed_subtitles.bat")
            
            # 尝试直接使用ffmpeg-python库
            try:
                print("尝试使用ffmpeg-python库...")
                import ffmpeg
                
                # 构建ffmpeg命令
                (
                    ffmpeg
                    .input(video_path_abs)
                    .output(
                        output_path_abs,
                        vf=f"ass={temp_subtitle_abs}",
                        acodec='copy',
                        vcodec='libx264',
                        crf=20,
                        vsync='cfr',
                        y=None
                    )
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                print(f"ffmpeg-python库执行成功")
                
                # 清理临时文件
                try:
                    os.remove(temp_subtitle)
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"清理临时文件失败: {str(e)}")
                    
                print(f"字幕嵌入完成: {output_path}")
                return output_path
                
            except Exception as ffmpeg_error:
                print(f"ffmpeg-python库执行失败: {str(ffmpeg_error)}")
                print("尝试使用直接命令行方式...")
                
                # 查找系统中的ffmpeg可执行文件
                ffmpeg_executable = ffmpeg_path  # 使用之前找到的路径
                
                # 常见的ffmpeg安装路径
                possible_paths = [
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe"),
                    r"D:\soft\bin\ffmpeg.exe"  # 添加用户环境中的路径
                ]
                
                # 检查可能的路径
                for path in possible_paths:
                    if os.path.exists(path):
                        ffmpeg_executable = path
                        print(f"找到ffmpeg可执行文件: {ffmpeg_executable}")
                        break
                
                # 构建批处理文件内容 - 使用简化的命令，避免引号和特殊字符问题
                batch_content = f"""@echo off
                cd /d "{os.path.dirname(temp_subtitle_abs)}"
                "{ffmpeg_executable}" -i "{video_path_abs}" -vf "subtitles=temp_subtitle{Path(subtitle_path).suffix}" -c:a copy -c:v libx264 -crf 20 -vsync cfr -y "{output_path_abs}"
                """
                
                # 写入批处理文件
                with open(batch_file, 'w', encoding='utf-8') as f:
                    f.write(batch_content)
                
                print(f"执行批处理文件: {batch_file}")
                print(f"批处理文件内容:\n{batch_content}")
                
                # 执行批处理文件
                process = subprocess.Popen(
                    batch_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    print(f"批处理执行错误: {stderr}")
                    
                    # 最后尝试直接使用subprocess调用
                    print("尝试直接使用subprocess调用...")
                    
                    # 切换到临时目录
                    original_dir = os.getcwd()
                    os.chdir(os.path.dirname(temp_subtitle_abs))
                    
                    try:
                        # 使用相对路径引用字幕文件
                        cmd = [
                            ffmpeg_executable,
                            "-i", video_path_abs,
                            "-vf", f"subtitles=temp_subtitle{Path(subtitle_path).suffix}",
                            "-c:a", "copy",
                            "-c:v", "libx264",
                            "-crf", "20",
                            "-vsync", "cfr",
                            "-y",
                            output_path_abs
                        ]
                        
                        print(f"执行命令: {' '.join(cmd)}")
                        
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False
                        )
                        
                        if result.returncode != 0:
                            print(f"命令执行错误: {result.stderr}")
                            raise Exception(f"命令执行失败，返回代码: {result.returncode}")
                        
                    finally:
                        # 恢复原始目录
                        os.chdir(original_dir)
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_subtitle)
                        os.remove(batch_file)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        print(f"清理临时文件失败: {str(e)}")
                    
                    print(f"字幕嵌入完成: {output_path}")
                    return output_path
                
                # 清理临时文件
                try:
                    os.remove(temp_subtitle)
                    os.remove(batch_file)
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"清理临时文件失败: {str(e)}")
                
                print(f"字幕嵌入完成: {output_path}")
                return output_path
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise Exception(f"嵌入字幕失败: {str(e)}")

def process_local_audio(audio_path, model=None, api_key=None, base_url=None, whisper_model_size="medium", stream=True, summary_dir="summaries", custom_prompt=None, template_path=None, generate_subtitles=False, translate_to_chinese=True):
    """
    处理本地音频文件的主函数
    :param audio_path: 本地音频文件路径
    :param model: 使用的模型名称，默认从环境变量获取
    :param api_key: API密钥，默认从环境变量获取
    :param base_url: 自定义API基础URL，默认从环境变量获取
    :param whisper_model_size: Whisper模型大小，默认为medium
    :param stream: 是否使用流式输出生成总结，默认为True
    :param summary_dir: 总结文件保存目录，默认为summaries
    :param custom_prompt: 自定义提示词，如果提供则使用此提示词代替默认提示词
    :param template_path: 模板文件路径，如果提供则使用此模板
    :param generate_subtitles: 是否生成字幕文件，默认为False
    :param translate_to_chinese: 是否将字幕翻译成中文，默认为True
    :return: 总结文件的路径
    """
    try:
        print("1. 开始转录音频...")
        text_path = transcribe_audio_to_text(audio_path, model_size=whisper_model_size)
        print(f"转录文本已保存到: {text_path}")
        
        # 生成字幕文件
        subtitle_path = None
        if generate_subtitles:
            print("\n2. 生成字幕文件...")
            subtitle_path = create_bilingual_subtitles(
                audio_path, 
                output_dir="subtitles", 
                model_size=whisper_model_size,
                translate_to_chinese=translate_to_chinese
            )
            if subtitle_path:
                print(f"字幕文件已生成: {subtitle_path}")
            else:
                print("字幕生成失败")
        
        print("\n3. 开始生成文章...")
        summary_path = summarize_text(
            text_path, 
            model=model, 
            api_key=api_key, 
            base_url=base_url, 
            stream=stream,
            output_dir=summary_dir,
            custom_prompt=custom_prompt,
            template_path=template_path
        )
        print(f"文章已保存到: {summary_path}")
        
        return summary_path
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

def process_local_video(video_path, model=None, api_key=None, base_url=None, whisper_model_size="medium", stream=True, summary_dir="summaries", custom_prompt=None, template_path=None, generate_subtitles=False, translate_to_chinese=True, embed_subtitles=False):
    """
    处理本地视频文件的主函数
    :param video_path: 本地视频文件路径
    :param model: 使用的模型名称，默认从环境变量获取
    :param api_key: API密钥，默认从环境变量获取
    :param base_url: 自定义API基础URL，默认从环境变量获取
    :param whisper_model_size: Whisper模型大小，默认为medium
    :param stream: 是否使用流式输出生成总结，默认为True
    :param summary_dir: 总结文件保存目录，默认为summaries
    :param custom_prompt: 自定义提示词，如果提供则使用此提示词代替默认提示词
    :param template_path: 模板文件路径，如果提供则使用此模板
    :param generate_subtitles: 是否生成字幕文件，默认为False
    :param translate_to_chinese: 是否将字幕翻译成中文，默认为True
    :param embed_subtitles: 是否将字幕嵌入到视频中，默认为False
    :return: 总结文件的路径
    """
    try:
        print("1. 从视频中提取音频...")
        audio_path = extract_audio_from_video(video_path, output_dir="downloads")
        print(f"音频已提取到: {audio_path}")
        
        print("2. 开始转录音频...")
        text_path = transcribe_audio_to_text(audio_path, model_size=whisper_model_size)
        print(f"转录文本已保存到: {text_path}")
        
        # 生成字幕文件
        subtitle_path = None
        if generate_subtitles:
            print("\n3. 生成字幕文件...")
            subtitle_path = create_bilingual_subtitles(
                audio_path, 
                output_dir="subtitles", 
                model_size=whisper_model_size,
                translate_to_chinese=translate_to_chinese
            )
            if subtitle_path:
                print(f"字幕文件已生成: {subtitle_path}")
                
                # 将字幕嵌入到视频中
                if embed_subtitles:
                    print("\n4. 将字幕嵌入到视频中...")
                    video_with_subtitles = embed_subtitles_to_video(
                        video_path,
                        subtitle_path,
                        output_dir="videos_with_subtitles"
                    )
                    if video_with_subtitles:
                        print(f"带字幕的视频已生成: {video_with_subtitles}")
                    else:
                        print("字幕嵌入失败")
            else:
                print("字幕生成失败")
        
        print("\n5. 开始生成文章...")
        summary_path = summarize_text(
            text_path, 
            model=model, 
            api_key=api_key, 
            base_url=base_url, 
            stream=stream,
            output_dir=summary_dir,
            custom_prompt=custom_prompt,
            template_path=template_path
        )
        print(f"文章已保存到: {summary_path}")
        
        return summary_path
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

def summarize_text(text_path, model=None, api_key=None, base_url=None, stream=False, output_dir="summaries", custom_prompt=None, template_path=None):
    """
    使用大语言模型总结文本内容
    :param text_path: 文本文件路径
    :param model: 使用的模型名称，默认从环境变量获取
    :param api_key: API密钥，默认从环境变量获取
    :param base_url: 自定义API基础URL，默认从环境变量获取
    :param stream: 是否使用流式输出，默认为False
    :param output_dir: 输出目录，默认为summaries
    :param custom_prompt: 自定义提示词，如果提供则使用此提示词代替默认提示词
    :param template_path: 模板文件路径，如果提供则使用此模板
    :return: Markdown格式的总结文本
    """
    try:
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 读取文本文件
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 使用组合模型生成摘要
        composite = TextSummaryComposite()
        
        # 生成输出文件名
        base_name = Path(text_path).stem.replace("_transcript", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_{timestamp}_article.md"
        output_path = os.path.join(output_dir, output_filename)
        
        # 使用组合模型生成摘要
        print("开始使用组合模型生成文章...")
        article = composite.generate_summary(content, stream=stream, custom_prompt=custom_prompt, template_path=template_path)
        
        # 保存摘要
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(article)
        
        print("文章生成完成!")
        return output_path
    except Exception as e:
        print(f"文章生成失败: {str(e)}")
        raise Exception(f"文章生成失败: {str(e)}")

class TextSummaryComposite:
    """处理 DeepSeek 和其他 OpenAI 兼容模型的组合，用于文本摘要生成"""
    
    def __init__(self):
        """初始化组合模型"""
        # 从环境变量获取配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
        self.is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "true").lower() == "true"
        
        self.target_api_key = os.getenv("OPENAI_COMPOSITE_API_KEY") or os.getenv("CLAUDE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.target_api_url = os.getenv("OPENAI_COMPOSITE_API_URL") or os.getenv("CLAUDE_API_URL") or "https://api.openai.com/v1"
        self.target_model = os.getenv("OPENAI_COMPOSITE_MODEL") or os.getenv("CLAUDE_MODEL") or "gpt-3.5-turbo"
        
        # 检查必要的API密钥
        if not self.deepseek_api_key:
            raise ValueError("缺少 DeepSeek API 密钥，请在环境变量中设置 DEEPSEEK_API_KEY")
        
        if not self.target_api_key:
            raise ValueError("缺少目标模型 API 密钥，请在环境变量中设置相应的 API 密钥")
    
    def get_short_model_name(self):
        """
        获取目标模型的简短名称，用于文件命名
        :return: 简化的模型名称
        """
        # 从完整模型名称中提取简短名称
        model_name = self.target_model
        
        # 移除路径前缀 (例如 "anthropic/" 或 "google/")
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        
        # 提取主要模型名称 (例如 "claude-3-sonnet" 变为 "claude")
        if "claude" in model_name.lower():
            return "claude"
        elif "gpt" in model_name.lower():
            return "gpt"
        elif "gemini" in model_name.lower():
            return "gemini"
        elif "llama" in model_name.lower():
            return "llama"
        elif "qwen" in model_name.lower():
            return "qwen"
        else:
            # 如果无法识别，返回原始名称的前10个字符
            return model_name[:10].lower()
    
    def generate_summary(self, content, stream=False, custom_prompt=None, template_path=None):
        """
        生成文本摘要
        :param content: 需要摘要的文本内容
        :param stream: 是否使用流式输出
        :param custom_prompt: 自定义提示词，如果提供则使用此提示词代替默认提示词
        :param template_path: 模板文件路径，如果提供则使用此模板
        :return: 生成的摘要文本
        """
        # 准备提示词
        system_prompt = "你是一个专业的内容编辑和文章撰写专家。"
        
        # 使用自定义提示词、模板或默认提示词
        if custom_prompt:
            user_prompt = custom_prompt.format(content=content)
        elif template_path:
            template = load_template(template_path)
            user_prompt = template.format(content=content)
        else:
            template = load_template()
            user_prompt = template.format(content=content)
        
        # 使用 DeepSeek 生成推理过程
        print("1. 使用 DeepSeek 生成推理过程...")
        reasoning = self._get_deepseek_reasoning(system_prompt, user_prompt)
        
        # 使用目标模型生成最终摘要
        print("2. 使用目标模型基于推理过程生成最终文章...")
        if stream:
            return self._get_target_model_summary_stream(system_prompt, user_prompt, reasoning)
        else:
            return self._get_target_model_summary(system_prompt, user_prompt, reasoning)
    
    def _get_deepseek_reasoning(self, system_prompt, user_prompt):
        """
        获取 DeepSeek 的推理过程
        :param system_prompt: 系统提示词
        :param user_prompt: 用户提示词
        :return: 推理过程文本
        """
        try:
            # 准备请求头和数据
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "stream": False
            }
            
            # 发送请求
            import requests
            response = requests.post(
                self.deepseek_api_url,
                headers=headers,
                json=data
            )
            
            # 检查响应
            if response.status_code != 200:
                raise Exception(f"DeepSeek API 请求失败: {response.status_code}, {response.text}")
            
            # 解析响应
            response_data = response.json()
            
            # 提取推理内容
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0]["message"]
                
                # 检查是否有原生推理内容
                if "reasoning_content" in message:
                    return message["reasoning_content"]
                
                # 如果没有原生推理内容，尝试从普通内容中提取
                content = message.get("content", "")
                
                # 尝试从内容中提取 <div className="think-block">...</div> 标签
                import re
                think_match = re.search(r'<div className="think-block">(.*?)</div>', content, re.DOTALL)
                if think_match:
                    return think_match.group(1).strip()
                
                # 如果没有找到标签，则使用完整内容作为推理
                return content
            
            raise Exception("无法从 DeepSeek 响应中提取推理内容")
        
        except Exception as e:
            print(f"获取 DeepSeek 推理过程失败: {str(e)}")
            # 返回一个简单的提示，表示推理过程获取失败
            return "无法获取推理过程，但我会尽力生成一篇高质量的文章。"
    
    def _get_target_model_summary(self, system_prompt, user_prompt, reasoning):
        """
        使用目标模型生成最终摘要
        :param system_prompt: 系统提示词
        :param user_prompt: 用户提示词
        :param reasoning: DeepSeek 的推理过程
        :return: 生成的摘要文本
        """
        try:
            # 创建 OpenAI 客户端
            client = OpenAI(
                api_key=self.target_api_key,
                base_url=self.target_api_url
            )
            
            # 构造结合推理过程的提示词
            combined_prompt = f"""这是我的原始请求：
            
            {user_prompt}
            
            以下是另一个模型的推理过程：
            
            {reasoning}
            
            请基于上述推理过程，提供你的最终文章。直接输出文章内容，不需要解释你的思考过程。
            """
            
            # 发送请求
            response = client.chat.completions.create(
                model=self.target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.7
            )
            
            # 提取回答
            if response.choices and len(response.choices) > 0:
                article = response.choices[0].message.content
                # 清理 Markdown 格式
                cleaned_markdown = clean_markdown_formatting(article)
                return cleaned_markdown
            
            raise Exception("无法从目标模型响应中提取内容")
        
        except Exception as e:
            print(f"获取目标模型摘要失败: {str(e)}")
            # 如果目标模型失败，则返回 DeepSeek 的推理作为备用
            return f"目标模型生成失败，以下是推理过程:\n\n{reasoning}"
    
    def _get_target_model_summary_stream(self, system_prompt, user_prompt, reasoning):
        """
        使用目标模型流式生成最终摘要
        :param system_prompt: 系统提示词
        :param user_prompt: 用户提示词
        :param reasoning: DeepSeek 的推理过程
        :return: 生成的摘要文本
        """
        try:
            # 创建 OpenAI 客户端
            client = OpenAI(
                api_key=self.target_api_key,
                base_url=self.target_api_url
            )
            
            # 构造结合推理过程的提示词
            combined_prompt = f"""这是我的原始请求：
            
            {user_prompt}
            
            以下是另一个模型的推理过程：
            
            {reasoning}
            
            请基于上述推理过程，提供你的最终文章。直接输出文章内容，不需要解释你的思考过程。
            """
            
            # 发送流式请求
            stream_response = client.chat.completions.create(
                model=self.target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.7,
                stream=True
            )
            
            # 收集完整响应
            full_response = ""
            
            print("生成文章中...")
            for chunk in stream_response:
                if not chunk.choices:
                    continue
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    # 打印进度
                    print(".", end="", flush=True)
                    # 收集完整响应
                    full_response += content_chunk
            print("\n文章生成完成!")
            
            # 清理 Markdown 格式
            cleaned_markdown = clean_markdown_formatting(full_response)
            return cleaned_markdown
        
        except Exception as e:
            print(f"获取目标模型流式摘要失败: {str(e)}")
            # 如果目标模型失败，则返回 DeepSeek 的推理作为备用
            return f"目标模型生成失败，以下是推理过程:\n\n{reasoning}"

def process_youtube_video(youtube_url, model=None, api_key=None, base_url=None, whisper_model_size="medium", stream=True, summary_dir="summaries", download_video=False, custom_prompt=None, template_path=None, generate_subtitles=False, translate_to_chinese=True, embed_subtitles=False):
    """
    处理YouTube视频的主函数
    :param youtube_url: YouTube视频链接
    :param model: 使用的模型名称，默认从环境变量获取
    :param api_key: API密钥，默认从环境变量获取
    :param base_url: 自定义API基础URL，默认从环境变量获取
    :param whisper_model_size: Whisper模型大小，默认为medium
    :param stream: 是否使用流式输出生成总结，默认为True
    :param summary_dir: 总结文件保存目录，默认为summaries
    :param download_video: 是否下载视频（True）或仅音频（False），默认为False
    :param custom_prompt: 自定义提示词，如果提供则使用此提示词代替默认提示词
    :param template_path: 模板文件路径，如果提供则使用此模板
    :param generate_subtitles: 是否生成字幕文件，默认为False
    :param translate_to_chinese: 是否将字幕翻译成中文，默认为True
    :param embed_subtitles: 是否将字幕嵌入到视频中，默认为False
    :return: 总结文件的路径
    """
    try:
        print("1. 开始下载YouTube内容...")
        audio_path = None
        
        if download_video:
            print("下载视频（最佳画质）...")
            try:
                # 使用videos目录存储视频
                file_path = download_youtube_video(youtube_url, output_dir="videos", audio_only=False)
                print(f"视频已下载到: {file_path}")
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    raise Exception(f"下载的视频文件不存在: {file_path}")
                
                # 如果下载的是视频，我们需要提取音频
                print("从视频中提取音频...")
                try:
                    audio_path = extract_audio_from_video(file_path, output_dir="downloads")
                    print(f"音频已提取到: {audio_path}")
                except Exception as e:
                    print(f"从视频提取音频失败: {str(e)}")
                    print("尝试直接下载音频作为备选方案...")
                    audio_path = download_youtube_video(youtube_url, output_dir="downloads", audio_only=True)
            except Exception as e:
                print(f"视频下载失败: {str(e)}")
                print("尝试改为下载音频...")
                audio_path = download_youtube_video(youtube_url, output_dir="downloads", audio_only=True)
        else:
            print("仅下载音频...")
            # 使用downloads目录存储音频
            audio_path = download_youtube_video(youtube_url, output_dir="downloads", audio_only=True)
        
        if not audio_path or not os.path.exists(audio_path):
            raise Exception(f"无法获取有效的音频文件")
            
        print(f"音频文件路径: {audio_path}")
        
        print("\n2. 开始转录音频...")
        text_path = transcribe_audio_to_text(audio_path, model_size=whisper_model_size)
        print(f"转录文本已保存到: {text_path}")
        
        # 生成字幕文件
        subtitle_path = None
        video_path = None
        if generate_subtitles:
            print("\n3. 生成字幕文件...")
            subtitle_path = create_bilingual_subtitles(
                audio_path, 
                output_dir="subtitles", 
                model_size=whisper_model_size,
                translate_to_chinese=translate_to_chinese
            )
            if subtitle_path:
                print(f"字幕文件已生成: {subtitle_path}")
                
                # 如果需要嵌入字幕到视频中，并且已下载了视频
                if embed_subtitles and download_video and 'file_path' in locals() and os.path.exists(file_path):
                    video_path = file_path
                    print("\n4. 将字幕嵌入到视频中...")
                    video_with_subtitles = embed_subtitles_to_video(
                        video_path,
                        subtitle_path,
                        output_dir="videos_with_subtitles"
                    )
                    if video_with_subtitles:
                        print(f"带字幕的视频已生成: {video_with_subtitles}")
                    else:
                        print("字幕嵌入失败")
            else:
                print("字幕生成失败")
        
        print("\n5. 开始生成文章...")
        summary_path = summarize_text(
            text_path, 
            model=model, 
            api_key=api_key, 
            base_url=base_url, 
            stream=stream,
            output_dir=summary_dir,
            custom_prompt=custom_prompt,
            template_path=template_path
        )
        print(f"文章已保存到: {summary_path}")
        
        return summary_path
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return None

def process_youtube_videos_batch(youtube_urls, model=None, api_key=None, base_url=None, whisper_model_size="medium", stream=True, summary_dir="summaries", download_video=False, custom_prompt=None, template_path=None, generate_subtitles=False, translate_to_chinese=True, embed_subtitles=False):
    """
    批量处理多个YouTube视频
    :param youtube_urls: YouTube视频链接列表
    :param model: 使用的模型名称，默认从环境变量获取
    :param api_key: API密钥，默认从环境变量获取
    :param base_url: 自定义API基础URL，默认从环境变量获取
    :param whisper_model_size: Whisper模型大小，默认为medium
    :param stream: 是否使用流式输出生成总结，默认为True
    :param summary_dir: 总结文件保存目录，默认为summaries
    :param download_video: 是否下载视频（True）或仅音频（False），默认为False
    :param custom_prompt: 自定义提示词，如果提供则使用此提示词代替默认提示词
    :param template_path: 模板文件路径，如果提供则使用此模板
    :param generate_subtitles: 是否生成字幕文件，默认为False
    :param translate_to_chinese: 是否将字幕翻译成中文，默认为True
    :param embed_subtitles: 是否将字幕嵌入到视频中，默认为False
    :return: 处理结果的字典，键为URL，值为对应的总结文件路径或错误信息
    """
    results = {}
    total_urls = len(youtube_urls)
    
    print(f"开始批量处理 {total_urls} 个YouTube视频...")
    print(f"下载选项: {'完整视频' if download_video else '仅音频'}")
    
    for i, url in enumerate(youtube_urls):
        print(f"\n处理第 {i+1}/{total_urls} 个视频: {url}")
        try:
            summary_path = process_youtube_video(
                url,
                model=model,
                api_key=api_key,
                base_url=base_url,
                whisper_model_size=whisper_model_size,
                stream=stream,
                summary_dir=summary_dir,
                download_video=download_video,  # 确保正确传递download_video参数
                custom_prompt=custom_prompt,
                template_path=template_path,
                generate_subtitles=generate_subtitles,
                translate_to_chinese=translate_to_chinese,
                embed_subtitles=embed_subtitles
            )
            
            if summary_path:
                print(f"视频处理成功: {url}")
                results[url] = {
                    "status": "success",
                    "summary_path": summary_path
                }
            else:
                print(f"视频处理失败: {url}")
                results[url] = {
                    "status": "failed",
                    "error": "处理过程中出现错误，请查看日志获取详细信息"
                }
        except Exception as e:
            print(f"处理视频时出错: {url}")
            print(f"错误详情: {str(e)}")
            results[url] = {
                "status": "failed",
                "error": str(e)
            }
    
    # 打印处理结果统计
    success_count = sum(1 for result in results.values() if result["status"] == "success")
    failed_count = sum(1 for result in results.values() if result["status"] == "failed")
    
    print("\n批量处理完成!")
    print(f"总计: {total_urls} 个视频")
    print(f"成功: {success_count} 个视频")
    print(f"失败: {failed_count} 个视频")
    
    if failed_count > 0:
        print("\n失败的视频:")
        for url, result in results.items():
            if result["status"] == "failed":
                print(f"- {url}: {result['error']}")
    
    return results

def process_local_text(text_path, model=None, api_key=None, base_url=None, stream=True, summary_dir="summaries", custom_prompt=None, template_path=None):
    """
    处理本地文本文件，直接生成摘要和文章
    
    参数:
        text_path (str): 本地文本文件路径
        model (str): 模型名称
        api_key (str): API密钥
        base_url (str): API基础URL
        stream (bool): 是否使用流式输出
        summary_dir (str): 摘要保存目录
        custom_prompt (str): 自定义提示词
        template_path (str): 模板路径
    
    返回:
        str: 生成的文章文件路径
    """
    print(f"正在处理本地文本文件: {text_path}")
    
    # 检查文件是否存在
    if not os.path.exists(text_path):
        print(f"错误: 文件 {text_path} 不存在")
        return None
    
    # 检查文件是否为文本文件
    if not text_path.lower().endswith(('.txt', '.md')):
        print(f"警告: 文件 {text_path} 可能不是文本文件，但仍将尝试处理")
    
    # 直接生成摘要
    summary_file = summarize_text(
        text_path, 
        model=model, 
        api_key=api_key, 
        base_url=base_url, 
        stream=stream, 
        output_dir=summary_dir,
        custom_prompt=custom_prompt,
        template_path=template_path
    )
    
    print(f"文本处理完成，文章已保存至: {summary_file}")
    return summary_file

def create_template(template_name, content=None):
    """
    创建新的模板文件
    :param template_name: 模板名称
    :param content: 模板内容，如果为None则使用默认模板内容
    :return: 模板文件路径
    """
    if not template_name.endswith('.txt'):
        template_name = f"{template_name}.txt"
    
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    
    if content is None:
        content = DEFAULT_TEMPLATE
    
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"模板已创建: {template_path}")
    return template_path

def list_templates():
    """
    列出所有可用的模板
    :return: 模板文件列表
    """
    templates = []
    for file in os.listdir(TEMPLATES_DIR):
        if file.endswith('.txt'):
            templates.append(file)
    
    return templates

def clean_markdown_formatting(markdown_text):
    """
    Clean up markdown formatting issues
    :param markdown_text: Original markdown text
    :return: Cleaned markdown text
    """
    import re
    
    # Split the text into lines for processing
    lines = markdown_text.split('\n')
    result_lines = []
    
    # Track if we're inside a code block
    in_code_block = False
    current_code_language = None
    
    # First, check if the first line is ```markdown and remove it
    if lines and (lines[2].strip() == '```markdown' or lines[2].strip() == '```Markdown' or lines[2].strip() == '``` markdown'):
        lines = lines[3:]  # Remove the first line
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for code block start
        code_block_start = re.match(r'^(\s*)```\s*(\w*)\s*$', line)
        if code_block_start and not in_code_block:
            # Starting a code block
            in_code_block = True
            indent = code_block_start.group(1)
            language = code_block_start.group(2)
            current_code_language = language
            
            # Add the properly formatted code block start
            if language:
                result_lines.append(f"{indent}```{language}")
            else:
                result_lines.append(f"{indent}```")
        
        # Check for code block end
        elif re.match(r'^(\s*)```\s*$', line) and in_code_block:
            # Ending a code block
            in_code_block = False
            current_code_language = None
            result_lines.append(line)
        
        # Check for standalone triple backticks that aren't part of code blocks
        elif re.match(r'^(\s*)```\s*(markdown|Markdown)\s*$', line) and not in_code_block:
            # Skip unnecessary ```markdown markers
            pass
        elif line.strip() == '```' and not in_code_block:
            # Skip standalone closing backticks that aren't closing a code block
            pass
        
        # Regular line, add it to the result
        else:
            result_lines.append(line)
        
        i += 1
    
    # Ensure all code blocks are closed
    if in_code_block:
        result_lines.append("```")
    
    # Remove any trailing empty lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines)

if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从YouTube视频或本地音频/视频文件中提取文本，并生成文章')
    
    # 创建互斥组，用户必须提供YouTube URL或本地音频/视频文件
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--youtube', type=str, help='YouTube视频URL')
    source_group.add_argument('--audio', type=str, help='本地音频文件路径')
    source_group.add_argument('--video', type=str, help='本地视频文件路径')
    source_group.add_argument('--text', type=str, help='本地文本文件路径，直接进行摘要生成')
    source_group.add_argument('--batch', type=str, help='包含多个YouTube URL的文本文件路径，每行一个URL')
    source_group.add_argument('--urls', nargs='+', type=str, help='多个YouTube URL，用空格分隔')
    source_group.add_argument('--create-batch-file', action='store_true', help='创建示例批处理文件')
    source_group.add_argument('--create-template', type=str, help='创建新模板，需要指定模板名称')
    source_group.add_argument('--list-templates', action='store_true', help='列出所有可用的模板')
    
    # 其他参数
    parser.add_argument('--model', type=str, help='使用的模型名称，默认从环境变量获取')
    parser.add_argument('--api-key', type=str, help='API密钥，默认从环境变量获取')
    parser.add_argument('--base-url', type=str, help='自定义API基础URL，默认从环境变量获取')
    parser.add_argument('--whisper-model', type=str, default='small', 
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper模型大小，默认为small')
    parser.add_argument('--no-stream', action='store_true', help='不使用流式输出')
    parser.add_argument('--summary-dir', type=str, default='summaries', help='文章保存目录，默认为summaries')
    parser.add_argument('--download-video', action='store_true', help='下载视频而不仅仅是音频（仅适用于YouTube）')
    parser.add_argument('--batch-file-name', type=str, default='youtube_urls.txt', help='创建示例批处理文件时的文件名')
    parser.add_argument('--prompt', type=str, help='自定义提示词，用于指导文章生成。使用{content}作为占位符表示转录内容')
    parser.add_argument('--template', type=str, help='使用指定的模板文件，可以是模板名称或完整路径')
    parser.add_argument('--template-content', type=str, help='创建模板时的模板内容，仅与--create-template一起使用')
    parser.add_argument('--transcribe-only', action='store_true', help='仅将音频转换为文本，不进行摘要生成')
    # 字幕相关参数
    parser.add_argument('--generate-subtitles', action='store_true', help='生成字幕文件（SRT、VTT和ASS格式）')
    parser.add_argument('--no-translate', action='store_true', help='不将字幕翻译成中文')
    parser.add_argument('--embed-subtitles', action='store_true', help='将字幕嵌入到视频中（仅当下载视频或处理本地视频时有效）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 处理模板路径
    template_path = None
    if args.template:
        # 检查是否是完整路径
        if os.path.exists(args.template):
            template_path = args.template
        else:
            # 检查是否是模板名称
            if not args.template.endswith('.txt'):
                template_name = f"{args.template}.txt"
            else:
                template_name = args.template
            
            potential_path = os.path.join(TEMPLATES_DIR, template_name)
            if os.path.exists(potential_path):
                template_path = potential_path
            else:
                print(f"警告: 找不到模板 '{args.template}'，将使用默认模板")
    
    # 如果用户请求创建示例批处理文件
    if args.create_batch_file:
        create_example_batch_file(args.batch_file_name)
        exit(0)
    
    # 如果用户请求创建新模板
    if args.create_template:
        create_template(args.create_template, args.template_content)
        exit(0)
    
    # 如果用户请求列出所有模板
    if args.list_templates:
        templates = list_templates()
        if templates:
            print("可用的模板:")
            for template in templates:
                print(f"- {template}")
        else:
            print("没有找到可用的模板")
        exit(0)
    
    # 如果没有提供参数，显示帮助信息
    if not (args.youtube or args.audio or args.video or args.text or args.batch or args.urls):
        parser.print_help()
        print("\n示例用法:")
        print("# 处理单个YouTube视频:")
        print("python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id")
        print("python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id --whisper-model large --no-stream")
        print("python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id --download-video")
        
        print("\n# 批量处理多个YouTube视频:")
        print("python youtube_transcriber.py --urls https://www.youtube.com/watch?v=id1 https://www.youtube.com/watch?v=id2")
        print("python youtube_transcriber.py --batch urls.txt  # 文件中每行一个URL")
        print("python youtube_transcriber.py --create-batch-file  # 创建示例批处理文件")
        
        print("\n# 处理本地音频文件:")
        print("python youtube_transcriber.py --audio path/to/your/audio.mp3")
        print("python youtube_transcriber.py --audio path/to/your/audio.mp3 --whisper-model large --summary-dir my_articles")
        
        print("\n# 处理本地视频文件:")
        print("python youtube_transcriber.py --video path/to/your/video.mp4")
        print("python youtube_transcriber.py --video path/to/your/video.mp4 --whisper-model large --summary-dir my_articles")
        print("python youtube_transcriber.py --video path/to/your/video.mp4 --generate-subtitles --embed-subtitles")
        
        print("\n# 处理本地文本文件:")
        print("python youtube_transcriber.py --text path/to/your/text.txt")
        print("python youtube_transcriber.py --text path/to/your/text.txt --summary-dir my_articles")
        
        print("\n# 使用自定义提示词:")
        print('python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id --prompt "请将以下内容总结为一篇新闻报道：\\n\\n{content}"')
        
        print("\n# 使用模板功能:")
        print("python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id --template news")
        print("python youtube_transcriber.py --create-template news --template-content \"请将以下内容改写为新闻报道格式：\\n\\n{content}\"")
        print("python youtube_transcriber.py --list-templates")
    else:
        # 处理自定义提示词
        custom_prompt = args.prompt
        
        # 处理YouTube视频、批量处理或本地音频/视频
        if args.youtube:
            # 处理单个YouTube视频
            if args.transcribe_only:
                summary_path = transcribe_only(download_youtube_video(args.youtube, output_dir="downloads", audio_only=True), whisper_model_size=args.whisper_model, output_dir="transcripts")
            else:
                summary_path = process_youtube_video(
                    args.youtube,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    whisper_model_size=args.whisper_model,
                    stream=not args.no_stream,
                    summary_dir=args.summary_dir,
                    download_video=args.download_video,
                    custom_prompt=custom_prompt,
                    template_path=template_path,
                    generate_subtitles=args.generate_subtitles,
                    translate_to_chinese=not args.no_translate,
                    embed_subtitles=args.embed_subtitles
                )
            
            if summary_path:
                print(f"\n处理完成! 文章已保存到: {summary_path}")
            else:
                print("\n处理失败，请检查错误信息。")
                
        elif args.urls:
            # 直接从命令行处理多个URL
            results = process_youtube_videos_batch(
                args.urls,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                whisper_model_size=args.whisper_model,
                stream=not args.no_stream,
                summary_dir=args.summary_dir,
                download_video=args.download_video,
                custom_prompt=custom_prompt,
                template_path=template_path,
                generate_subtitles=args.generate_subtitles,
                translate_to_chinese=not args.no_translate,
                embed_subtitles=args.embed_subtitles
            )
            
        elif args.batch:
            # 从文件读取URL列表
            try:
                with open(args.batch, 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                
                if not urls:
                    print(f"错误: 文件 {args.batch} 中没有找到有效的URL")
                else:
                    print(f"从文件 {args.batch} 中读取了 {len(urls)} 个URL")
                    results = process_youtube_videos_batch(
                        urls,
                        model=args.model,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        whisper_model_size=args.whisper_model,
                        stream=not args.no_stream,
                        summary_dir=args.summary_dir,
                        download_video=args.download_video,
                        custom_prompt=custom_prompt,
                        template_path=template_path,
                        generate_subtitles=args.generate_subtitles,
                        translate_to_chinese=not args.no_translate,
                        embed_subtitles=args.embed_subtitles
                    )
            except Exception as e:
                print(f"读取批处理文件时出错: {str(e)}")
                
        elif args.video:
            # 处理本地视频文件
            if args.transcribe_only:
                summary_path = transcribe_only(extract_audio_from_video(args.video, output_dir="downloads"), whisper_model_size=args.whisper_model, output_dir="transcripts")
            else:
                summary_path = process_local_video(
                    args.video, 
                    model=args.model, 
                    api_key=args.api_key, 
                    base_url=args.base_url, 
                    whisper_model_size=args.whisper_model, 
                    stream=not args.no_stream, 
                    summary_dir=args.summary_dir,
                    custom_prompt=custom_prompt,
                    template_path=template_path,
                    generate_subtitles=args.generate_subtitles,
                    translate_to_chinese=not args.no_translate,
                    embed_subtitles=args.embed_subtitles
                )
            
            if summary_path:
                print(f"\n处理完成! 文章已保存到: {summary_path}")
            else:
                print("\n处理失败，请检查错误信息。")
                
        elif args.audio:
            # 处理本地音频文件
            if args.transcribe_only:
                summary_path = transcribe_only(args.audio, whisper_model_size=args.whisper_model, output_dir="transcripts")
            else:
                summary_path = process_local_audio(
                    args.audio, 
                    model=args.model, 
                    api_key=args.api_key, 
                    base_url=args.base_url, 
                    whisper_model_size=args.whisper_model, 
                    stream=not args.no_stream, 
                    summary_dir=args.summary_dir,
                    custom_prompt=custom_prompt,
                    template_path=template_path,
                    generate_subtitles=args.generate_subtitles,
                    translate_to_chinese=not args.no_translate
                )
            
            if summary_path:
                print(f"\n处理完成! 文章已保存到: {summary_path}")
            else:
                print("\n处理失败，请检查错误信息。")
                
        elif args.text:
            # 处理本地文本文件
            summary_path = process_local_text(
                args.text,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                stream=not args.no_stream,
                summary_dir=args.summary_dir,
                custom_prompt=custom_prompt,
                template_path=template_path
            )
            
            if summary_path:
                print(f"\n处理完成! 文章已保存到: {summary_path}")
            else:
                print("\n处理失败，请检查错误信息。")
                
        else:
            parser.print_help()