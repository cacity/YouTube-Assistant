"""
YouTube转录工具 - 主程序

此程序整合了所有模块功能，提供命令行接口用于处理YouTube视频和本地音视频文件。
"""

import os
import sys
import argparse
from pathlib import Path

# 导入自定义模块
from youtube_assistant import config
from youtube_assistant.utils.downloader import download_youtube_video, extract_audio_from_video, download_youtube_batch
from youtube_assistant.utils.subtitle_extractor import transcribe_audio_to_text, create_subtitles, create_bilingual_subtitles, embed_subtitles_to_video
from youtube_assistant.utils.translator import translate_text
from youtube_assistant.utils.summarizer import save_summary
from youtube_assistant.utils.composite import TextSummaryComposite
from youtube_assistant.utils.common import ensure_directory, read_text_from_file

def process_local_audio(audio_path, model=None, api_key=None, base_url=None, whisper_model_size="tiny", 
                        stream=True, summary_dir=None, generate_subtitles=True, translate_to_chinese=True,
                        generate_summary=True):
    """
    处理本地音频文件的主函数
    
    参数:
        audio_path (str): 本地音频文件路径
        model (str): 使用的模型名称，默认从环境变量获取
        api_key (str): API密钥，默认从环境变量获取
        base_url (str): API基础URL，默认从环境变量获取
        whisper_model_size (str): Whisper模型大小
        stream (bool): 是否使用流式响应
        summary_dir (str): 摘要输出目录
        generate_subtitles (bool): 是否生成字幕
        translate_to_chinese (bool): 是否翻译成中文
        generate_summary (bool): 是否生成摘要
        
    返回:
        dict: 包含处理结果的字典
    """
    try:
        results = {}
        
        # 首先验证音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        print(f"开始处理音频文件: {audio_path}")
        
        # 转录音频为文本
        transcript_path = transcribe_audio_to_text(
            audio_path, 
            output_dir=config.TRANSCRIPTS_DIR, 
            model_size=whisper_model_size
        )
        results["transcript_path"] = transcript_path
        
        # 读取转录文本
        transcript_text = read_text_from_file(transcript_path)
        
        # 创建摘要生成器实例
        summary_generator = TextSummaryComposite()
        if model:
            summary_generator.target_model = model
        if api_key:
            summary_generator.target_api_key = api_key
        if base_url:
            summary_generator.target_api_url = base_url
            
        # 如果需要生成摘要
        if generate_summary:
            try:
                summary = summary_generator.generate_summary(
                    transcript_text,
                    stream=stream
                )
            except Exception as e:
                print(f"摘要生成失败: {str(e)}")
                summary = f"摘要生成失败: {str(e)}\n\n原始文本:\n{transcript_text[:500]}..."
            
            # 保存摘要
            summary_path = save_summary(
                summary,
                transcript_path=transcript_path,
                output_dir=summary_dir
            )
            results["summary_path"] = summary_path
        
        # 如果需要生成字幕
        if generate_subtitles:
            if translate_to_chinese:
                # 生成双语字幕
                subtitle_path = create_bilingual_subtitles(
                    audio_path,
                    output_dir=config.SUBTITLES_DIR,
                    model_size=whisper_model_size,
                    translate_to_chinese=True
                )
            else:
                # 生成单语字幕
                subtitle_path = create_subtitles(
                    audio_path,
                    output_dir=config.SUBTITLES_DIR,
                    model_size=whisper_model_size
                )
            
            results["subtitle_path"] = subtitle_path
        
        print(f"音频处理完成: {audio_path}")
        return results
    except Exception as e:
        print(f"处理音频文件时出错: {str(e)}")
        raise

def process_local_video(video_path, model=None, api_key=None, base_url=None, whisper_model_size="tiny", 
                        stream=True, summary_dir=None, generate_subtitles=True, translate_to_chinese=True,
                        embed_subtitles=True, generate_summary=True):
    """
    处理本地视频文件的主函数
    
    参数:
        video_path (str): 本地视频文件路径
        model (str): 使用的模型名称，默认从环境变量获取
        api_key (str): API密钥，默认从环境变量获取
        base_url (str): API基础URL，默认从环境变量获取
        whisper_model_size (str): Whisper模型大小
        stream (bool): 是否使用流式响应
        summary_dir (str): 摘要输出目录
        generate_subtitles (bool): 是否生成字幕
        translate_to_chinese (bool): 是否翻译成中文
        embed_subtitles (bool): 是否将字幕嵌入到视频中
        generate_summary (bool): 是否生成摘要
        
    返回:
        dict: 包含处理结果的字典
    """
    try:
        results = {}
        
        # 首先验证视频文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        print(f"开始处理视频文件: {video_path}")
        
        # 从视频中提取音频
        audio_path = extract_audio_from_video(
            video_path,
            output_dir=config.EXTRACTED_AUDIO_DIR
        )
        results["audio_path"] = audio_path
        
        # 处理提取的音频
        audio_results = process_local_audio(
            audio_path,
            model=model,
            api_key=api_key,
            base_url=base_url,
            whisper_model_size=whisper_model_size,
            stream=stream,
            summary_dir=summary_dir,
            generate_subtitles=generate_subtitles,
            translate_to_chinese=translate_to_chinese,
            generate_summary=generate_summary
        )
        
        # 合并结果
        results.update(audio_results)
        
        # 如果需要将字幕嵌入到视频中
        if embed_subtitles and generate_subtitles and "subtitle_path" in audio_results:
            # 将字幕嵌入到视频中
            output_video_path = embed_subtitles_to_video(
                video_path,
                audio_results["subtitle_path"],
                output_dir=config.VIDEOS_WITH_SUBTITLES_DIR
            )
            results["output_video_path"] = output_video_path
        
        print(f"视频处理完成: {video_path}")
        return results
    except Exception as e:
        print(f"处理视频文件时出错: {str(e)}")
        raise

def process_youtube_video(youtube_url, model=None, api_key=None, base_url=None, whisper_model_size="tiny", 
                         stream=True, summary_dir=None, download_video=False, generate_subtitles=True, 
                         translate_to_chinese=True, embed_subtitles=True, generate_summary=True):
    """
    处理YouTube视频的主函数
    
    参数:
        youtube_url (str): YouTube视频链接
        model (str): 使用的模型名称，默认从环境变量获取
        api_key (str): API密钥，默认从环境变量获取
        base_url (str): API基础URL，默认从环境变量获取
        whisper_model_size (str): Whisper模型大小
        stream (bool): 是否使用流式响应
        summary_dir (str): 摘要输出目录
        download_video (bool): 是否下载完整视频，如果为False则只下载音频
        generate_subtitles (bool): 是否生成字幕
        translate_to_chinese (bool): 是否翻译成中文
        embed_subtitles (bool): 是否将字幕嵌入到视频中
        generate_summary (bool): 是否生成摘要
        
    返回:
        dict: 包含处理结果的字典
    """
    try:
        results = {}
        
        print(f"开始处理YouTube视频: {youtube_url}")
        
        # 下载YouTube视频或音频
        if download_video:
            # 下载完整视频
            video_path = download_youtube_video(
                youtube_url,
                output_dir=config.VIDEOS_DIR,
                audio_only=False
            )
            results["video_path"] = video_path
            
            # 处理下载的视频
            video_results = process_local_video(
                video_path,
                model=model,
                api_key=api_key,
                base_url=base_url,
                whisper_model_size=whisper_model_size,
                stream=stream,
                summary_dir=summary_dir,
                generate_subtitles=generate_subtitles,
                translate_to_chinese=translate_to_chinese,
                embed_subtitles=embed_subtitles,
                generate_summary=generate_summary
            )
            
            # 合并结果
            results.update(video_results)
        else:
            # 只下载音频
            audio_path = download_youtube_video(
                youtube_url,
                output_dir=config.DOWNLOADS_DIR,
                audio_only=True
            )
            results["audio_path"] = audio_path
            
            # 处理下载的音频
            audio_results = process_local_audio(
                audio_path,
                model=model,
                api_key=api_key,
                base_url=base_url,
                whisper_model_size=whisper_model_size,
                stream=stream,
                summary_dir=summary_dir,
                generate_subtitles=generate_subtitles,
                translate_to_chinese=translate_to_chinese,
                generate_summary=generate_summary
            )
            
            # 合并结果
            results.update(audio_results)
        
        print(f"YouTube视频处理完成: {youtube_url}")
        return results
    except Exception as e:
        print(f"处理YouTube视频时出错: {str(e)}")
        raise

def process_youtube_videos_batch(youtube_urls, model=None, api_key=None, base_url=None, whisper_model_size="tiny", 
                                stream=True, summary_dir=None, download_video=False, generate_subtitles=True, 
                                translate_to_chinese=True, embed_subtitles=True, generate_summary=True):
    """
    批量处理多个YouTube视频
    
    参数:
        youtube_urls (list): YouTube视频链接列表
        model (str): 使用的模型名称，默认从环境变量获取
        api_key (str): API密钥，默认从环境变量获取
        base_url (str): API基础URL，默认从环境变量获取
        whisper_model_size (str): Whisper模型大小
        stream (bool): 是否使用流式响应
        summary_dir (str): 摘要输出目录
        download_video (bool): 是否下载完整视频，如果为False则只下载音频
        generate_subtitles (bool): 是否生成字幕
        translate_to_chinese (bool): 是否翻译成中文
        embed_subtitles (bool): 是否将字幕嵌入到视频中
        generate_summary (bool): 是否生成摘要
        
    返回:
        list: 包含每个视频处理结果的列表
    """
    results = []
    
    print(f"开始批量处理 {len(youtube_urls)} 个YouTube视频")
    
    for i, url in enumerate(youtube_urls):
        print(f"处理第 {i+1}/{len(youtube_urls)} 个视频: {url}")
        
        try:
            # 处理单个YouTube视频
            result = process_youtube_video(
                url,
                model=model,
                api_key=api_key,
                base_url=base_url,
                whisper_model_size=whisper_model_size,
                stream=stream,
                summary_dir=summary_dir,
                download_video=download_video,
                generate_subtitles=generate_subtitles,
                translate_to_chinese=translate_to_chinese,
                embed_subtitles=embed_subtitles,
                generate_summary=generate_summary
            )
            
            results.append({"url": url, "result": result})
        except Exception as e:
            print(f"处理视频时出错: {url}, 错误: {str(e)}")
            results.append({"url": url, "error": str(e)})
            continue
    
    return results

def main_cli():
    """
    命令行入口函数，处理命令行参数并执行相应的功能
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="YouTube转录工具 - 下载视频、生成字幕和摘要")
    
    # 添加互斥的主要功能参数组
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--youtube", metavar="URL", help="处理单个YouTube视频，需要提供视频链接")
    group.add_argument("--batch", metavar="FILE", help="批量处理YouTube视频，需要提供包含链接的文本文件路径")
    group.add_argument("--audio", metavar="PATH", help="处理本地音频文件，需要提供文件路径")
    group.add_argument("--video", metavar="PATH", help="处理本地视频文件，需要提供文件路径")
    
    # 添加通用选项
    parser.add_argument("--download-video", action="store_true", help="下载完整视频而不仅仅是音频")
    parser.add_argument("--model-size", default="tiny", choices=config.WHISPER_MODEL_SIZES, help="Whisper模型大小")
    parser.add_argument("--no-subtitles", action="store_true", help="不生成字幕")
    parser.add_argument("--no-translation", action="store_true", help="不翻译字幕")
    parser.add_argument("--no-embed", action="store_true", help="不将字幕嵌入到视频中")
    parser.add_argument("--no-summary", action="store_true", help="不生成文本摘要")
    parser.add_argument("--model", help="使用的AI模型，例如gpt-3.5-turbo、claude-3-sonnet等")
    parser.add_argument("--api-key", help="API密钥，如果不提供则从环境变量获取")
    parser.add_argument("--base-url", help="API基础URL，如果不提供则从环境变量获取")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据参数执行相应的功能
    if args.youtube:
        process_youtube_video(
            args.youtube,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            whisper_model_size=args.model_size,
            download_video=args.download_video,
            generate_subtitles=not args.no_subtitles,
            translate_to_chinese=not args.no_translation,
            embed_subtitles=not args.no_embed and args.download_video,
            generate_summary=not args.no_summary
        )
    elif args.batch:
        # 读取链接文件
        with open(args.batch, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        
        process_youtube_videos_batch(
            urls,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            whisper_model_size=args.model_size,
            download_video=args.download_video,
            generate_subtitles=not args.no_subtitles,
            translate_to_chinese=not args.no_translation,
            embed_subtitles=not args.no_embed and args.download_video,
            generate_summary=not args.no_summary
        )
    elif args.audio:
        process_local_audio(
            args.audio,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            whisper_model_size=args.model_size,
            generate_subtitles=not args.no_subtitles,
            translate_to_chinese=not args.no_translation,
            generate_summary=not args.no_summary
        )
    elif args.video:
        process_local_video(
            args.video,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            whisper_model_size=args.model_size,
            generate_subtitles=not args.no_subtitles,
            translate_to_chinese=not args.no_translation,
            embed_subtitles=not args.no_embed,
            generate_summary=not args.no_summary
        )
    else:
        parser.print_help()

def main():
    """
    向后兼容的主函数，调用命令行入口函数
    """
    main_cli()

if __name__ == "__main__":
    main_cli()
