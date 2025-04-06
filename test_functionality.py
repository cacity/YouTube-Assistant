#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YouTube转录工具功能测试脚本
该脚本用于测试YouTube转录工具的各项功能是否正常工作
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
from datetime import datetime

# 测试结果状态
STATUS_PASS = "✅ 通过"
STATUS_FAIL = "❌ 失败"
STATUS_SKIP = "⚠️ 跳过"

class TestResult:
    def __init__(self, feature, status, message=""):
        self.feature = feature
        self.status = status
        self.message = message
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_command(command, timeout=300):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "命令执行超时",
            "success": False
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }

def check_directory_for_files(directory, extension=None):
    """检查目录中是否有文件，可以指定扩展名"""
    if not os.path.exists(directory):
        return False
    
    files = os.listdir(directory)
    if extension:
        files = [f for f in files if f.endswith(extension)]
    
    return len(files) > 0

def test_basic_functionality(youtube_url):
    """测试基本功能：下载音频并生成字幕和摘要"""
    print(f"测试基本功能：下载音频并生成字幕和摘要...")
    
    # 清理之前的测试文件
    for directory in ["downloads", "transcripts", "summaries", "subtitles"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"清理文件时出错: {e}")
    
    # 运行命令
    command = f"python main.py --youtube {youtube_url}"
    result = run_command(command, timeout=600)  # 10分钟超时
    
    # 检查结果
    if not result["success"]:
        return TestResult("基本功能", STATUS_FAIL, f"命令执行失败: {result['stderr']}")
    
    # 检查是否生成了相应的文件
    has_audio = check_directory_for_files("downloads", ".mp3")
    has_transcript = check_directory_for_files("transcripts", ".txt")
    has_summary = check_directory_for_files("summaries", ".md")
    has_subtitles = check_directory_for_files("subtitles", ".srt")
    
    if has_audio and has_transcript and has_summary and has_subtitles:
        return TestResult("基本功能", STATUS_PASS, "成功下载音频并生成字幕和摘要")
    else:
        missing = []
        if not has_audio: missing.append("音频文件")
        if not has_transcript: missing.append("转录文本")
        if not has_summary: missing.append("摘要文件")
        if not has_subtitles: missing.append("字幕文件")
        
        return TestResult("基本功能", STATUS_FAIL, f"缺少以下文件: {', '.join(missing)}")

def test_no_subtitles(youtube_url):
    """测试不生成字幕功能"""
    print(f"测试不生成字幕功能...")
    
    # 清理之前的测试文件
    for directory in ["downloads", "transcripts", "summaries", "subtitles"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"清理文件时出错: {e}")
    
    # 运行命令
    command = f"python main.py --youtube {youtube_url} --no-subtitles"
    result = run_command(command, timeout=600)
    
    # 检查结果
    if not result["success"]:
        return TestResult("不生成字幕", STATUS_FAIL, f"命令执行失败: {result['stderr']}")
    
    # 检查是否生成了相应的文件
    has_audio = check_directory_for_files("downloads", ".mp3")
    has_transcript = check_directory_for_files("transcripts", ".txt")
    has_summary = check_directory_for_files("summaries", ".md")
    has_subtitles = check_directory_for_files("subtitles", ".srt")
    
    if has_audio and has_transcript and has_summary and not has_subtitles:
        return TestResult("不生成字幕", STATUS_PASS, "成功下载音频并生成摘要，没有生成字幕")
    else:
        if has_subtitles:
            return TestResult("不生成字幕", STATUS_FAIL, "虽然指定了--no-subtitles参数，但仍然生成了字幕文件")
        
        missing = []
        if not has_audio: missing.append("音频文件")
        if not has_transcript: missing.append("转录文本")
        if not has_summary: missing.append("摘要文件")
        
        return TestResult("不生成字幕", STATUS_FAIL, f"缺少以下文件: {', '.join(missing)}")

def test_no_translation(youtube_url):
    """测试不翻译字幕功能"""
    print(f"测试不翻译字幕功能...")
    
    # 清理之前的测试文件
    for directory in ["downloads", "transcripts", "summaries", "subtitles"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"清理文件时出错: {e}")
    
    # 运行命令
    command = f"python main.py --youtube {youtube_url} --no-translation"
    result = run_command(command, timeout=600)
    
    # 检查结果
    if not result["success"]:
        return TestResult("不翻译字幕", STATUS_FAIL, f"命令执行失败: {result['stderr']}")
    
    # 检查是否生成了相应的文件
    has_audio = check_directory_for_files("downloads", ".mp3")
    has_transcript = check_directory_for_files("transcripts", ".txt")
    has_summary = check_directory_for_files("summaries", ".md")
    has_subtitles = check_directory_for_files("subtitles", ".srt")
    
    # 检查是否有双语字幕文件
    has_bilingual_subtitles = False
    if os.path.exists("subtitles"):
        for file in os.listdir("subtitles"):
            if "_bilingual" in file:
                has_bilingual_subtitles = True
                break
    
    if has_audio and has_transcript and has_summary and has_subtitles and not has_bilingual_subtitles:
        return TestResult("不翻译字幕", STATUS_PASS, "成功下载音频并生成摘要和字幕，没有生成双语字幕")
    else:
        if has_bilingual_subtitles:
            return TestResult("不翻译字幕", STATUS_FAIL, "虽然指定了--no-translation参数，但仍然生成了双语字幕文件")
        
        missing = []
        if not has_audio: missing.append("音频文件")
        if not has_transcript: missing.append("转录文本")
        if not has_summary: missing.append("摘要文件")
        if not has_subtitles: missing.append("字幕文件")
        
        return TestResult("不翻译字幕", STATUS_FAIL, f"缺少以下文件: {', '.join(missing)}")

def test_no_summary(youtube_url):
    """测试不生成摘要功能"""
    print(f"测试不生成摘要功能...")
    
    # 清理之前的测试文件
    for directory in ["downloads", "transcripts", "summaries", "subtitles"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"清理文件时出错: {e}")
    
    # 运行命令
    command = f"python main.py --youtube {youtube_url} --no-summary"
    result = run_command(command, timeout=600)
    
    # 检查结果
    if not result["success"]:
        return TestResult("不生成摘要", STATUS_FAIL, f"命令执行失败: {result['stderr']}")
    
    # 检查是否生成了相应的文件
    has_audio = check_directory_for_files("downloads", ".mp3")
    has_transcript = check_directory_for_files("transcripts", ".txt")
    has_summary = check_directory_for_files("summaries", ".md")
    has_subtitles = check_directory_for_files("subtitles", ".srt")
    
    if has_audio and has_transcript and not has_summary and has_subtitles:
        return TestResult("不生成摘要", STATUS_PASS, "成功下载音频并生成字幕，没有生成摘要")
    else:
        if has_summary:
            return TestResult("不生成摘要", STATUS_FAIL, "虽然指定了--no-summary参数，但仍然生成了摘要文件")
        
        missing = []
        if not has_audio: missing.append("音频文件")
        if not has_transcript: missing.append("转录文本")
        if not has_subtitles: missing.append("字幕文件")
        
        return TestResult("不生成摘要", STATUS_FAIL, f"缺少以下文件: {', '.join(missing)}")

def test_download_video(youtube_url):
    """测试下载完整视频功能"""
    print(f"测试下载完整视频功能...")
    
    # 清理之前的测试文件
    for directory in ["downloads", "transcripts", "summaries", "subtitles", "videos", "videos_with_subtitles"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"清理文件时出错: {e}")
    
    # 运行命令
    command = f"python main.py --youtube {youtube_url} --download-video"
    result = run_command(command, timeout=900)  # 15分钟超时
    
    # 检查结果
    if not result["success"]:
        return TestResult("下载完整视频", STATUS_FAIL, f"命令执行失败: {result['stderr']}")
    
    # 检查是否生成了相应的文件
    has_video = check_directory_for_files("videos", ".mp4")
    has_transcript = check_directory_for_files("transcripts", ".txt")
    has_summary = check_directory_for_files("summaries", ".md")
    has_subtitles = check_directory_for_files("subtitles", ".srt")
    has_video_with_subtitles = check_directory_for_files("videos_with_subtitles", ".mp4")
    
    if has_video and has_transcript and has_summary and has_subtitles and has_video_with_subtitles:
        return TestResult("下载完整视频", STATUS_PASS, "成功下载视频并生成字幕、摘要，以及嵌入字幕的视频")
    else:
        missing = []
        if not has_video: missing.append("视频文件")
        if not has_transcript: missing.append("转录文本")
        if not has_summary: missing.append("摘要文件")
        if not has_subtitles: missing.append("字幕文件")
        if not has_video_with_subtitles: missing.append("嵌入字幕的视频")
        
        return TestResult("下载完整视频", STATUS_FAIL, f"缺少以下文件: {', '.join(missing)}")

def test_no_embed(youtube_url):
    """测试不嵌入字幕功能"""
    print(f"测试不嵌入字幕功能...")
    
    # 清理之前的测试文件
    for directory in ["downloads", "transcripts", "summaries", "subtitles", "videos", "videos_with_subtitles"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"清理文件时出错: {e}")
    
    # 运行命令
    command = f"python main.py --youtube {youtube_url} --download-video --no-embed"
    result = run_command(command, timeout=900)
    
    # 检查结果
    if not result["success"]:
        return TestResult("不嵌入字幕", STATUS_FAIL, f"命令执行失败: {result['stderr']}")
    
    # 检查是否生成了相应的文件
    has_video = check_directory_for_files("videos", ".mp4")
    has_transcript = check_directory_for_files("transcripts", ".txt")
    has_summary = check_directory_for_files("summaries", ".md")
    has_subtitles = check_directory_for_files("subtitles", ".srt")
    has_video_with_subtitles = check_directory_for_files("videos_with_subtitles", ".mp4")
    
    if has_video and has_transcript and has_summary and has_subtitles and not has_video_with_subtitles:
        return TestResult("不嵌入字幕", STATUS_PASS, "成功下载视频并生成字幕、摘要，没有嵌入字幕")
    else:
        if has_video_with_subtitles:
            return TestResult("不嵌入字幕", STATUS_FAIL, "虽然指定了--no-embed参数，但仍然生成了嵌入字幕的视频")
        
        missing = []
        if not has_video: missing.append("视频文件")
        if not has_transcript: missing.append("转录文本")
        if not has_summary: missing.append("摘要文件")
        if not has_subtitles: missing.append("字幕文件")
        
        return TestResult("不嵌入字幕", STATUS_FAIL, f"缺少以下文件: {', '.join(missing)}")

def generate_report(results):
    """生成测试报告"""
    report = f"""
# YouTube转录工具功能测试报告

测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 测试结果摘要

| 功能 | 状态 | 备注 |
|------|------|------|
"""
    
    for result in results:
        report += f"| {result.feature} | {result.status} | {result.message} |\n"
    
    report += """
## 详细说明

### 测试环境
- 操作系统: Windows
- Python版本: """ + sys.version.split()[0] + """
- 测试时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

### 测试项目说明
1. **基本功能**: 测试下载音频并生成字幕和摘要的基本功能
2. **不生成字幕**: 测试使用--no-subtitles参数时是否正确跳过字幕生成
3. **不翻译字幕**: 测试使用--no-translation参数时是否正确跳过字幕翻译
4. **不生成摘要**: 测试使用--no-summary参数时是否正确跳过摘要生成
5. **下载完整视频**: 测试使用--download-video参数时是否正确下载完整视频并嵌入字幕
6. **不嵌入字幕**: 测试使用--no-embed参数时是否正确跳过字幕嵌入

### 测试结论
"""
    
    # 统计通过、失败和跳过的数量
    pass_count = sum(1 for r in results if r.status == STATUS_PASS)
    fail_count = sum(1 for r in results if r.status == STATUS_FAIL)
    skip_count = sum(1 for r in results if r.status == STATUS_SKIP)
    total_count = len(results)
    
    if fail_count == 0:
        report += f"所有测试项目均通过，功能正常。共测试{total_count}项功能，通过{pass_count}项，跳过{skip_count}项。"
    else:
        report += f"测试未全部通过，请检查失败项目。共测试{total_count}项功能，通过{pass_count}项，失败{fail_count}项，跳过{skip_count}项。"
    
    return report

def main():
    parser = argparse.ArgumentParser(description="YouTube转录工具功能测试脚本")
    parser.add_argument("youtube_url", help="用于测试的YouTube视频URL")
    parser.add_argument("--skip-video-tests", action="store_true", help="跳过需要下载完整视频的测试")
    args = parser.parse_args()
    
    print(f"开始测试YouTube转录工具功能...")
    print(f"测试URL: {args.youtube_url}")
    
    results = []
    
    # 测试基本功能
    results.append(test_basic_functionality(args.youtube_url))
    
    # 测试不生成字幕功能
    results.append(test_no_subtitles(args.youtube_url))
    
    # 测试不翻译字幕功能
    results.append(test_no_translation(args.youtube_url))
    
    # 测试不生成摘要功能
    results.append(test_no_summary(args.youtube_url))
    
    # 测试下载完整视频功能
    if not args.skip_video_tests:
        results.append(test_download_video(args.youtube_url))
        
        # 测试不嵌入字幕功能
        results.append(test_no_embed(args.youtube_url))
    else:
        results.append(TestResult("下载完整视频", STATUS_SKIP, "根据参数跳过此测试"))
        results.append(TestResult("不嵌入字幕", STATUS_SKIP, "根据参数跳过此测试"))
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    with open("test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n测试完成，报告已保存到 test_report.md")
    
    # 打印摘要
    pass_count = sum(1 for r in results if r.status == STATUS_PASS)
    fail_count = sum(1 for r in results if r.status == STATUS_FAIL)
    skip_count = sum(1 for r in results if r.status == STATUS_SKIP)
    total_count = len(results)
    
    print(f"\n测试结果摘要:")
    print(f"总测试项: {total_count}")
    print(f"通过: {pass_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {skip_count}")
    
    # 如果有失败的测试，返回非零退出码
    return 1 if fail_count > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
