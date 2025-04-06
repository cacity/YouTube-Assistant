"""
安装脚本 - 用于向后兼容

此文件包含与pyproject.toml相同的项目信息，用于支持传统的setuptools安装方式。
"""

from setuptools import setup, find_packages

setup(
    name="youtube-assistant",
    version="2.1.0",
    description="一个功能强大的模块化工具，可以从YouTube视频中提取音频，转录为文本，生成字幕，翻译字幕，并使用AI生成高质量文章摘要",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license='MIT',
    license_files='LICENSE',
    author="cacity",
    author_email="gf7823332@gmail.com",
    url="https://github.com/cacity/youtube-assistant",
    packages=find_packages(),
    install_requires=[
        "yt-dlp",
        "openai-whisper",
        "requests",
        "python-dotenv",
        "torch",
        "ffmpeg-python",
        "tqdm",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "youtube-assistant=youtube_assistant.main:main_cli",
        ],
    },
)
