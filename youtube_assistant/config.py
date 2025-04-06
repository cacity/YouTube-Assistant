"""
配置模块 - 管理API密钥和模型设置

此模块负责加载环境变量、设置默认配置和提供配置访问接口。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 基础目录结构
# 使用用户数据目录来存储下载和生成的文件
USER_HOME = os.path.expanduser("~")
USER_DATA_DIR = os.path.join(USER_HOME, ".youtube_assistant")
DOWNLOADS_DIR = os.path.join(USER_DATA_DIR, "downloads")
VIDEOS_DIR = os.path.join(USER_DATA_DIR, "videos")
EXTRACTED_AUDIO_DIR = os.path.join(USER_DATA_DIR, "extracted_audio")
SUBTITLES_DIR = os.path.join(USER_DATA_DIR, "subtitles")
TRANSCRIPTS_DIR = os.path.join(USER_DATA_DIR, "transcripts")
SUMMARIES_DIR = os.path.join(USER_DATA_DIR, "summaries")
VIDEOS_WITH_SUBTITLES_DIR = os.path.join(USER_DATA_DIR, "videos_with_subtitles")
TEMPLATES_DIR = os.path.join(USER_DATA_DIR, "templates")

# 确保所有目录存在
for directory in [DOWNLOADS_DIR, VIDEOS_DIR, EXTRACTED_AUDIO_DIR, SUBTITLES_DIR, 
                 TRANSCRIPTS_DIR, SUMMARIES_DIR, VIDEOS_WITH_SUBTITLES_DIR, TEMPLATES_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# API密钥配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")

# 模型配置
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")

# API基础URL配置
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1")

# Whisper模型配置
DEFAULT_WHISPER_MODEL = "tiny"  # 可选: tiny, base, small, medium, large
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]

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

def get_api_key(service="openai"):
    """
    获取指定服务的API密钥
    
    参数:
        service (str): 服务名称，可选值: openai, deepseek, claude
        
    返回:
        str: API密钥
    """
    if service.lower() == "openai":
        return OPENAI_API_KEY
    elif service.lower() == "deepseek":
        return DEEPSEEK_API_KEY
    elif service.lower() == "claude":
        return CLAUDE_API_KEY
    else:
        raise ValueError(f"不支持的服务: {service}")

def get_model_name(service="openai"):
    """
    获取指定服务的默认模型名称
    
    参数:
        service (str): 服务名称，可选值: openai, deepseek, claude
        
    返回:
        str: 模型名称
    """
    if service.lower() == "openai":
        return OPENAI_MODEL
    elif service.lower() == "deepseek":
        return DEEPSEEK_MODEL
    elif service.lower() == "claude":
        return CLAUDE_MODEL
    else:
        raise ValueError(f"不支持的服务: {service}")

def get_base_url(service="openai"):
    """
    获取指定服务的API基础URL
    
    参数:
        service (str): 服务名称，可选值: openai, claude
        
    返回:
        str: API基础URL
    """
    if service.lower() == "openai":
        return OPENAI_BASE_URL
    elif service.lower() == "claude":
        return CLAUDE_API_URL
    else:
        raise ValueError(f"不支持的服务: {service}")

def load_template(template_path=None):
    """
    加载模板文件
    
    参数:
        template_path (str): 模板文件路径，如果为None则使用默认模板
        
    返回:
        str: 模板内容
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
