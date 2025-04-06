"""
组合模型处理模块 - 用于处理不同API的组合调用
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from .summarizer import generate_summary_openai, generate_summary_claude, generate_summary_deepseek

# 加载环境变量
load_dotenv()

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
        if not self.deepseek_api_key and not self.target_api_key:
            print("警告: 未找到任何API密钥，将使用默认模型")
    
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
        # 如果没有API密钥，直接返回错误信息
        if not self.target_api_key:
            raise ValueError("未找到任何API密钥，请在.env文件中设置相应的API密钥")
        
        # 确定使用哪个模型
        model_name = self.target_model.lower()
        
        # 根据模型类型调用不同的生成函数
        if "claude" in model_name:
            return generate_summary_claude(
                content, 
                model=self.target_model, 
                api_key=self.target_api_key, 
                base_url=self.target_api_url,
                stream=stream, 
                template=custom_prompt
            )
        elif "deepseek" in model_name:
            return generate_summary_deepseek(
                content, 
                model=self.target_model, 
                api_key=self.deepseek_api_key, 
                stream=stream, 
                template=custom_prompt
            )
        else:  # 默认使用OpenAI
            return generate_summary_openai(
                content, 
                model=self.target_model, 
                api_key=self.target_api_key, 
                base_url=self.target_api_url,
                stream=stream, 
                template=custom_prompt
            )
