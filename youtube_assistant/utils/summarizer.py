"""
摘要生成器模块 - 负责文本摘要生成

此模块使用AI模型（如GPT、Claude、DeepSeek）生成文本摘要。
"""

import os
import json
import time
from openai import OpenAI
import requests

from youtube_assistant import config
from youtube_assistant.utils.common import ensure_directory, save_text_to_file

def generate_summary(text, model=None, api_key=None, base_url=None, stream=True, template=None):
    """
    生成文本摘要
    
    参数:
        text (str): 要摘要的文本
        model (str): 使用的模型名称，如果为None则从配置中获取
        api_key (str): API密钥，如果为None则从配置中获取
        base_url (str): API基础URL，如果为None则从配置中获取
        stream (bool): 是否使用流式响应
        template (str): 摘要模板，如果为None则使用默认模板
        
    返回:
        str: 生成的摘要
    """
    # 如果未指定模型，则使用OpenAI模型
    if model is None or "gpt" in model.lower():
        return generate_summary_openai(text, model, api_key, base_url, stream, template)
    elif "claude" in model.lower():
        return generate_summary_claude(text, model, api_key, base_url, stream, template)
    elif "deepseek" in model.lower():
        return generate_summary_deepseek(text, model, api_key, base_url, stream, template)
    else:
        # 默认使用OpenAI
        return generate_summary_openai(text, model, api_key, base_url, stream, template)

def generate_summary_openai(text, model=None, api_key=None, base_url=None, stream=True, template=None):
    """
    使用OpenAI API生成文本摘要
    
    参数:
        text (str): 要摘要的文本
        model (str): 使用的模型名称，如果为None则从配置中获取
        api_key (str): API密钥，如果为None则从配置中获取
        base_url (str): API基础URL，如果为None则从配置中获取
        stream (bool): 是否使用流式响应
        template (str): 摘要模板，如果为None则使用默认模板
        
    返回:
        str: 生成的摘要
    """
    try:
        # 获取API密钥和模型名称
        if api_key is None:
            api_key = config.get_api_key('openai')
        
        if model is None:
            model = config.get_model_name('openai')
        
        if base_url is None:
            base_url = config.get_base_url('openai')
        
        if not api_key:
            raise ValueError("未找到OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY")
        
        # 获取模板
        if template is None:
            template = config.load_template()
        
        # 使用模板格式化提示
        prompt = template.format(content=text)
        
        # 创建OpenAI客户端
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        print(f"使用OpenAI模型 {model} 生成摘要...")
        
        # 发送请求
        if stream:
            # 流式响应
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本摘要助手，擅长将长文本转化为结构化的摘要。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                stream=True
            )
            
            # 收集响应
            collected_messages = []
            
            # 打印进度
            print("正在生成摘要...", end="", flush=True)
            
            # 处理流式响应
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_messages.append(content)
                    print(".", end="", flush=True)
            
            print("\n摘要生成完成！")
            
            # 合并所有消息
            full_response = "".join(collected_messages)
            
            return full_response
        else:
            # 非流式响应
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本摘要助手，擅长将长文本转化为结构化的摘要。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            # 提取摘要
            summary = response.choices[0].message.content
            
            print("摘要生成完成！")
            
            return summary
    except Exception as e:
        print(f"OpenAI摘要生成失败: {str(e)}")
        raise

def generate_summary_claude(text, model=None, api_key=None, base_url=None, stream=True, template=None):
    """
    使用Claude API生成文本摘要
    
    参数:
        text (str): 要摘要的文本
        model (str): 使用的模型名称，如果为None则从配置中获取
        api_key (str): API密钥，如果为None则从配置中获取
        stream (bool): 是否使用流式响应
        template (str): 摘要模板，如果为None则使用默认模板
        
    返回:
        str: 生成的摘要
    """
    try:
        # 获取API密钥和模型名称
        if api_key is None:
            api_key = config.get_api_key('claude')
        
        if model is None:
            model = config.get_model_name('claude')
        
        # 获取API基础URL
        if base_url is None:
            base_url = config.get_base_url('claude')
            if base_url is None:
                base_url = "https://api.anthropic.com/v1"
        
        if not api_key:
            raise ValueError("未找到Claude API密钥，请在.env文件中设置CLAUDE_API_KEY")
        
        # 获取模板
        if template is None:
            template = config.load_template()
        
        # 使用模板格式化提示
        prompt = template.format(content=text)
        
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        print(f"使用Claude模型 {model} 生成摘要...")
        
        # 请求参数
        data = {
            "model": model,
            "max_tokens": 4000,
            "system": "你是一个专业的文本摘要助手，擅长将长文本转化为结构化的摘要。",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # 发送请求
        if stream:
            # 流式响应
            data["stream"] = True
            response = requests.post(f"{base_url}/messages", headers=headers, json=data, stream=True)
            
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"Claude API请求失败: {response.status_code}, {response.text}")
            
            # 收集响应
            collected_messages = []
            
            # 打印进度
            print("正在生成摘要...", end="", flush=True)
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]
                        if json_str != "[DONE]":
                            try:
                                chunk = json.loads(json_str)
                                if chunk.get("type") == "content_block_delta" and "delta" in chunk:
                                    delta = chunk["delta"]
                                    if "text" in delta:
                                        collected_messages.append(delta["text"])
                                        print(".", end="", flush=True)
                            except json.JSONDecodeError:
                                pass
            
            print("\n摘要生成完成！")
            
            # 合并所有消息
            full_response = "".join(collected_messages)
            
            return full_response
        else:
            # 非流式响应
            response = requests.post(f"{base_url}/messages", headers=headers, json=data)
            
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"Claude API请求失败: {response.status_code}, {response.text}")
            
            # 解析响应
            result = response.json()
            
            # 提取摘要
            if "content" in result and len(result["content"]) > 0:
                for content_block in result["content"]:
                    if content_block["type"] == "text":
                        summary = content_block["text"]
                        print("摘要生成完成！")
                        return summary
                
                raise Exception("Claude API返回的结果中没有文本内容")
            else:
                raise Exception("Claude API返回的结果格式不正确")
    except Exception as e:
        print(f"Claude摘要生成失败: {str(e)}")
        # 如果Claude摘要失败，尝试使用OpenAI
        print("尝试使用OpenAI生成摘要...")
        return generate_summary_openai(text, None, None, base_url, stream, template)

def generate_summary_deepseek(text, model=None, api_key=None, base_url=None, stream=True, template=None):
    """
    使用DeepSeek API生成文本摘要
    
    参数:
        text (str): 要摘要的文本
        model (str): 使用的模型名称，如果为None则从配置中获取
        api_key (str): API密钥，如果为None则从配置中获取
        stream (bool): 是否使用流式响应
        template (str): 摘要模板，如果为None则使用默认模板
        
    返回:
        str: 生成的摘要
    """
    try:
        # 获取API密钥和模型名称
        if api_key is None:
            api_key = config.get_api_key('deepseek')
        
        if model is None:
            model = config.get_model_name('deepseek')
        
        if not api_key:
            raise ValueError("未找到DeepSeek API密钥，请在.env文件中设置DEEPSEEK_API_KEY")
        
        # 获取模板
        if template is None:
            template = config.load_template()
        
        # 使用模板格式化提示
        prompt = template.format(content=text)
        
        # DeepSeek API URL
        url = "https://api.deepseek.com/v1/chat/completions"
        
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 请求参数
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个专业的文本摘要助手，擅长将长文本转化为结构化的摘要。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "stream": stream
        }
        
        print(f"使用DeepSeek模型 {model} 生成摘要...")
        
        # 发送请求
        if stream:
            # 流式响应
            response = requests.post(url, headers=headers, json=data, stream=True)
            
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"DeepSeek API请求失败: {response.status_code}, {response.text}")
            
            # 收集响应
            collected_messages = []
            
            # 打印进度
            print("正在生成摘要...", end="", flush=True)
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]
                        if json_str != "[DONE]":
                            try:
                                chunk = json.loads(json_str)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        collected_messages.append(delta["content"])
                                        print(".", end="", flush=True)
                            except json.JSONDecodeError:
                                pass
            
            print("\n摘要生成完成！")
            
            # 合并所有消息
            full_response = "".join(collected_messages)
            
            return full_response
        else:
            # 非流式响应
            data["stream"] = False
            response = requests.post(url, headers=headers, json=data)
            
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"DeepSeek API请求失败: {response.status_code}, {response.text}")
            
            # 解析响应
            result = response.json()
            
            # 提取摘要
            if "choices" in result and len(result["choices"]) > 0:
                summary = result["choices"][0]["message"]["content"]
                
                print("摘要生成完成！")
                
                return summary
            else:
                raise Exception("DeepSeek API返回的结果格式不正确")
    except Exception as e:
        print(f"DeepSeek摘要生成失败: {str(e)}")
        # 如果DeepSeek摘要失败，尝试使用OpenAI
        print("尝试使用OpenAI生成摘要...")
        return generate_summary_openai(text, None, None, base_url, stream, template)

def save_summary(summary, transcript_path=None, output_dir=None):
    """
    保存摘要到文件
    
    参数:
        summary (str): 生成的摘要
        transcript_path (str): 转录文本文件路径，用于生成摘要文件名
        output_dir (str): 输出目录，如果为None则使用默认目录
        
    返回:
        str: 摘要文件路径
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = config.SUMMARIES_DIR
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    try:
        # 生成摘要文件名
        if transcript_path:
            # 从转录文件路径获取文件名
            transcript_filename = os.path.basename(transcript_path)
            base_name = os.path.splitext(transcript_filename)[0].replace("_transcript", "")
            summary_filename = f"{base_name}_summary.md"
        else:
            # 使用当前时间作为文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"summary_{timestamp}.md"
        
        # 设置摘要文件路径
        summary_path = os.path.join(output_dir, summary_filename)
        
        # 保存摘要
        save_text_to_file(summary, summary_path)
        
        print(f"摘要已保存到: {summary_path}")
        return summary_path
    except Exception as e:
        print(f"保存摘要失败: {str(e)}")
        return None
