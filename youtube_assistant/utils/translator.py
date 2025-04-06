"""
翻译器模块 - 负责文本翻译功能

此模块提供多种翻译服务接口，包括Google翻译、DeepL等。
"""

import requests
import html
import os
import json
from openai import OpenAI

from youtube_assistant import config

def translate_text(text, target_language='zh-CN', source_language='auto', service='google'):
    """
    翻译文本
    
    参数:
        text (str): 要翻译的文本
        target_language (str): 目标语言代码
        source_language (str): 源语言代码，默认为自动检测
        service (str): 翻译服务，可选值: google, openai, deepl
        
    返回:
        str: 翻译后的文本
    """
    if not text or text.strip() == "":
        return ""
        
    if service.lower() == 'google':
        return translate_text_google(text, target_language, source_language)
    elif service.lower() == 'openai':
        return translate_text_openai(text, target_language, source_language)
    elif service.lower() == 'deepl':
        return translate_text_deepl(text, target_language, source_language)
    else:
        raise ValueError(f"不支持的翻译服务: {service}")

def translate_text_google(text, target_language='zh-CN', source_language='auto'):
    """
    使用Google翻译API翻译文本
    
    参数:
        text (str): 要翻译的文本
        target_language (str): 目标语言代码，默认为中文
        source_language (str): 源语言代码，默认为自动检测
        
    返回:
        str: 翻译后的文本
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

def translate_text_openai(text, target_language='zh-CN', source_language='auto'):
    """
    使用OpenAI API翻译文本
    
    参数:
        text (str): 要翻译的文本
        target_language (str): 目标语言代码，默认为中文
        source_language (str): 源语言代码，默认为自动检测
        
    返回:
        str: 翻译后的文本
    """
    try:
        # 获取API密钥和模型名称
        api_key = config.get_api_key('openai')
        model = config.get_model_name('openai')
        base_url = config.get_base_url('openai')
        
        if not api_key:
            raise ValueError("未找到OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY")
        
        # 创建OpenAI客户端
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 获取目标语言的完整名称
        language_map = {
            'zh-CN': '中文',
            'en': '英语',
            'ja': '日语',
            'ko': '韩语',
            'fr': '法语',
            'de': '德语',
            'es': '西班牙语',
            'it': '意大利语',
            'ru': '俄语',
            'pt': '葡萄牙语',
            'nl': '荷兰语',
            'ar': '阿拉伯语',
            'hi': '印地语',
            'bn': '孟加拉语',
            'vi': '越南语',
            'th': '泰语',
            'id': '印尼语',
            'ms': '马来语'
        }
        
        target_language_name = language_map.get(target_language, target_language)
        
        # 构建提示
        prompt = f"请将以下文本翻译成{target_language_name}，只需要返回翻译结果，不要添加任何解释或其他内容：\n\n{text}"
        
        # 发送请求
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的翻译助手，请准确翻译用户提供的文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        # 提取翻译结果
        translated_text = response.choices[0].message.content.strip()
        
        return translated_text
    except Exception as e:
        print(f"OpenAI翻译过程中出错: {str(e)}")
        # 如果OpenAI翻译失败，回退到Google翻译
        print("回退到Google翻译...")
        return translate_text_google(text, target_language, source_language)

def translate_text_deepl(text, target_language='ZH', source_language=None):
    """
    使用DeepL API翻译文本
    
    参数:
        text (str): 要翻译的文本
        target_language (str): 目标语言代码，DeepL使用的是大写语言代码，如ZH, EN-US
        source_language (str): 源语言代码，默认为None（自动检测）
        
    返回:
        str: 翻译后的文本
    """
    try:
        # 获取API密钥
        api_key = os.getenv("DEEPL_API_KEY")
        
        if not api_key:
            raise ValueError("未找到DeepL API密钥，请在.env文件中设置DEEPL_API_KEY")
        
        # 语言代码映射（将标准语言代码转换为DeepL使用的格式）
        language_map = {
            'zh-CN': 'ZH',
            'en': 'EN-US',
            'ja': 'JA',
            'ko': 'KO',
            'fr': 'FR',
            'de': 'DE',
            'es': 'ES',
            'it': 'IT',
            'ru': 'RU',
            'pt': 'PT-BR',
            'nl': 'NL',
            'auto': None
        }
        
        # 转换语言代码
        deepl_target = language_map.get(target_language, target_language)
        deepl_source = language_map.get(source_language, source_language) if source_language != 'auto' else None
        
        # DeepL API URL
        url = "https://api-free.deepl.com/v2/translate"
        
        # 请求头
        headers = {
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json"
        }
        
        # 请求参数
        data = {
            "text": [text],
            "target_lang": deepl_target
        }
        
        # 如果指定了源语言，则添加到请求参数中
        if deepl_source:
            data["source_lang"] = deepl_source
        
        # 发送请求
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            # 解析响应
            result = response.json()
            
            # 提取翻译文本
            if "translations" in result and len(result["translations"]) > 0:
                translated_text = result["translations"][0]["text"]
                return translated_text
            else:
                print("DeepL返回的翻译结果格式不正确")
                return text
        else:
            print(f"DeepL翻译请求失败: {response.status_code}, {response.text}")
            return text
    except Exception as e:
        print(f"DeepL翻译过程中出错: {str(e)}")
        # 如果DeepL翻译失败，回退到Google翻译
        print("回退到Google翻译...")
        return translate_text_google(text, target_language, source_language)

def batch_translate(texts, target_language='zh-CN', source_language='auto', service='google'):
    """
    批量翻译文本
    
    参数:
        texts (list): 要翻译的文本列表
        target_language (str): 目标语言代码
        source_language (str): 源语言代码，默认为自动检测
        service (str): 翻译服务，可选值: google, openai, deepl
        
    返回:
        list: 翻译后的文本列表
    """
    translated_texts = []
    
    for i, text in enumerate(texts):
        try:
            print(f"翻译第 {i+1}/{len(texts)} 个文本...")
            translated = translate_text(text, target_language, source_language, service)
            translated_texts.append(translated)
        except Exception as e:
            print(f"翻译第 {i+1} 个文本时出错: {str(e)}")
            # 如果翻译失败，保留原文
            translated_texts.append(text)
    
    return translated_texts
