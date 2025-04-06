# YouTube Assistant 📺 ➡️ 📝

![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-yt--dlp%20%7C%20whisper%20%7C%20openai-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Last Commit](https://img.shields.io/badge/Last%20Commit-April%202025-yellowgreen)
![Code Size](https://img.shields.io/badge/Code%20Size-Lightweight-blue)

一个功能强大的模块化工具，可以从YouTube视频中提取音频，转录为文本，生成字幕，翻译字幕，并使用AI生成高质量文章摘要。

## ✨ 主要功能

- 🎬 从YouTube下载视频或仅下载音频
- 🔊 从本地视频文件中提取音频
- 🎯 使用OpenAI Whisper模型进行高精度音频转录
- 📝 利用大语言模型（GPT、Claude、DeepSeek等）生成高质量文章摘要
- 🔄 支持批量处理多个YouTube视频
- 🌐 支持多种字幕格式（SRT、WebVTT、ASS）
- 🔠 支持自动翻译字幕，生成双语字幕
- 📋 支持自定义模板，灵活控制文章生成风格
- 📺 支持将字幕嵌入到视频中
- 🎵 支持仅将音频转换为文本，不进行文章生成

## 🚀 快速开始

### 安装依赖

```bash
# 使用 pip 安装所有依赖
# 这将安装 yt-dlp、whisper、torch 等必要库
pip install -r requirements.txt
```

### 环境变量设置

创建一个`.env`文件，包含以下内容：

```
# API密钥配置（至少需要配置一个）
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# 模型配置
OPENAI_MODEL=gpt-3.5-turbo
CLAUDE_MODEL=claude-3-sonnet-20240229
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-R1

# API基础URL配置
OPENAI_BASE_URL=https://api.openai.com/v1
CLAUDE_API_URL=https://api.anthropic.com/v1
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

# 组合模型配置（优先级高于单独配置）
OPENAI_COMPOSITE_API_KEY=your_composite_api_key
OPENAI_COMPOSITE_MODEL=gpt-4-turbo
OPENAI_COMPOSITE_API_URL=https://api.openai.com/v1
```

### 基本用法

#### 处理单个YouTube视频

```bash
# 下载音频并生成字幕和摘要
python main.py --youtube https://www.youtube.com/watch?v=your_video_id

# 下载完整视频并嵌入字幕
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --download-video
```

#### 处理本地视频文件

```bash
# 处理本地视频文件，生成字幕和摘要
python main.py --video path/to/your/video.mp4

# 使用更大的模型提高转录精度
python main.py --video path/to/your/video.mp4 --model-size medium
```

#### 处理本地音频文件

```bash
# 处理本地音频文件，生成字幕和摘要
python main.py --audio path/to/your/audio.mp3

# 不生成字幕，只生成摘要
python main.py --audio path/to/your/audio.mp3 --no-subtitles
```

#### 批量处理多个YouTube视频

```bash
# 从文件读取URL列表（每行一个URL）
python main.py --batch urls.txt

# 批量下载完整视频并嵌入字幕
python main.py --batch urls.txt --download-video
```

#### 高级选项

```bash
# 使用特定的AI模型生成摘要
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --model "claude-3-sonnet-20240229"

# 指定API密钥和基础URL
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --api-key "your_api_key" --base-url "https://api.example.com/v1"

# 不翻译字幕（仅生成原语言字幕）
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --no-translation

# 不将字幕嵌入到视频中
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --download-video --no-embed
```

#### 使用自定义模板

```bash
# 使用指定模板处理视频
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --template news
```

## 📋 参数说明

| 参数 | 描述 |
|------|------|
| `--youtube URL` | 指定要处理的YouTube视频URL |
| `--video PATH` | 指定要处理的本地视频文件路径 |
| `--audio PATH` | 指定要处理的本地音频文件路径 |
| `--text PATH` | 指定要处理的本地文本文件路径，直接进行文章生成 |
| `--batch FILE` | 指定包含多个YouTube URL的文本文件路径 |
| `--model NAME` | 指定使用的AI模型名称，例如 "gpt-4" 或 "claude-3-sonnet-20240229" |
| `--api-key KEY` | 指定API密钥，覆盖环境变量中的设置 |
| `--base-url URL` | 指定自定义API基础URL，用于自定义API端点或代理 |
| `--model-size SIZE` | 指定Whisper模型大小，可选：tiny, base, small, medium, large，默认为tiny |
| `--no-stream` | 禁用流式输出 |
| `--summary-dir DIR` | 指定文章保存目录，默认为summaries |
| `--download-video` | 下载完整视频而不仅仅是音频 |
| `--no-subtitles` | 不生成字幕 |
| `--no-translation` | 不翻译字幕 |
| `--no-embed` | 不将字幕嵌入到视频中 |
| `--no-summary` | 不生成文本摘要 |
| `--template NAME` | 使用指定的模板，可以是模板名称或完整路径 |

## 📚 内置模板系统

程序提供了强大的模板系统，让您可以自定义文章生成的风格和内容。模板文件存储在`templates`目录下，使用简单的文本格式。

### 默认模板

系统自带一个默认模板`default.txt`，内容如下：

```
请将以下文本改写成一篇完整、连贯、专业的文章。

要求：
1. 你是一名资深科技领域编辑，同时具备优秀的文笔，文本转为一篇文章，确保段落清晰，文字连贯，可读性强，必要修改调整段落结构，确保内容具备良好的逻辑性。
2. 添加适当的小标题来组织内容
3. 以markdown格式输出，充分利用标题、列表、引用等格式元素
4. 如果原文有技术内容，确保准确表达并提供必要的解释

原文内容：
{content}
```

### 作家写作模板

系统还提供了一个专业的作家风格匹配模板`作家写作.txt`，可以根据内容自动匹配最适合的写作风格：

```
## 核心指令
我是一个作家风格匹配助手。当你提供写作任务时，我会：
1. 分析写作需求和目标受众
2. 从作家库中匹配最适合的写作风格
3. 提供具体的风格应用方案
4. 当无完全匹配时，推荐最接近的选项并说明原因

## 作家风格数据库
包含多种类型的作家风格：
- 生产力与系统思维专家（Tim Ferriss、Marie Poulin、Ali Abdaal等）
- 数字写作教育家（Nicolas Cole、Dickie Bush、David Perell等）
- 个人发展专家（James Clear、Mark Manson等）
- 知识管理与学习专家（Tiago Forte、Anne-Laure Le Cunff等）
- 创意与艺术思维（Austin Kleon等）
- 商业与创业思维（Paul Graham、Morgan Housel等）
- 极简主义与个人品牌（Dan Koe、Derek Sivers等）
- 系统思维与营销（Sahil Bloom、Julian Shapiro等）
- 创新思维与生产力（Khe Hy、Nathan Barry等）
- 内容创作与策略（Josh Spector等）

## 输出格式
1. 推荐作家风格及理由
2. 具体写作建议和框架
3. 风格应用示例
4. 如无完全匹配，提供最佳替代方案
5. 直接输出一篇按照推荐风格撰写的文章
```

### 使用模板

您可以通过以下方式使用模板：

```bash
# 使用默认模板
python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id

# 使用作家写作模板
python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id --template 作家写作

# 使用自定义模板
python youtube_transcriber.py --youtube https://www.youtube.com/watch?v=your_video_id --template your_template_name
```

### 创建自定义模板

您可以直接在templates目录下创建文本文件：

```
# 文件名为模板名称，内容中使用{content}作为占位符表示转录内容
```

### 模板编写指南

创建有效的模板时，请遵循以下原则：

1. 使用`{content}`作为占位符，表示转录内容将插入的位置
2. 提供清晰的指导和要求，告诉AI模型您期望的输出格式和风格
3. 可以指定目标受众、文章类型、风格特点等具体要求
4. 可以包含示例或参考，帮助AI模型更好地理解您的期望

### 模板管理

可以通过文本编辑器打开templates目录下的文件查看特定模板内容。

## 📄 本地文本处理功能

除了处理视频和音频文件外，本工具还支持直接处理本地文本文件，跳过下载和转录步骤，直接生成高质量文章。这在以下场景特别有用：

- 已有转录文本，想要生成文章
- 处理会议记录、演讲稿、采访内容等文本材料
- 改写或优化现有文章
- 批量处理多个文本文件

### 基本用法

```bash
python main.py --text path/to/your/text.txt
```

### 使用模板处理文本

```bash
# 使用默认模板
python main.py --text path/to/your/text.txt

# 使用作家写作模板
python main.py --text path/to/your/text.txt --template 作家写作

# 使用自定义模板
python main.py --text path/to/your/text.txt --template your_template_name
```

### 指定输出目录

```bash
python main.py --text path/to/your/text.txt --summary-dir my_articles
```

### 支持的文本格式

工具支持处理以下格式的文本文件：
- `.txt` - 纯文本文件
- `.md` - Markdown文件

其他格式的文件也可以尝试处理，但可能会出现格式问题。

### 文本处理流程

1. 读取本地文本文件内容
2. 应用指定的模板或自定义提示词
3. 使用AI模型生成高质量文章
4. 将生成的文章保存到指定目录

### 批量处理文本文件

您可以创建批处理脚本来处理多个文本文件：

**Windows (batch.bat)**
```batch
@echo off
python main.py --text texts/file1.txt --template 作家写作
python main.py --text texts/file2.txt --template 作家写作
python main.py --text texts/file3.txt --template 作家写作
```

**Linux/Mac (batch.sh)**
```bash
#!/bin/bash
python main.py --text texts/file1.txt --template 作家写作
python main.py --text texts/file2.txt --template 作家写作
python main.py --text texts/file3.txt --template 作家写作
```

## 🔧 技术实现细节

### 文件名处理

为确保文件名在各操作系统中的兼容性，程序使用`sanitize_filename`函数处理所有文件名：

- 替换不安全字符（如 `< > : " / \ | ? *`等）为下划线
- 将空格替换为下划线，避免路径问题
- 移除前导和尾随空格
- 确保文件名不为空

### 视频和音频下载

使用`yt-dlp`库下载YouTube内容：

- 支持选择仅下载音频或完整视频
- 自动选择最佳质量
- 使用FFmpeg进行音频提取和格式转换
- 详细的下载进度和错误信息显示

### 音频转录

使用OpenAI的Whisper模型进行音频转录：

- 支持多种模型大小（tiny到large）以平衡速度和准确性
- 自动检测并使用GPU加速（如果可用）
- 按段落格式化转录文本，提高可读性

### AI文章生成

采用创新的组合模型方法生成高质量文章：

- 使用 TextSummaryComposite 类灵活处理不同的API调用
- 支持 OpenAI、Claude 和 DeepSeek 等多种模型
- 智能错误处理，当一个模型失败时自动尝试其他模型
- 支持自定义API端点，便于使用第三方API服务
- 支持流式输出，实时显示生成进度
- 自动清理Markdown格式，确保输出文章格式正确

### 本地视频处理

支持处理本地视频文件：

- 自动提取视频中的音频轨道
- 使用Whisper模型转录音频内容
- 生成高质量文章摘要
- 支持自定义提示词和模板

### 批量处理

支持高效的批量处理多个YouTube视频：

- 从文本文件或命令行参数读取多个URL
- 详细的进度和错误报告
- 批处理总结，显示成功和失败的数量

## 💻 模块化结构

本项目采用模块化设计，将各个功能分离到不同的模块中，以提高代码的可维护性和可扩展性。

### 模块说明

- **config.py**: 配置模块，管理API密钥、模型设置和目录结构
- **downloader.py**: 下载器模块，负责从YouTube下载视频/音频和从视频中提取音频
- **subtitle_extractor.py**: 字幕提取器模块，负责音频转录和字幕生成
- **translator.py**: 翻译器模块，负责字幕翻译功能
- **summarizer.py**: 摘要生成器模块，负责文本摘要生成
- **utils.py**: 工具函数模块，提供各种通用工具方法
- **main.py**: 主程序模块，整合所有功能并提供命令行接口

### 模块交互

![](https://raw.githubusercontent.com/cacity/py_test/master/test/ChatGPT Image 2025年4月6日 12_20_21.png)



## 📁 目录结构

### 代码文件

- `main.py`: 主程序文件
- `config.py`: 配置模块
- `utils/`: 工具模块目录
  - `__init__.py`: 包初始化文件
  - `common.py`: 通用工具函数模块
  - `composite.py`: 组合模型处理模块，用于处理不同API的组合调用
  - `downloader.py`: 下载器模块
  - `subtitle_extractor.py`: 字幕提取器模块
  - `translator.py`: 翻译器模块
  - `summarizer.py`: 摘要生成模块
  - `summarizer.py`: 摘要生成器模块
- `requirements.txt`: 依赖项文件
- `.env`: 环境变量文件

### 自动创建的目录

程序会自动创建以下目录：

- `downloads/`: 存储下载的音频文件
- `videos/`: 存储下载的视频文件
- `transcripts/`: 存储音频转录文本
- `summaries/`: 存储生成的摘要
- `templates/`: 存储摘要生成模板
- `subtitles/`: 存储生成的字幕文件
- `extracted_audio/`: 存储从视频中提取的音频
- `videos_with_subtitles/`: 存储嵌入字幕的视频

## 📊 输出示例

处理完成后，会生成以下文件：

1. 音频文件：`downloads/视频标题.mp3`
2. 视频文件（如果选择下载视频）：`videos/视频标题.mp4`
3. 提取的音频（从本地视频提取）：`extracted_audio/视频标题.mp3`
4. 转录文本：`transcripts/视频标题_transcript.txt`
5. 字幕文件：
   - `subtitles/视频标题.srt`（SRT格式）
   - `subtitles/视频标题.vtt`（WebVTT格式）
   - `subtitles/视频标题.ass`（ASS格式）
6. 双语字幕文件：
   - `subtitles/视频标题_bilingual.srt`
   - `subtitles/视频标题_bilingual.vtt`
   - `subtitles/视频标题_bilingual.ass`
7. 嵌入字幕的视频：`videos_with_subtitles/视频标题_with_subtitles.mp4`
8. 生成摘要：`summaries/视频标题_20240101_123456_summary.md`

## ⚠️ 注意事项

- 确保已安装FFmpeg，用于音频处理
- 大型Whisper模型需要较多GPU内存
- 处理长视频可能需要较长时间
- 请遵守YouTube的服务条款和API使用政策

## 🔄 当前版本

### v2.1.0 (2025-04-06)
- 引入 TextSummaryComposite 类，优化 API 调用处理
- 移除 anthropic 库依赖，改用 requests 库直接调用 Claude API
- 增强错误处理机制，提高程序稳定性
- 添加命令行参数 --api-key 和 --base-url，支持自定义 API 配置
- 优化环境变量加载逻辑，支持多种 API 服务提供商
- 添加 --no-summary 选项，控制是否生成文本摘要

## 📄 许可证

MIT License

## 📝 YouTube 字幕功能

本工具提供了强大的 YouTube 视频字幕生成和处理功能，支持多种字幕格式和双语翻译。

### 主要特性

- 🎬 支持从 YouTube 视频中提取音频并生成字幕
- 🌐 支持自动将字幕翻译成中文（双语字幕）
- 🎯 支持多种字幕格式（SRT、WebVTT、ASS）
- 📊 支持将字幕嵌入到视频中
- 🔄 支持批量处理多个 YouTube 视频
- 🎵 支持处理本地视频和音频文件

### 基本用法

#### 处理单个 YouTube 视频并生成字幕

```bash
python main.py --youtube https://www.youtube.com/watch?v=your_video_id
```

#### 处理本地视频文件

```bash
python main.py --video path/to/your/video.mp4
```

#### 处理本地音频文件

```bash
python main.py --audio path/to/your/audio.mp3
```

#### 批量处理多个 YouTube 视频

```bash
# 创建一个文本文件，每行包含一个 YouTube 链接
python main.py --batch path/to/your/urls.txt
```

### 高级选项

#### 下载完整视频并嵌入字幕

```bash
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --download-video
```

#### 选择 Whisper 模型大小

```bash
# 可选模型大小: tiny, base, small, medium, large
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --model-size medium
```

#### 禁用字幕翻译

```bash
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --no-translation
```

#### 禁用字幕嵌入

```bash
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --download-video --no-embed
```

### 字幕格式说明

处理完成后，会生成以下三种格式的字幕文件：

1. **SRT 格式**：最常用的字幕格式，兼容大多数视频播放器
2. **WebVTT 格式**：适用于网页视频播放，支持更多样式
3. **ASS 格式**：高级字幕格式，支持丰富的样式和特效，适合嵌入视频

所有字幕文件都保存在 `subtitles` 目录下，文件名格式为 `视频名称_bilingual.扩展名`。

### 字幕嵌入功能

本工具支持将生成的字幕嵌入到视频中，生成带有硬编码字幕的新视频文件。嵌入过程使用 FFmpeg，优先使用 ASS 格式字幕以获得最佳效果。

```bash
# 下载视频并嵌入字幕
python main.py --youtube https://www.youtube.com/watch?v=your_video_id --download-video
```

嵌入字幕的视频保存在 `videos_with_subtitles` 目录下。

### 技术实现细节

- 使用 yt-dlp 下载 YouTube 视频或音频
- 使用 OpenAI Whisper 模型进行高精度音频转录
- 使用 Google 翻译 API 将字幕翻译成中文
- 使用 FFmpeg 将字幕嵌入到视频中
- 支持多种字幕格式（SRT、WebVTT、ASS）
- 自动检测源语言，仅在非中文内容时进行翻译
- 使用缓存机制避免重复翻译相同内容

## 🙏 致谢

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)：强大的YouTube下载工具
- [OpenAI Whisper](https://github.com/openai/whisper)：高精度音频转录模型
- [OpenAI API](https://openai.com/blog/openai-api)：提供文本生成能力
- [DeepSeek](https://www.deepseek.com/)：提供推理能力
- [FFmpeg](https://ffmpeg.org/)：视频处理工具
