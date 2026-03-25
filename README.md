# 🤖 RAG 智能问答机器人

## 📖 项目简介

这是一个**智能问答机器人系统**，能够理解你的问题并从知识库中查找答案。就像你有一个随时待命的智能助手，它可以阅读你提供的文档资料，然后根据这些资料回答你的各种问题。

### 🌟 核心功能

本项目包含两个独立的机器人：

#### 1️⃣ **文档问答机器人** (`rag_bot_deepseek.py`)
- **功能**：阅读你提供的文档（如产品手册、知识库等），然后回答相关问题
- **应用场景**：
  - 客服自动问答（根据产品说明书回答问题）
  - 企业内部知识查询（员工手册、规章制度）
  - 学习资料问答（笔记、教材）
- **工作原理**：
  1. 读取你提供的知识文档
  2. 使用 AI 技术理解文档内容
  3. 当你提问时，从文档中找到相关信息并生成答案

#### 2️⃣ **图片检索机器人** (`rag_bot_image.py`)
- **功能**：根据一张图片，快速找到相似的图片
- **应用场景**：
  - 以图搜图（找相似商品、同款商品）
  - 图片分类管理（自动将相似图片归类）
  - 设计素材检索（找风格相似的设计）
- **工作原理**：
  1. 分析你图片库中所有图片的特征
  2. 当你提供一张查询图片时，计算它与库中图片的相似度
  3. 找出最相似的几张图片并展示给你

---

## 🚀 快速开始指南

### 第一步：检查环境要求

**系统要求**：
- Windows 10/11 操作系统
- Python 3.8 或更高版本（推荐 Python 3.10+）
- 至少 4GB 可用内存
- 至少 10GB 可用磁盘空间

**检查 Python 是否已安装**：
1. 按 `Win + R` 键
2. 输入 `cmd` 并回车
3. 在黑色窗口中输入：`python --version`
4. 如果显示版本号（如 `Python 3.10.0`），说明已安装；如果提示"不是内部命令"，需要先安装 Python

**安装 Python**（如果未安装）：
1. 访问官网：https://www.python.org/downloads/
2. 下载最新版本的 Python
3. 运行安装包，**务必勾选 "Add Python to PATH"**
4. 点击 "Install Now"

---

### 第二步：安装项目依赖

依赖包是让程序运行所需的工具包。

**方法一：使用自动安装脚本（推荐）**

1. 打开项目文件夹：`d:\PyProject\RAG_BOT`
2. 在文件夹空白处按住 `Shift` 键并点击鼠标右键
3. 选择 "在此处打开 PowerShell 窗口" 或 "在终端中打开"
4. 输入以下命令并回车：

```bash
pip install -r requirements.txt
```

5. 等待安装完成（大约需要 5-10 分钟，取决于网速）

**方法二：手动安装**

如果上面的方法失败，可以逐个安装：

```bash
pip install langchain==1.2.13
pip install langchain-chroma==1.1.0
pip install sentence-transformers==5.3.0
pip install python-dotenv==1.2.2
pip install openai==2.29.0
pip install pillow==12.1.1
pip install matplotlib==3.10.8
```

---

### 第三步：配置 API 密钥（仅文档问答机器人需要）

**什么是 API 密钥？**
API 密钥就像是网站的登录密码，有了它程序才能连接到 AI 服务。

**获取 DeepSeek API 密钥**：
1. 访问 DeepSeek 官网：https://platform.deepseek.com/
2. 注册账号并登录
3. 进入控制台，找到 "API Keys" 页面
4. 点击 "创建新的 API Key"
5. 复制生成的密钥（格式类似：`sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`）

**配置密钥到项目**：
1. 在项目文件夹中找到 `.env` 文件
2. 用记事本或代码编辑器打开
3. 修改以下内容：

```
DEEPSEEK_API_KEY=你的 API 密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

把 `你的 API 密钥` 替换成你刚才复制的密钥，保存文件。

> 💡 **提示**：`.env` 文件已经存在，你只需要修改 API 密钥即可

---

### 第四步：准备你的知识库

#### 对于文档问答机器人：

项目默认使用 `mobile_knowledge_base.md` 文件作为知识库。

**如何自定义知识库**：

1. **方式 A：直接编辑现有文件**
   - 用记事本打开 `mobile_knowledge_base.md`
   - 删除里面的内容，填入你自己的知识
   
2. **方式 B：创建新的知识文件**
   - 在项目文件夹中新建一个 `.txt` 或 `.md` 文件
   - 例如：`my_knowledge.txt`
   - 在里面写入你的知识内容

**知识库文件格式示例**：

```markdown
# 公司产品介绍

## 产品 A
- 价格：1999 元
- 颜色：黑色、白色、蓝色
- 保修期：2 年

## 产品 B
- 价格：2999 元
- 特点：防水、防尘
- 电池容量：5000mAh
```

**修改代码指向你的知识库**（如果用新文件）：
1. 用记事本打开 `rag_bot_deepseek.py`
2. 找到第 20 行：`DATA_PATH = "./mobile_knowledge_base.md"`
3. 改成：`DATA_PATH = "./my_knowledge.txt"`
4. 保存文件

---

#### 对于图片检索机器人：

项目默认从 `images_db` 文件夹读取图片。

**如何使用自己的图片**：

1. 打开项目文件夹中的 `images_db` 文件夹
2. 删除里面的示例图片（可选）
3. 把你自己的图片复制到这个文件夹
4. 支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**设置查询图片**：

项目默认使用 `query_image.jpg` 作为查询图片。

1. 准备一张你想用来搜索相似的图片
2. 把它放到项目根目录
3. 重命名为 `query_image.jpg`
4. 或者修改代码：打开 `rag_bot_image.py`，找到第 13 行，改成你的图片路径

---

### 第五步：运行机器人

#### 🎯 运行文档问答机器人

1. 确保已完成前面所有步骤
2. 在项目文件夹中打开 PowerShell/命令行窗口
3. 输入以下命令：

```bash
python rag_bot_deepseek.py
```

4. 等待程序启动（首次运行会下载模型，可能需要几分钟）
5. 看到提示 `✅ 系统初始化完成！开始对话` 后，就可以开始提问了

**使用示例**：

```
👤 你：幻影 X-Pro 的摄像头参数是什么？
🤖 DeepSeek: [根据知识库回答...]

👤 你：哪款手机性价比最高？
🤖 DeepSeek: [根据知识库回答...]

👤 你：quit
👋 再见！
```

输入 `quit`、`exit` 或 `q` 可以退出程序。

---

#### 🖼️ 运行图片检索机器人

1. 确保 `images_db` 文件夹里有图片
2. 确保 `query_image.jpg` 存在（或者会使用库中第一张图测试）
3. 在命令行输入：

```bash
python rag_bot_image.py
```

4. 程序会自动：
   - 分析图片库中的所有图片
   - 计算相似度
   - 弹出一个窗口显示查询图片和最相似的几张结果

**查看结果**：
- 左边是查询图片
- 右边是找到的相似图片（按相似度排序）
- 每张图片下方显示文件名和相似度分数（分数越低越相似）

---

## ⚙️ 高级配置（可选）

### 修改文档问答机器人的参数

打开 `rag_bot_deepseek.py`，可以修改：

```python
# 向量数据库存储位置
PERSIST_DIRECTORY = "./chroma_db_storage"

# Embedding 模型（本地）
EMBEDDING_MODEL_NAME = "moka-ai/m3e-base"

# DeepSeek 模型
DEEPSEEK_MODEL_NAME = "deepseek-chat"  # 或 "deepseek-coder"

# 每次检索返回几个相关片段
search_kwargs={"k": 3}  # 改成其他数字
```

### 修改图片检索机器人的参数

打开 `rag_bot_image.py`，可以修改：

```python
# 图片库位置
IMAGE_LIBRARY_PATH = "./images_db"

# 向量数据库存储位置
PERSIST_DIRECTORY = "./chroma_img_storage"

# 查询图片路径
QUERY_IMAGE_PATH = "./query_image.jpg"

# 显示几张相似结果
top_k=3  # 在第 157 行修改

# Embedding 模型
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
```

---

## ❓ 常见问题解答

### Q1: 运行时提示找不到模块/包？
**A**: 说明依赖包没有安装成功。重新执行第二步的安装命令：
```bash
pip install -r requirements.txt
```

### Q2: 文档机器人说"未找到 DEEPSEEK_API_KEY"？
**A**: 
1. 检查 `.env` 文件是否存在
2. 打开 `.env` 确认 API 密钥是否正确填写
3. 确保格式正确：`DEEPSEEK_API_KEY=sk-xxxxx`（等号两边不要有空格）
4. 保存文件后重新运行

### Q3: 首次运行很慢，卡住不动？
**A**: 
- 第一次运行时会下载 AI 模型（几个 GB），这是正常现象
- 建议在网络良好的环境下运行
- 下载完成后，后续运行速度会很快

### Q4: 图片检索没有弹出窗口？
**A**: 
1. 检查是否安装了图形界面支持
2. 尝试更新 matplotlib：`pip install --upgrade matplotlib`
3. 检查 `images_db` 文件夹是否有图片

### Q5: 问答机器人回答不准确？
**A**: 
1. 检查知识库文件是否包含相关信息
2. 确保知识库内容清晰、结构化
3. 可以尝试修改提问方式，更具体一些
4. 调整 `search_kwargs={"k": 3}` 中的数字，增加检索片段数量

### Q6: 如何完全清空向量数据库重新开始？
**A**: 
- 文档机器人：删除 `chroma_db_storage` 文件夹
- 图片机器人：删除 `chroma_img_storage` 文件夹
- 重新运行程序会自动创建新的数据库

---

## 📁 项目文件说明

```
RAG_BOT/
├── rag_bot_deepseek.py          # 文档问答机器人主程序
├── rag_bot_image.py             # 图片检索机器人主程序
├── mobile_template.py           # 问答提示词模板（定义 AI 回答风格）
├── mobile_knowledge_base.md     # 示例知识库（手机产品信息）
├── knowledge.txt                # 向量数据库科普知识
├── .env                         # API 密钥配置文件
├── requirements.txt             # Python 依赖包列表
├── images_db/                   # 图片库文件夹
│   ├── cat1.jpg
│   └── ...                      # 放入你的图片
├── query_image.jpg              # 查询图片（用于图片检索）
├── chroma_db_storage/           # 文档向量数据库（自动生成）
└── chroma_img_storage/          # 图片向量数据库（自动生成）
```

---

## 💡 使用技巧

### 让文档机器人回答更好的技巧：

1. **知识库编写要结构化**
   - 使用标题分级（##、###）
   - 分点列出信息
   - 避免大段连续文字

2. **提问要具体**
   - ✅ 好问题："幻影 X-Pro 的电池容量是多少？"
   - ❌ 差问题："电池怎么样？"

3. **定期更新知识库**
   - 保持知识库内容最新
   - 删除过时信息

### 让图片检索更准确的技巧：

1. **图片质量要高**
   - 使用清晰的图片
   - 避免过于模糊的图片

2. **图片库要有代表性**
   - 放入足够多的参考图片
   - 覆盖不同的类别/风格

3. **查询图片要典型**
   - 使用能代表你想找的风格/类型的图片

---

## 🛠️ 故障排除

### 如果程序完全无法启动：

1. **检查 Python 版本**
   ```bash
   python --version
   ```
   确保是 3.8 或以上

2. **重新安装依赖**
   ```bash
   pip uninstall -y langchain langchain-chroma sentence-transformers
   pip install -r requirements.txt
   ```

3. **检查文件完整性**
   - 确保所有 `.py` 文件没有被修改破坏
   - 确保 `.env` 文件存在且格式正确

### 如果遇到编码错误：

确保你的知识文件使用 UTF-8 编码保存：
- 用记事本打开文件
- 点击 "文件" → "另存为"
- 在编码选项中选择 "UTF-8"
- 保存

---

## 📞 技术支持

如果遇到问题：
1. 先查看本 README 的"常见问题解答"部分
2. 检查项目文件是否完整
3. 确认环境变量和 API 密钥配置正确

---

## 📝 版本信息

- **项目版本**: 1.0
- **最后更新**: 2026 年 3 月 25 日
- **适用系统**: Windows 10/11
- **Python 版本**: 3.8+

---

## 🎉 开始使用

现在你已经掌握了所有必要的知识，开始使用 RAG 智能问答机器人吧！

**快速回顾**：
1. ✅ 安装 Python
2. ✅ 安装依赖包：`pip install -r requirements.txt`
3. ✅ 配置 API 密钥（文档机器人）
4. ✅ 准备知识库和图片
5. ✅ 运行机器人：`python rag_bot_deepseek.py` 或 `python rag_bot_image.py`

祝你使用愉快！🚀
