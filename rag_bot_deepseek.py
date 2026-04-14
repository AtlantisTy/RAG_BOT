import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompt import rag_prompt

# 加载 .env 环境变量
load_dotenv()

# =================配置区域=================
PERSIST_DIRECTORY = "./chroma_db_storage"
DATA_PATH = "./doc"

# Embedding 模型 (依然推荐本地运行，节省 Token 费用且速度快)
EMBEDDING_MODEL_NAME = "moka-ai/m3e-base"

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL_NAME = "kimi"  # 或者 "deepseek-coder"

# 元数据默认值
DEFAULT_DEPARTMENT = "人事"
DEFAULT_EFFECTIVE_DATE = "2026-01-01"
DEFAULT_STATUS = "valid"
# =================配置区域=================


def extract_doc_type(file_path):
    """从文件名提取 doc_type，去掉扩展名"""
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name


def extract_chapter_section(text):
    """
    从文本开头提取章节信息。
    匹配模式（按优先级）：
      - 第一部分 / 第二部分 / 第一章 等
      - 一、二、三、 或 一. 二. 等
      - 1.  1.1  1.1.1  等
      - (1) (2) 等
    返回 (chapter, section)
    """
    # 取文本前 300 字符进行匹配，避免全文扫描
    head = text[:300].strip()

    # 匹配 "第一部分 xxx" / "第二部分 xxx" / "第一章 xxx"
    part_pattern = re.compile(r"^(第[一二三四五六七八九十\d]+[部分章节])[\s、.:：]*(.{0,30})")
    m = part_pattern.search(head)
    if m:
        chapter = m.group(1).strip()
        section = m.group(2).strip()
        return chapter, section

    # 匹配 "一、xxx" / "二、xxx" / "一. xxx"
    chinese_num_pattern = re.compile(r"^([一二三四五六七八九十])[、.．\s]+(.{0,30})")
    m = chinese_num_pattern.search(head)
    if m:
        chapter = m.group(1) + "、"
        section = m.group(2).strip()
        return chapter, section

    # 匹配 "1. xxx" / "1.1 xxx" / "1.1.1 xxx"
    arabic_num_pattern = re.compile(r"^(\d+(?:\.\d+)*)[\s.．]+(.{0,30})")
    m = arabic_num_pattern.search(head)
    if m:
        chapter = m.group(1).strip()
        section = m.group(2).strip()
        return chapter, section

    # 匹配 "(1) xxx" / "(2) xxx"
    bracket_pattern = re.compile(r"^(\(\d+\))[\s.．]*(.{0,30})")
    m = bracket_pattern.search(head)
    if m:
        chapter = m.group(1).strip()
        section = m.group(2).strip()
        return chapter, section

    return None, None


def split_documents_with_metadata(documents):
    """
    使用策略 B：先按层级标题切分，再为每个切片附加元数据。
    """
    # 定义分隔符：优先按大标题切分，再按小标题
    separators = [
        "\n第一部分 ", "\n第二部分 ", "\n第三部分 ", "\n第四部分 ", "\n第五部分 ",
        "\n第一章 ", "\n第二章 ", "\n第三章 ", "\n第四章 ", "\n第五章 ",
        "\n一、", "\n二、", "\n三、", "\n四、", "\n五、",
        "\n六、", "\n七、", "\n八、", "\n九、", "\n十、",
        "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ",
        "\n6. ", "\n7. ", "\n8. ", "\n9. ", "\n10. ",
        "\n(1)", "\n(2)", "\n(3)", "\n(4)", "\n(5)",
        "\n\n", "\n"
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=separators,
        is_separator_regex=False,
    )

    all_splits = []
    for doc in documents:
        base_meta = {
            "source": doc.metadata.get("source", ""),
            "doc_type": extract_doc_type(doc.metadata.get("source", "")),
            "department": DEFAULT_DEPARTMENT,
            "effective_date": DEFAULT_EFFECTIVE_DATE,
            "status": DEFAULT_STATUS,
        }

        # 先对该文档进行切分
        splits = text_splitter.split_documents([doc])

        for split in splits:
            chapter, section = extract_chapter_section(split.page_content)
            split.metadata.update(base_meta)
            if chapter:
                split.metadata["chapter"] = chapter
            if section:
                split.metadata["section"] = section
            all_splits.append(split)

    return all_splits


def main():
    # 1. 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("错误：未找到 DEEPSEEK_API_KEY。请检查 .env 文件或环境变量。")
        return

    print("正在启动基于 DeepSeek 的文档问答机器人...")

    # 2. 加载 PDF 文档
    if not os.path.exists(DATA_PATH):
        print(f"目录 {DATA_PATH} 不存在，请先创建该目录并放入 PDF 文件。")
        return

    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    print(f"已加载 {len(documents)} 页 PDF 内容。")

    # 3. 切分并附加元数据
    splits = split_documents_with_metadata(documents)
    print(f"文本已分割为 {len(splits)} 个片段。")

    # 4. 初始化 Embedding 模型 (本地)
    print(f"加载本地 Embedding 模型：{EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 5. 创建/加载向量数据库
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    print("向量数据库就绪。")

    # 6. 初始化 DeepSeek LLM
    print(f"正在连接 DeepSeek 模型 ({DEEPSEEK_MODEL_NAME})...")
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1024,
    )

    # 7. 构建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt},
    )

    print("\n系统初始化完成！开始对话 (输入 'quit' 退出):")
    print("-" * 30)

    while True:
        try:
            user_input = input("\n你: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break

            if not user_input.strip():
                continue

            result = qa_chain.invoke({"query": user_input})
            print(f"DeepSeek: {result['result']}")

        except Exception as e:
            print(f"发生错误: {e}")
            print("请检查你的 API Key 是否正确，或网络连接是否正常。")


if __name__ == "__main__":
    main()
