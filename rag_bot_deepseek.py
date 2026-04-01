import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mobile_template import rag_prompt

# 加载 .env 环境变量
load_dotenv()

# =================配置区域=================
PERSIST_DIRECTORY = "./chroma_db_storage"
DATA_PATH = "./mobile_knowledge_base.md"

# Embedding 模型 (依然推荐本地运行，节省 Token 费用且速度快)
EMBEDDING_MODEL_NAME = "moka-ai/m3e-base"

# DeepSeek 配置
# 注意：DeepSeek 兼容 OpenAI 协议，所以可以使用 ChatOpenAI 类
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL_NAME = "kimi" # 或者 "deepseek-coder"
# =================配置区域=================


def main():
    # 1. 加载文档
    if not DEEPSEEK_API_KEY:
        print("❌ 错误：未找到 DEEPSEEK_API_KEY。请检查 .env 文件或环境变量。")
        return

    print("🚀 正在启动基于 DeepSeek 的文档问答机器人...")
    if os.path.isfile(DATA_PATH):
        if not os.path.exists(DATA_PATH):
            print(f"❌ 文件 {DATA_PATH} 不存在，请先创建该文件。")
            return
        loader = TextLoader(DATA_PATH,encoding="utf-8")
        documents = loader.load()
    else:
        loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
        documents = loader.load()
    print(f"📄 已加载 {len(documents)} 个文档。")

    # 2. 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n"]  # 优先按标题切分
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️ 文本已分割为 {len(splits)} 个片段。")

    # 3. 初始化 Embedding 模型 (本地)
    print(f"⬇️  加载本地 Embedding 模型：{EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 创建/加载向量数据库
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"💾 向量数据库就绪。")

    # 5. 初始化 DeepSeek LLM
    print(f"🔗 正在连接 DeepSeek 模型 ({DEEPSEEK_MODEL_NAME})...")
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,  # 降低温度使回答更稳定、准确
        max_tokens=1024
    )

    # 6. 设置提示词模板 (RAG 核心) 引用mobile_template.py


    # 7. 构建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # 每次检索最相关的 3 个片段
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )

    print("\n✅ 系统初始化完成！开始对话 (输入 'quit' 退出):")
    print("-" * 30)

    while True:
        try:
            user_input = input("\n👤 你: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 再见！")
                break

            if not user_input.strip():
                continue

            # 执行检索和生成
            result = qa_chain.invoke({"query": user_input})

            print(f"🤖 DeepSeek: {result['result']}")

            # (可选) 显示参考来源，方便调试
            # print("\n[参考来源片段]:")
            # for i, doc in enumerate(result['source_documents']):
            #     print(f"  {i+1}. ...{doc.page_content[:60]}...")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("💡 请检查你的 API Key 是否正确，或网络连接是否正常。")

if __name__ == "__main__":
    main()