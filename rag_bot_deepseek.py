import os
import re
import pickle
from typing import List

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from prompt import rag_prompt

# 加载 .env 环境变量
load_dotenv()

# =================配置加载=================
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_PATH = CFG["data_path"]
PERSIST_DIRECTORY = CFG["persist_directory"]
REBUILD_DB = CFG["rebuild_db"]

EMBEDDING_CFG = CFG["embedding"]
SPLITTER_CFG = CFG["text_splitter"]
RETRIEVAL_CFG = CFG["retrieval"]
RERANK_CFG = CFG["rerank"]
LLM_CFG = CFG["llm"]
META_CFG = CFG["metadata"]

DEEPSEEK_API_KEY = os.getenv(LLM_CFG["api_key_env"])
DEEPSEEK_BASE_URL = os.getenv(LLM_CFG["base_url_env"], LLM_CFG["default_base_url"])
# =================配置加载=================


def extract_doc_type(file_path: str) -> str:
    """从文件名提取 doc_type，去掉扩展名"""
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name


def extract_chapter_section(text: str):
    """从文本开头提取章节信息"""
    head = text[:300].strip()

    part_pattern = re.compile(r"^(第[一二三四五六七八九十\d]+[部分章节])[\s、.:：]*(.{0,30})")
    m = part_pattern.search(head)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    chinese_num_pattern = re.compile(r"^([一二三四五六七八九十])[、.．\s]+(.{0,30})")
    m = chinese_num_pattern.search(head)
    if m:
        return m.group(1) + "、", m.group(2).strip()

    arabic_num_pattern = re.compile(r"^(\d+(?:\.\d+)*)[\s.．]+(.{0,30})")
    m = arabic_num_pattern.search(head)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    bracket_pattern = re.compile(r"^(\(\d+\))[\s.．]*(.{0,30})")
    m = bracket_pattern.search(head)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None, None


def split_documents_with_metadata(documents: List[Document]) -> List[Document]:
    """按层级标题切分，并为每个切片附加元数据"""
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
        chunk_size=SPLITTER_CFG["chunk_size"],
        chunk_overlap=SPLITTER_CFG["chunk_overlap"],
        separators=separators,
        is_separator_regex=False,
    )

    all_splits = []
    for doc in documents:
        base_meta = {
            "source": doc.metadata.get("source", ""),
            "doc_type": extract_doc_type(doc.metadata.get("source", "")),
            "department": META_CFG["department"],
            "effective_date": META_CFG["effective_date"],
            "status": META_CFG["status"],
        }
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


class HybridRetriever:
    """多路召回检索器：向量检索 + BM25 检索 + RRF 融合 + Cross-Encoder 重排"""

    def __init__(self, vectordb, documents: List[Document], rerank_model=None):
        self.vectordb = vectordb
        self.documents = documents
        self.doc_map = {doc.metadata.get("source", "") + "_" + str(i): doc for i, doc in enumerate(documents)}
        # BM25 初始化
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.rerank_model = rerank_model

    def _tokenize(self, text: str) -> List[str]:
        """简单中文分词：按字符切分，BM25 对中文足够有效"""
        return list(text.strip())

    def _vector_search(self, query: str, k: int) -> List[Document]:
        return self.vectordb.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k_indices]

    @staticmethod
    def _rrf_fuse(results_list: List[List[Document]], k: int = 60) -> List[Document]:
        """RRF 融合多路召回结果"""
        scores = {}
        doc_id_map = {}
        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = id(doc)
                doc_id_map[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id_map[doc_id] for doc_id, _ in sorted_docs]

    def _rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not self.rerank_model or len(documents) == 0:
            return documents[:top_k]
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.rerank_model.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def retrieve(self, query: str) -> List[Document]:
        vector_k = RETRIEVAL_CFG["vector_top_k"]
        bm25_k = RETRIEVAL_CFG["bm25_top_k"]
        fusion_k = RETRIEVAL_CFG["fusion_top_k"]
        final_k = RETRIEVAL_CFG["final_top_k"]

        vector_results = self._vector_search(query, vector_k)
        bm25_results = self._bm25_search(query, bm25_k)

        fused_results = self._rrf_fuse([vector_results, bm25_results])
        fused_results = fused_results[:fusion_k]

        final_results = self._rerank(query, fused_results, final_k)
        return final_results


def build_vectordb(splits: List[Document], embeddings):
    """构建或加载向量数据库"""
    if REBUILD_DB or not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        print("正在构建向量数据库...")
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )
        # 保存 splits 用于 BM25
        with open(os.path.join(PERSIST_DIRECTORY, "documents.pkl"), "wb") as f:
            pickle.dump(splits, f)
        print("向量数据库构建完成并已持久化。")
    else:
        print("正在加载已有向量数据库...")
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        print("向量数据库加载完成。")
    return vectordb


def load_documents():
    """加载持久化的 documents，用于 BM25"""
    pkl_path = os.path.join(PERSIST_DIRECTORY, "documents.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None


def main():
    if not DEEPSEEK_API_KEY:
        print("错误：未找到 DEEPSEEK_API_KEY。请检查 .env 文件或环境变量。")
        return

    print("正在启动基于 DeepSeek 的文档问答机器人...")

    # 1. 加载 Embedding 模型
    print(f"加载本地 Embedding 模型：{EMBEDDING_CFG['model_name']}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CFG["model_name"],
        model_kwargs={"device": EMBEDDING_CFG["device"]},
        encode_kwargs={"normalize_embeddings": EMBEDDING_CFG["normalize_embeddings"]},
    )

    splits = None
    if REBUILD_DB:
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

        splits = split_documents_with_metadata(documents)
        print(f"文本已分割为 {len(splits)} 个片段。")
    else:
        splits = load_documents()
        if splits is None:
            print("错误：未找到已持久化的文档数据，请将 rebuild_db 设为 true 重新构建。")
            return
        print(f"已从缓存加载 {len(splits)} 个文档片段。")

    # 2. 构建/加载向量数据库
    vectordb = build_vectordb(splits, embeddings)

    # 3. 初始化重排模型
    rerank_model = None
    if RERANK_CFG["enabled"]:
        print(f"加载重排模型：{RERANK_CFG['model_name']}...")
        rerank_model = CrossEncoder(RERANK_CFG["model_name"], device=EMBEDDING_CFG["device"])

    # 4. 初始化混合检索器
    hybrid_retriever = HybridRetriever(vectordb, splits, rerank_model=rerank_model)

    # 5. 初始化 LLM
    print(f"正在连接 DeepSeek 模型 ({LLM_CFG['model_name']})...")
    llm = ChatOpenAI(
        model=LLM_CFG["model_name"],
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=LLM_CFG["temperature"],
        max_tokens=LLM_CFG["max_tokens"],
    )

    # 6. 自定义检索问答链（使用混合检索器）
    class CustomRetriever(BaseRetriever):
        retriever: HybridRetriever

        def _get_relevant_documents(self, query: str) -> List[Document]:
            return self.retriever.retrieve(query)

        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)

    custom_retriever = CustomRetriever(retriever=hybrid_retriever)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
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
