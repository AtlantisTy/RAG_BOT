import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt  # 导入绘图库

# LangChain & Chroma
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# =================配置区域=================
IMAGE_LIBRARY_PATH = "./images_db"
PERSIST_DIRECTORY = "./chroma_img_storage"
QUERY_IMAGE_PATH = "./query_image.jpg"
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


# =================配置区域=================

def get_image_embeddings(model, image_paths):
    """批量加载图片并计算向量"""
    images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"⚠️  跳过损坏的图片 {path}: {e}")

    if not images:
        return [], []

    embeddings = model.encode(images, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist(), valid_paths


def initialize_vector_db():
    """初始化数据库并手动添加图片向量"""
    print(f"📂 扫描目录：{IMAGE_LIBRARY_PATH}")

    if not os.path.exists(IMAGE_LIBRARY_PATH):
        os.makedirs(IMAGE_LIBRARY_PATH)
        print(f"❌ 目录不存在，已创建 {IMAGE_LIBRARY_PATH}。请放入图片后重新运行。")
        return None, None

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [str(f.resolve()) for f in Path(IMAGE_LIBRARY_PATH).iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print("❌ 目录下没有找到任何图片文件。")
        return None, None

    print(f"🖼️  找到 {len(image_files)} 张图片。")
    print("🔄 正在加载模型并计算向量...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    vectors, valid_paths = get_image_embeddings(model, image_files)

    if not vectors:
        return None, None

    ids = [Path(p).name for p in valid_paths]
    metadatas = [{"source": p, "filename": Path(p).name} for p in valid_paths]
    documents_content = [f"Image: {m['filename']}" for m in metadatas]

    print("💾 正在写入向量数据库...")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=None)
    collection = vectordb._client.get_collection(name=vectordb._collection.name)

    # 处理 ID 冲突 (如果重新运行脚本)
    existing_ids = collection.get(ids=ids)["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(ids=ids, embeddings=vectors, documents=documents_content, metadatas=metadatas)
    print("✅ 数据库就绪！")
    return vectordb, model


def search_and_show(vectordb, model, query_path, top_k=2):
    """搜索并显示图片"""
    if not os.path.exists(query_path):
        # 自动 fallback 到第一张图
        collection = vectordb._client.get_collection(name=vectordb._collection.name)
        first = collection.get(limit=1)
        if first['ids']:
            query_path = first['metadatas'][0]['source']
            print(f"💡 未找到指定查询图，自动使用库中图片测试：{os.path.basename(query_path)}")
        else:
            return

    print(f"🔍 正在分析：{os.path.basename(query_path)}...")
    try:
        img = Image.open(query_path).convert('RGB')
        query_vector = model.encode([img], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        print(f"❌ 读取图片失败：{e}")
        return

    collection = vectordb._client.get_collection(name=vectordb._collection.name)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k, include=["metadatas", "distances"])

    ids = results['ids'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    if not ids:
        print("❌ 未找到相似图片。")
        return

    # ================= 绘图核心逻辑 =================
    print("\n📊 正在生成预览窗口...")

    # 创建画布：1 行，(结果数量 + 1) 列 (多一列放查询图)
    fig, axes = plt.subplots(1, len(ids) + 1, figsize=(5 * (len(ids) + 1), 5))

    # 1. 显示查询图片 (最左边)
    # 如果只有一张结果，axes 不是列表，需要特殊处理，但通常 len >= 1 时 axes 是列表
    if len(ids) == 0:
        return

    # 确保 axes 总是可迭代的
    if len(ids) + 1 == 1:
        axes = [axes]

    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title(f"🔍 query image:\n{os.path.basename(query_path)}", fontsize=12, fontweight='bold')
    axes[0].axis('off')  # 关闭坐标轴

    # 2. 显示检索结果
    for i, img_id in enumerate(ids):
        meta = metadatas[i]
        dist = distances[i]

        res_img_path = meta['source']
        res_img = Image.open(res_img_path)

        axes[i + 1].imshow(res_img)
        # 标题显示文件名和相似度距离 (CLIP 余弦距离越小越相似)
        title_text = f"result {i + 1}\n{meta['filename']}\nDistance: {dist:.4f}"
        axes[i + 1].set_title(title_text, fontsize=10)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
    # ===========================================


def main():
    vectordb, model = initialize_vector_db()
    if not vectordb:
        return

    search_and_show(vectordb, model, QUERY_IMAGE_PATH, top_k=3)  # 这里设置显示几张结果


if __name__ == "__main__":
    main()