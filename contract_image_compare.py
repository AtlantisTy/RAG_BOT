import argparse
import base64
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from PIL import Image
except ModuleNotFoundError as e:
    raise SystemExit(
        "❌ 缺少依赖：pillow（PIL）。\n"
        f"当前 Python：{sys.executable}\n"
        "请在同一个环境里安装依赖后再运行：\n"
        "  pip install -r requirements.txt\n"
        "或直接用虚拟环境解释器运行：\n"
        r"  .\.venv\Scripts\python.exe contract_image_compare.py"
    ) from e

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit(
        "❌ 缺少依赖：numpy。\n"
        f"当前 Python：{sys.executable}\n"
        "请先在当前 Python 环境安装依赖后再运行：\n"
        "  pip install -r requirements.txt\n"
        "或直接用虚拟环境解释器运行：\n"
        r"  .\.venv\Scripts\python.exe contract_image_compare.py"
    ) from e

from dotenv import load_dotenv

try:
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as e:
    raise SystemExit(
        f"❌ 缺少依赖：{e.name}。\n"
        f"当前 Python：{sys.executable}\n"
        "请先在当前 Python 环境安装依赖后再运行：\n"
        "  pip install -r requirements.txt\n"
        "或直接用虚拟环境解释器运行：\n"
        r"  .\.venv\Scripts\python.exe contract_image_compare.py"
    ) from e


# ================ 默认配置（可用 CLI 覆盖） ================
IMAGE_LIBRARY_PATH = "./images_db"
PERSIST_DIRECTORY = "./chroma_img_storage"
QUERY_IMAGE_PATH = "./contract1.png"
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

# DeepSeek/OpenAI 兼容配置
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEFAULT_LLM_MODEL = "kimi"
# =========================================================


@dataclass
class CandidateEvidence:
    filename: str
    source: str
    clip_distance: Optional[float]
    clip_cosine_sim: Optional[float]
    sha256_equal: bool
    phash_hamming: int
    pixel_mse: float
    same_size: bool
    content_equivalent_likely: bool
    content_equivalent_score: float
    ocr_text_similarity: Optional[float] = None
    query_ocr_text_preview: Optional[str] = None
    cand_ocr_text_preview: Optional[str] = None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dct_2d(a: np.ndarray) -> np.ndarray:
    # DCT-II with numpy only (via FFT trick is overkill here); use separability with matrix multiplication.
    # This is small (32x32), so a direct cosine matrix is fine and dependency-free.
    n = a.shape[0]
    x = np.arange(n)
    k = x.reshape(-1, 1)
    cos = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    alpha = np.ones((n, 1))
    alpha[0, 0] = 1.0 / np.sqrt(2)
    c = np.sqrt(2 / n) * alpha * cos
    return c @ a @ c.T


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """
    Simple perceptual hash (pHash) using DCT.
    Returns a boolean array of shape (hash_size, hash_size).
    """
    img_size = hash_size * highfreq_factor
    img = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    dct = _dct_2d(pixels)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq[1:, 1:])  # ignore DC component area for threshold
    return dctlowfreq > med


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.flatten() ^ b.flatten()))


def pixel_mse(img_a: Image.Image, img_b: Image.Image, resize_to: Tuple[int, int] = (512, 512)) -> Tuple[float, bool]:
    same_size = img_a.size == img_b.size
    a = img_a.convert("RGB").resize(resize_to, Image.Resampling.LANCZOS)
    b = img_b.convert("RGB").resize(resize_to, Image.Resampling.LANCZOS)
    arr_a = np.asarray(a, dtype=np.float32) / 255.0
    arr_b = np.asarray(b, dtype=np.float32) / 255.0
    mse = float(np.mean((arr_a - arr_b) ** 2))
    return mse, same_size


def try_ocr_text(image_path: str) -> Optional[str]:
    """
    Optional OCR. If user has pytesseract + tesseract installed, we use it.
    Otherwise return None silently.
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        text = " ".join(text.split())
        return text if text.strip() else None
    except Exception:
        return None


def text_jaccard(a: str, b: str) -> float:
    # char 2-gram Jaccard for Chinese/English mixed text; robust to whitespace.
    def grams(s: str) -> set:
        s = "".join(ch for ch in s if not ch.isspace())
        if len(s) < 2:
            return {s} if s else set()
        return {s[i : i + 2] for i in range(len(s) - 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def list_images(root: str) -> List[str]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(root)
    if not p.exists():
        return []
    return [str(f.resolve()) for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]


def build_or_load_vectordb(
    image_root: str,
    persist_directory: str,
    embedding_model_name: str,
    rebuild: bool = False,
) -> Tuple[Chroma, SentenceTransformer]:
    if not os.path.exists(image_root):
        os.makedirs(image_root)
        raise FileNotFoundError(f"图片库目录不存在，已创建：{image_root}。请放入图片后重试。")

    image_files = list_images(image_root)
    if not image_files:
        raise FileNotFoundError(f"图片库目录 {image_root} 下没有图片。")

    model = SentenceTransformer(embedding_model_name, device="cpu")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=None)
    collection = vectordb._client.get_collection(name=vectordb._collection.name)

    existing_count = 0
    try:
        existing_count = int(collection.count())
    except Exception:
        existing_count = 0

    if existing_count > 0 and not rebuild:
        return vectordb, model

    # 重建/初始化向量库
    vectors: List[List[float]] = []
    valid_paths: List[str] = []
    images: List[Image.Image] = []
    for p in image_files:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception:
            continue

    if not images:
        raise RuntimeError("图片全部无法读取，无法建立向量库。")

    emb = model.encode(images, batch_size=16, show_progress_bar=True, convert_to_numpy=True).tolist()
    vectors.extend(emb)

    ids = [Path(p).name for p in valid_paths]
    metadatas = [{"source": p, "filename": Path(p).name} for p in valid_paths]
    documents_content = [f"Image: {m['filename']}" for m in metadatas]

    # 清空同名 id（防止重复运行）
    try:
        existing_ids = collection.get(ids=ids)["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
    except Exception:
        pass

    collection.add(ids=ids, embeddings=vectors, documents=documents_content, metadatas=metadatas)
    return vectordb, model


def retrieve_similar_images(
    vectordb: Chroma,
    model: SentenceTransformer,
    query_path: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"原文件不存在：{query_path}")

    img = Image.open(query_path).convert("RGB")
    query_vector = model.encode([img], convert_to_numpy=True)[0].tolist()

    collection = vectordb._client.get_collection(name=vectordb._collection.name)
    # 注意：Chroma 的 include 不支持 "ids"，ids 会默认在 results 顶层返回
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["metadatas", "distances", "embeddings"],
    )

    out: List[Dict[str, Any]] = []
    for i, _id in enumerate(results["ids"][0]):
        out.append(
            {
                "id": _id,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "embedding": results["embeddings"][0][i] if results.get("embeddings") else None,
                "query_embedding": query_vector,
            }
        )
    return out


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _content_equivalence_score(
    *,
    phash_hamming: int,
    pixel_mse: float,
    clip_cosine_sim: Optional[float],
    ocr_text_similarity: Optional[float],
) -> float:
    """
    0..1 越大越可能“内容等价”（允许重编码/缩放/轻微噪声）。
    规则偏保守：只有多证据同时支持才会高分。
    """
    # pHash：0 最好；<=6 通常仍可能是重编码/轻微变化；>10 多为明显不同
    if phash_hamming <= 2:
        s_ph = 1.0
    elif phash_hamming <= 6:
        s_ph = 0.8
    elif phash_hamming <= 10:
        s_ph = 0.4
    else:
        s_ph = 0.0

    # MSE：0 最好；<=0.01 常见于轻微变化；<=0.03 可能仍是重保存/尺寸不同
    if pixel_mse <= 0.001:
        s_mse = 1.0
    elif pixel_mse <= 0.01:
        s_mse = 0.8
    elif pixel_mse <= 0.03:
        s_mse = 0.5
    else:
        s_mse = 0.0

    # CLIP：重保存通常仍很高（接近 1.0）。取 0.92/0.95/0.98 三档。
    if clip_cosine_sim is None:
        s_clip = 0.5  # 缺失时不给否决
    elif clip_cosine_sim >= 0.98:
        s_clip = 1.0
    elif clip_cosine_sim >= 0.95:
        s_clip = 0.8
    elif clip_cosine_sim >= 0.92:
        s_clip = 0.5
    else:
        s_clip = 0.0

    # OCR：如果可用，它是“内容一致”的强证据
    if ocr_text_similarity is None:
        s_ocr = 0.5
    elif ocr_text_similarity >= 0.98:
        s_ocr = 1.0
    elif ocr_text_similarity >= 0.95:
        s_ocr = 0.8
    elif ocr_text_similarity >= 0.90:
        s_ocr = 0.5
    else:
        s_ocr = 0.0

    # 权重：OCR（若有）与感知哈希更关键；CLIP 更偏“语义相似”，不等于逐字一致
    score = 0.35 * s_ph + 0.25 * s_mse + 0.20 * s_clip + 0.20 * s_ocr
    return float(max(0.0, min(1.0, score)))


def make_evidence(
    query_path: str,
    candidate_path: str,
    clip_distance: Optional[float],
    query_embedding: Optional[List[float]] = None,
    candidate_embedding: Optional[List[float]] = None,
) -> CandidateEvidence:
    q_img = Image.open(query_path)
    c_img = Image.open(candidate_path)

    q_sha = sha256_file(query_path)
    c_sha = sha256_file(candidate_path)
    sha_equal = q_sha == c_sha

    q_ph = phash(q_img)
    c_ph = phash(c_img)
    ph_dist = hamming_distance(q_ph, c_ph)

    mse, same_size = pixel_mse(q_img, c_img)

    q_text = try_ocr_text(query_path)
    c_text = try_ocr_text(candidate_path)
    ocr_sim = None
    q_preview = None
    c_preview = None
    if q_text is not None and c_text is not None:
        ocr_sim = float(text_jaccard(q_text, c_text))
        q_preview = q_text[:280]
        c_preview = c_text[:280]

    clip_cos = None
    if query_embedding is not None and candidate_embedding is not None:
        try:
            clip_cos = float(_cosine_sim(query_embedding, candidate_embedding))
        except Exception:
            clip_cos = None

    eq_score = _content_equivalence_score(
        phash_hamming=int(ph_dist),
        pixel_mse=float(mse),
        clip_cosine_sim=clip_cos,
        ocr_text_similarity=ocr_sim,
    )
    # 经验阈值：允许“重新保存/缩放/压缩”导致的轻微差异
    # - >=0.65：大概率内容等价
    # - >=0.85：非常接近，可视为强一致（即使 sha/尺寸不同）
    eq_likely = bool(eq_score >= 0.65)

    return CandidateEvidence(
        filename=Path(candidate_path).name,
        source=str(Path(candidate_path).resolve()),
        clip_distance=float(clip_distance) if clip_distance is not None else None,
        clip_cosine_sim=clip_cos,
        sha256_equal=sha_equal,
        phash_hamming=int(ph_dist),
        pixel_mse=float(mse),
        same_size=bool(same_size),
        content_equivalent_likely=eq_likely,
        content_equivalent_score=float(eq_score),
        ocr_text_similarity=ocr_sim,
        query_ocr_text_preview=q_preview,
        cand_ocr_text_preview=c_preview,
    )


def _maybe_image_to_data_url(path: str, max_side: int = 1024, quality: int = 85) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def ask_llm_to_judge(
    llm: ChatOpenAI,
    query_path: str,
    evidences: List[CandidateEvidence],
    include_images: bool = False,
    auto_pass_visual_threshold: float = 0.85,
) -> Dict[str, Any]:
    evidence_dicts: List[Dict[str, Any]] = []
    for e in evidences:
        evidence_dicts.append(
            {
                "filename": e.filename,
                "source": e.source,
                "clip_distance": e.clip_distance,
                "clip_cosine_sim": e.clip_cosine_sim,
                "sha256_equal": e.sha256_equal,
                "phash_hamming": e.phash_hamming,
                "pixel_mse": e.pixel_mse,
                "same_size": e.same_size,
                "content_equivalent_likely": e.content_equivalent_likely,
                "content_equivalent_score": e.content_equivalent_score,
                "ocr_text_similarity": e.ocr_text_similarity,
                "query_ocr_text_preview": e.query_ocr_text_preview,
                "cand_ocr_text_preview": e.cand_ocr_text_preview,
            }
        )

    system = (
        "你是合同图片篡改检测助手。你的任务是判断：候选合同图片是否与原合同图片“内容完全一致”。\n"
        "重要：哪怕只改动了一个数字、日期、金额、印章、签名、页码、条款顺序等，都视为“不一致”。\n"
        "你会拿到向量检索结果以及多种证据：\n"
        "- sha256_equal：只代表字节级文件是否相同。图片“重新保存/压缩/去 EXIF/重编码/改格式/改尺寸”会导致 sha256 不同，但内容可能仍一致，所以它不能作为否决项。\n"
        "- same_size：尺寸变化可能来自重保存/缩放；也不能单独作为否决项。\n"
        "- phash_hamming / pixel_mse / clip_cosine_sim：更贴近视觉内容是否一致。\n"
        "- ocr_text_similarity：若存在，是文字内容一致的重要证据。\n"
        "判定原则（请严格遵守）：\n"
        "1) sha256_equal、same_size 只能作为参考，不能单独否决（重新保存/压缩/缩放会改变它们）。\n"
        "2) 如果某个候选满足“强一致条件”，允许你判定 is_identical=true：\n"
        "   - content_equivalent_score >= 0.65\n"
        "   - clip_cosine_sim >= 0.98（若该字段缺失则降低信心）\n"
        "   - phash_hamming <= 6\n"
        "   - pixel_mse <= 0.03\n"
        "   这通常对应同一内容的重新保存/压缩版本。\n"
        "3) 如果 OCR 可用且 ocr_text_similarity 很高（>=0.98），可以显著提高 confidence 并把 needs_manual_review 设为 false。\n"
        "4) 如果不满足强一致条件，或证据冲突（例如 clip_cosine_sim 高但 phash/mse 明显大），必须判不一致，并 needs_manual_review=true。\n"
        "请输出严格 JSON（不要输出多余文本）。"
    )

    user_payload: Dict[str, Any] = {
        "query_image": str(Path(query_path).resolve()),
        "candidates": evidence_dicts,
        "decision_rule": {
            "identical": "内容完全一致（允许重新保存/压缩/格式变化/尺寸变化/轻微噪声，只要合同条款、数字金额、日期、盖章、签名等内容完全一致即可）",
            "tampered_or_not_identical": "任意内容差异或无法确认逐字一致（尤其涉及金额/日期/手机号/公章/签名）都判不一致，并说明原因",
        },
        "required_output_json_schema": {
            "is_identical": "boolean",
            "matched_filename": "string|null",
            "confidence": "number 0..1",
            "reason": "string",
            "key_evidence": "array of strings",
            "needs_manual_review": "boolean",
        },
    }

    if include_images:
        # 只有在模型/网关支持多模态时才可能有效；否则会被当成普通文本忽略或报错。
        user_payload["query_image_data_url"] = _maybe_image_to_data_url(query_path)
        for idx, e in enumerate(evidences):
            user_payload[f"candidate_{idx+1}_image_data_url"] = _maybe_image_to_data_url(e.source)

    prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    text = getattr(resp, "content", resp)
    if not isinstance(text, str):
        text = str(text)

    # 尝试从返回中提取 JSON（允许模型外层包裹解释文本）
    parsed: Optional[Dict[str, Any]] = None
    try:
        loaded = json.loads(text)
        parsed = loaded if isinstance(loaded, dict) else None
    except Exception:
        # 简易截取第一个 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                loaded = json.loads(text[start : end + 1])
                parsed = loaded if isinstance(loaded, dict) else None
            except Exception:
                parsed = None
        else:
            parsed = None

    if not isinstance(parsed, dict):
        return {
            "is_identical": False,
            "matched_filename": None,
            "confidence": 0.0,
            "reason": "模型返回无法解析为 JSON。",
            "key_evidence": ["raw_response_parse_failed"],
            "needs_manual_review": True,
            "raw_response": text[:2000],
        }

    # -------- 安全阀后处理（防止无 OCR 误放行）--------
    # 业务假设：如果没有 OCR/结构化字段校验，视觉相似只能“高概率一致”，仍建议人工复核。
    # 只有当 OCR 相似度很高，或内容等价分数达到 auto_pass_visual_threshold 时，才可自动放行。
    by_name = {e.filename: e for e in evidences}
    matched = parsed.get("matched_filename")
    if parsed.get("is_identical") is True and isinstance(matched, str) and matched in by_name:
        ev = by_name[matched]
        has_ocr = ev.ocr_text_similarity is not None
        very_high_visual = ev.content_equivalent_score >= float(auto_pass_visual_threshold)

        if (not has_ocr) and (not very_high_visual):
            parsed["needs_manual_review"] = True
            # 保守下调置信度
            try:
                parsed["confidence"] = float(min(float(parsed.get("confidence", 0.0)), 0.75))
            except Exception:
                parsed["confidence"] = 0.75
            parsed["reason"] = (
                str(parsed.get("reason", "")).rstrip()
                + f"（注意：OCR 不可用且视觉分数未达 {auto_pass_visual_threshold:.2f}，已自动转为需要人工复核以降低误放行风险。）"
            )

    return parsed


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="对比两份合同图片内容是否完全一致（防篡改）。")
    parser.add_argument("--query", default=QUERY_IMAGE_PATH, help="原合同图片路径（默认 contract1.png）")
    parser.add_argument("--image_db", default=IMAGE_LIBRARY_PATH, help="对比图片库目录（默认 ./images_db）")
    parser.add_argument("--persist", default=PERSIST_DIRECTORY, help="Chroma 持久化目录（默认 ./chroma_img_storage）")
    parser.add_argument("--top_k", type=int, default=3, help="向量检索召回数量（默认 3）")
    parser.add_argument("--rebuild_index", action="store_true", help="强制重建向量库")
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL_NAME, help="图片 embedding 模型（默认 CLIP ViT-B-32）")
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL, help="LLM 模型名（默认 kimi）")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM 温度（默认 0.1）")
    parser.add_argument("--include_images", action="store_true", help="尝试把图片以 data url 形式发给模型（需要网关支持多模态）")
    parser.add_argument(
        "--auto_pass_visual_threshold",
        type=float,
        default=0.85,
        help="无 OCR 时，达到该视觉分数阈值才允许自动放行（needs_manual_review=false）。默认 0.85（更安全）。",
    )
    args = parser.parse_args()

    api_key = os.getenv(DEEPSEEK_API_KEY_ENV)
    base_url = os.getenv(DEEPSEEK_BASE_URL_ENV)
    if not api_key:
        print(f"❌ 未找到环境变量 {DEEPSEEK_API_KEY_ENV}，请在 .env 中配置。")
        return 2

    vectordb, clip_model = build_or_load_vectordb(
        image_root=args.image_db,
        persist_directory=args.persist,
        embedding_model_name=args.embedding_model,
        rebuild=args.rebuild_index,
    )

    retrieved = retrieve_similar_images(vectordb, clip_model, args.query, top_k=args.top_k)
    if not retrieved:
        print("❌ 未检索到候选图片。")
        return 3

    evidences: List[CandidateEvidence] = []
    for r in retrieved:
        meta = r["metadata"]
        cand_path = meta["source"]
        evidences.append(
            make_evidence(
                args.query,
                cand_path,
                r.get("distance"),
                query_embedding=r.get("query_embedding"),
                candidate_embedding=r.get("embedding"),
            )
        )

    llm = ChatOpenAI(
        model=args.llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=800,
    )

    result = ask_llm_to_judge(
        llm,
        args.query,
        evidences,
        include_images=bool(args.include_images),
        auto_pass_visual_threshold=float(args.auto_pass_visual_threshold),
    )
    print("\n===== LLM 判定结果（JSON）=====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n===== 检索候选证据（本地计算）=====")
    for e in evidences:
        print(
            f"- {e.filename} | clip_distance={e.clip_distance:.4f} | sha256_equal={e.sha256_equal} "
            f"| phash_hamming={e.phash_hamming} | pixel_mse={e.pixel_mse:.6f} | same_size={e.same_size}"
            + (f" | clip_cos={e.clip_cosine_sim:.4f}" if e.clip_cosine_sim is not None else "")
            + f" | eq_score={e.content_equivalent_score:.2f} | eq_likely={e.content_equivalent_likely}"
            + (f" | ocr_sim={e.ocr_text_similarity:.3f}" if e.ocr_text_similarity is not None else "")
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

