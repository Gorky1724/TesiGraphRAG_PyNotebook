"""
Ingestione di un dataset gia' chunkato (JSON) + tag dei chunk nel sidecar, in un solo passaggio.
Salvare come ~/tesi_graphrag/src/evaluation/json_ingest.py

Formato chunk: [{"chunk_id", "doc_id", "true_domain", "chunk_index", "tags": [...], "text"}]
I tag NON vanno nel payload Qdrant (convenzione del progetto): finiscono in sidecar["tag_overrides"][chunk_id]["user_tags"],
e quindi in global_tags (colori inclusi), esattamente come farebbe il pre-tagging ma senza TagAssigner.
"""
import json, sys, time, uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("~/tesi_graphrag").expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDING_MODEL, QDRANT_URL
from src.sidecar_manager import SidecarManager


def load_tags_into_sidecar(chunks, sidecar_path, reset_tags=True) -> SidecarManager:
    """Scrive i tag di ogni chunk nel sidecar. reset_tags=True azzera prima tag_overrides e global_tags
    (le distanze e le predq gia' presenti NON vengono toccate)."""
    sm = SidecarManager(filepath=sidecar_path)
    if reset_tags:
        data = sm.load_data(force_reload=True)
        data["tag_overrides"], data["global_tags"] = {}, {}
        sm.save_data(data)
    sm.add_tag_overrides_batch({c["chunk_id"]: list(c["tags"]) for c in chunks})
    return sm


def ingest_chunks_json(chunks_path, collection_name, sidecar_path, recreate=True, batch_size=64, embeddings=None, client=None):
    from langchain_core.documents import Document
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    chunks = json.load(open(Path(chunks_path).expanduser(), encoding="utf-8"))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id duplicati"

    client = client or QdrantClient(url=QDRANT_URL)
    embeddings = embeddings or FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    dim = len(embeddings.embed_query("test"))
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    if not client.collection_exists(collection_name):
        client.create_collection(collection_name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    docs = [Document(page_content=c["text"], metadata={
        "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "shortened_doc_id": c["doc_id"],
        "true_domain": c.get("true_domain"), "chunk_index": c["chunk_index"], "user_tags": []}) for c in chunks]
    store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch_size):
        b = docs[i:i + batch_size]
        store.add_documents(b, ids=[str(uuid.uuid5(uuid.NAMESPACE_DNS, d.metadata["chunk_id"])) for d in b])

    reg_path = PROJECT_ROOT / "data/collection_registry.json"
    reg = json.load(open(reg_path, encoding="utf-8")) if reg_path.exists() else {}
    reg[collection_name] = {"embedding_model": EMBEDDING_MODEL, "vector_dimension": dim, "distance_metric": "COSINE",
                            "total_chunks": len(docs), "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(reg, open(reg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    sm = load_tags_into_sidecar(chunks, sidecar_path)
    print(f"!>> {len(docs)} chunk in '{collection_name}' ({time.perf_counter()-t0:.1f}s); tag nel sidecar: {list(sm.get_global_tags())}")
    return sm
