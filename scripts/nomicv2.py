"""nomic-embed-text-v2-moe loader (no ONNX build published, so PyTorch CPU it is).

Mean pooling + L2 normalisation, per the model card's 1_Pooling config. The model
requires a task instruction prefix ("search_query: " / "search_document: "), which
the caller supplies; without it the model is being used outside its training shape.
"""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "nomic-ai/nomic-embed-text-v2-moe"
_cache = {}

def _load():
    if not _cache:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
        _cache["tok"], _cache["mdl"] = tok, mdl
    return _cache["tok"], _cache["mdl"]

def embed(texts, batch=16):
    tok, mdl = _load()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            hidden = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.extend(pooled.float().numpy())
    return out
