PY := PYTHONPATH=/tmp/emblibs python3
.PHONY: up down data index eval ann clean
up:      ; docker compose up -d
down:    ; docker compose down
data:    ; $(PY) scripts/gen_dataset.py && python3 scripts/queries.py && python3 scripts/queries_paraphrase.py
index:   ; $(PY) scripts/build_index.py
eval:    ; $(PY) scripts/eval.py out/queries.json && $(PY) scripts/eval.py out/queries_paraphrase.json
ann:     ; $(PY) scripts/eval_ann.py
clean:   ; docker compose down -v
models:  ; $(PY) scripts/build_index.py items_potion minishlab/potion-multilingual-128M && $(PY) scripts/build_index.py items_e5large intfloat/multilingual-e5-large
compare: ; for m in "items|sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" "items_potion|minishlab/potion-multilingual-128M" "items_e5large|intfloat/multilingual-e5-large"; do t=$${m%%|*}; f=$${m##*|}; $(PY) scripts/eval.py out/queries_paraphrase.json $$t $$f; done
rerank:  ; $(PY) scripts/eval_rerank_jev.py out/queries_paraphrase.json items_potion minishlab/potion-multilingual-128M && $(PY) scripts/eval_rerank_jev.py out/queries_paraphrase.json items_e5large intfloat/multilingual-e5-large
rerank-choice: ; $(PY) scripts/eval_rerank_jev_choice.py out/queries_paraphrase.json items_e5large intfloat/multilingual-e5-large
nomic:   ; $(PY) scripts/build_index.py items_nomic nomic-ai/nomic-embed-text-v1.5 && $(PY) scripts/build_index.py items_nomicq nomic-ai/nomic-embed-text-v1.5-Q
# v2-moe has no ONNX build: needs PyTorch + transformers 4.x, and remote code
v2moe:   ; PYTHONPATH=/tmp/emblibs-t4:/tmp/emblibs $(PY) scripts/build_index.py items_nomicv2 nomic-ai/nomic-embed-text-v2-moe
filter:  ; $(PY) scripts/eval_filter.py
