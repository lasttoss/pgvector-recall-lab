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
