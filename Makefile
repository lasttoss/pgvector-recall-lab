PY := PYTHONPATH=/tmp/emblibs python3
.PHONY: up down data index eval ann clean
up:      ; docker compose up -d
down:    ; docker compose down
data:    ; $(PY) scripts/gen_dataset.py && python3 scripts/queries.py && python3 scripts/queries_paraphrase.py
index:   ; $(PY) scripts/build_index.py
eval:    ; $(PY) scripts/eval.py out/queries.json && $(PY) scripts/eval.py out/queries_paraphrase.json
ann:     ; $(PY) scripts/eval_ann.py
clean:   ; docker compose down -v
