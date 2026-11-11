# Zero-setup path: `make setup` installs the deps into ./.deps and every target
# picks them up. No venv: Debian images without python3-venv have no ensurepip,
# and a --target install works there too.
DEPS ?= .deps
PY   := python3
ifneq ($(wildcard $(DEPS)),)
  PY := PYTHONPATH=$(DEPS) python3
endif
.PHONY: setup up down data index eval ann clean

setup: ## install python deps into ./.deps (works without python3-venv)
	python3 -m pip install -q --upgrade --target $(DEPS) -r requirements.txt
	@echo "deps ready: $(DEPS)  →  make up, then make data index eval"

doctor: ## check container + python deps + database reachability
	@docker compose ps --format 'container: {{.Name}} {{.State}}'
	@$(PY) -c "import psycopg2, fastembed; print('python deps: ok')"
	@$(PY) -c "import psycopg2; psycopg2.connect('postgresql://postgres:lab@127.0.0.1:55432/lab'); print('database: reachable')"

up:   ; python3 -m pip install -q --upgrade --target $(DEPS) -r requirements.txt && echo "deps ready: $(DEPS)  (next: make up, make data index eval)"
doctor:  ; docker compose ps --format '{{.Name}} {{.State}}' | sed 's/^/container: /'; $(PY) -c "import psycopg2,fastembed;print('python deps: ok')"; $(PY) -c "import psycopg2;psycopg2.connect('postgresql://postgres:lab@127.0.0.1:55432/lab');print('database: reachable')"

up:      ; docker compose up -d
down:    ; docker compose down
data:    ; $(PY) scripts/gen_dataset.py && $(PY) scripts/queries.py && $(PY) scripts/queries_paraphrase.py
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
v2moe:   ; $(PY) scripts/build_index.py items_nomicv2 nomic-ai/nomic-embed-text-v2-moe
filter:  ; $(PY) scripts/eval_filter.py
translate: ; $(PY) scripts/eval_translate.py
dual:    ; $(PY) scripts/eval_dual.py
ragdual: ; $(PY) scripts/eval_rag_dual.py
zoo:      ; $(PY) scripts/eval_model_zoo.py
bootstrap: ; $(PY) scripts/bootstrap_zoo.py
power:   ; $(PY) scripts/eval_power.py
queries: ; $(PY) scripts/gen_queries_power.py
restore:  ; $(PY) scripts/eval_diacritics_restore.py
diacritics: ; $(PY) scripts/eval_diacritics.py
scale:   ; N=100000 $(PY) scripts/scale_test.py
halfvec: ; $(PY) scripts/eval_halfvec.py
