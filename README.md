Playing around with the new bolt chronos. Trying to fine tune it with the engression loss. 

Setup:
```sh
uv sync
uv sync --extra training --no-dev 
uv run scripts/get_finetuning_ds.py
```

```sh 

```

```sh
uv run scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml tmp.csv \
    --chronos-model-id "./output/run-1/checkpoint-final" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 64
```

Notes:
- validation mean wql for plain bolt: 417_520.69 (tmixup 10m only)
