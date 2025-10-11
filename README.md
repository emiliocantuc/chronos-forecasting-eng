Playing around with the new bolt chronos. Trying to fine tune it with the engression loss. 

Setup:
```sh
uv sync
uv sync --extra training --no-dev 
uv run scripts/get_finetune_ds.py
```

```sh 
uv run scripts/training/train_bolt.py --config scripts/training/configs/bolt-t5-tiny-eng-finetune.yaml
```

```sh
uv run scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml tmp.csv \
    --chronos-model-id "./output/run-1/checkpoint-final" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 256
```

Notes:

validation mean wql for plain bolt:
- kernelsynth only: 10
- tmixup only: 10


- $$\beta < 1$$ seems to stabilize training. TODO: play with $$p$$ as well.  

