Playing around with the new bolt chronos. Trying to fine tune it with the engression loss. 

Setup:
```sh
uv sync
uv sync --extra training --no-dev 
uv run scripts/get_finetuning_ds.py
```


Notes:
- validation mean wql for plain bolt: 417_520.69