# Model files

Fine-tuned model weights are intentionally not stored in this repository.

Expected local structure:

```text
models/
└── mdeberta_processed_article/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

Save the best fine-tuned model from the notebook:

```python
model.save_pretrained("models/mdeberta_processed_article")
tokenizer.save_pretrained("models/mdeberta_processed_article")
```

The FastAPI service reads the model path from the `MODEL_PATH` environment variable.
By default it expects:

```text
models/mdeberta_processed_article
```
