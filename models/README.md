# Model files

Fine-tuned model weights are intentionally not stored in this repository.

The current API serves the custom `EncoderClassifier` checkpoint produced by the training notebook. The expected local structure is:

```text
checkpoints/
└── task7/
    └── article_style/
        └── mdeberta_processed_article/
            └── best_model.pt
```

The notebook writes `best_model.pt` during training. The checkpoint contains:

- `state_dict`
- `model_name`
- `label_names`
- `max_length`
- `dropout`

The FastAPI service reads the encoder from `MODEL_NAME` and the checkpoint from `CHECKPOINT_PATH`.
By default it expects:

```text
MODEL_NAME=microsoft/mdeberta-v3-base
CHECKPOINT_PATH=checkpoints/task7/article_style/mdeberta_processed_article/best_model.pt
```
