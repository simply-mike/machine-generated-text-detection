# Data

This project expects the SemEval 2024 Task 8 / Subtask A multilingual data in the following layout:

```text
SemEval2024-task8/
└── subtaskA/
    └── data/
        ├── subtaskA_train_multilingual.jsonl
        └── subtaskA_dev_multilingual.jsonl
```

Expected fields:

- `text`
- `label`
- `model`
- `source`
- `id`

The dataset is not stored in this repository. Download it from the official SemEval 2024 Task 8 source and place the files locally using the structure above.
