from datasets import load_dataset


def tokenize(task: str, tokenizer, example: dict):
    if task.lower() == "mnli":
        tokenized = tokenizer(example["premise"],
                              example["hypothesis"],
                              truncation=True)
    elif task.lower() == "qnli":
        tokenized = tokenizer(example["question"],
                              example["sentence"],
                              truncation=True)
    else:
        raise NotImplementedError(f"No such task for tokenizer, you passed {task} task")
    return tokenized


def trim_unnecessary_columns(task, dataset):
    tasks = {
        "mnli": {
            "remove_cols": ["premise", "hypothesis"],
            "exclude_key": "test",
        },
        "qnli": {
            "remove_cols": ["question", "sentence"],
            "exclude_key": "test",
        },
    }
    if task.lower() not in tasks:
        raise NotImplementedError(f"No such task, you passed {task} task")

    task_info = tasks[task.lower()]
    remove_cols = task_info["remove_cols"]
    exclude_key = task_info["exclude_key"]

    for key, data in dataset.items():
        if exclude_key not in key:
            data = data.remove_columns(remove_cols + ["idx"])
        else:
            data = data.remove_columns(remove_cols)
        dataset[key] = data

    return dataset

def change_labels(example):
    example['label'] = 1
    return example

def get_processed_dataset(tokenizer, task: str, seed: int = 42):
    if task.lower() == "mnli":
        dataset = load_dataset("glue", "mnli")
    elif task.lower() == "qnli":
        dataset = load_dataset("glue", "qnli")
    else:
        raise NotImplementedError(f"No such task for tokenizer, you passed {task} task")
    dataset["train"] = dataset["train"].shuffle(seed=seed)
    tokenized_dataset = dataset.map(lambda x: tokenize(task, tokenizer, x), batched=True)
    tokenized_dataset = trim_unnecessary_columns(task, tokenized_dataset)
    return tokenized_dataset
