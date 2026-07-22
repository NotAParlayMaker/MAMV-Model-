"""Offline mechanical coverage for the LoRA training/save path.

This deliberately constructs a tiny random GPT-2 instead of downloading the configured
Qwen checkpoint, so it is safe for the ordinary CI job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.training


def test_tiny_random_causal_lm_lora_train_and_save(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("tokenizers")
    from peft import LoraConfig, TaskType, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        DataCollatorForLanguageModeling,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    tokenizer_backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "blue": 2}, unk_token="[UNK]")
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(tokenizer), n_positions=16, n_embd=16, n_layer=1, n_head=1
        )
    )
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, task_type=TaskType.CAUSAL_LM))
    data = datasets.Dataset.from_dict({"input_ids": [[2, 2]], "attention_mask": [[1, 1]]})
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "run"),
            per_device_train_batch_size=1,
            max_steps=1,
            report_to=[],
            save_strategy="no",
            logging_strategy="no",
        ),
        train_dataset=data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    output = tmp_path / "adapter"
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    assert (output / "adapter_config.json").is_file()
    assert any(path.stat().st_size > 100 for path in output.glob("adapter_model.*"))
    assert (output / "tokenizer_config.json").is_file()
