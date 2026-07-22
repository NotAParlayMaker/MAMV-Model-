#!/usr/bin/env python3
"""Fine-tune a real HF causal-LM checkpoint with LoRA/QLoRA and Trainer."""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from mamv_model.provenance import (
    checkpoint_manifest,
    environment,
    file_hash,
    fingerprint,
    git_sha,
    timestamp,
    write_json,
)
from mamv_model.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume-from-checkpoint")
    args = p.parse_args()
    c = load_config(args.config)
    random.seed(c.training.seed)
    output_dir = Path(c.training.output_dir)
    start = timestamp()
    config_hashes = {str(Path(args.config).name): file_hash(args.config)}
    dataset_manifest_path = Path(c.data["train_file"]).parent / "dataset_manifest.json"
    dataset = (
        __import__("json").loads(dataset_manifest_path.read_text())
        if dataset_manifest_path.is_file()
        else None
    )
    run_id = "run-" + fingerprint({"config": config_hashes, "start": start})[:16]
    if not c.model.base_model:
        raise SystemExit("config.model.base_model is required; no model weights are bundled.")
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    model_kwargs = {"device_map": "auto"} if c.adapter.method == "qlora" else {}
    if c.adapter.method == "qlora":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(
        c.model.base_model, revision=c.model.revision, trust_remote_code=c.model.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        c.model.base_model,
        revision=c.model.revision,
        trust_remote_code=c.model.trust_remote_code,
        **model_kwargs,
    )
    if c.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if c.adapter.enabled:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=c.adapter.r,
                lora_alpha=c.adapter.alpha,
                lora_dropout=c.adapter.dropout,
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    data = load_dataset(
        "json", data_files={"train": c.data["train_file"], "validation": c.data["validation_file"]}
    )
    template = c.data.get(
        "prompt_template", "Document:\n{document}\n\nQuestion: {question}\nAnswer: {answer}"
    )

    def tokenize(row):
        if c.data.get("use_chat_template", False):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Answer using only the document evidence."},
                    {
                        "role": "user",
                        "content": f"Document:\n{row['document']}\n\nQuestion: {row['question']}",
                    },
                    {"role": "assistant", "content": row["answer"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            return tokenizer(prompt, truncation=True, max_length=c.data.get("max_length", 1024))
        return tokenizer(
            template.format(**row), truncation=True, max_length=c.data.get("max_length", 1024)
        )

    data = data.map(tokenize, remove_columns=data["train"].column_names)
    training = TrainingArguments(
        output_dir=c.training.output_dir,
        num_train_epochs=c.training.num_train_epochs,
        per_device_train_batch_size=c.training.per_device_train_batch_size,
        per_device_eval_batch_size=c.training.per_device_eval_batch_size,
        learning_rate=c.training.learning_rate,
        gradient_accumulation_steps=c.training.gradient_accumulation_steps,
        fp16=c.training.fp16,
        bf16=c.training.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    resume = args.resume_from_checkpoint or c.training.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(c.training.output_dir)
    tokenizer.save_pretrained(c.training.output_dir)
    import dataclasses

    train_fp, val_fp = file_hash(c.data["train_file"]), file_hash(c.data["validation_file"])
    run = {
        "schema_version": "v1",
        "run_id": run_id,
        "git_commit_sha": git_sha(),
        "command": "scripts/train.py --config " + args.config,
        "config_file_hashes": config_hashes,
        "resolved_configuration": dataclasses.asdict(c),
        "base_model": {"id": c.model.base_model, "revision": c.model.revision},
        "tokenizer": {
            "id": c.model.base_model,
            "revision": c.data.get("tokenizer_revision", c.model.revision),
        },
        "adapter": dataclasses.asdict(c.adapter),
        "dataset_manifest_id": dataset.get("manifest_id") if dataset else None,
        "train_fingerprint": train_fp,
        "validation_fingerprint": val_fp,
        "preprocessing_version": (dataset or {}).get("preprocessing_version"),
        "seed": c.training.seed,
        "environment": environment(),
        "hardware_metadata": {"cuda_available": __import__("torch").cuda.is_available()},
        "start_timestamp": start,
        "completion_timestamp": timestamp(),
        "resume_checkpoint": resume,
        "parent_run_id": None,
        "known_nondeterminism": ["GPU kernels and mixed precision can be nondeterministic."],
        "limitations": ["Dataset manifest is required for full provenance."],
    }
    write_json(output_dir / "training_run.json", run)
    weights = {x.name: file_hash(x) for x in output_dir.glob("adapter_model.*")}
    checkpoint = checkpoint_manifest(
        parent_training_run=run_id,
        base_model=run["base_model"],
        adapter_method=c.adapter.method,
        global_step=trainer.state.global_step,
        epoch=trainer.state.epoch,
        dataset_fingerprint=train_fp,
        tokenizer_identity=run["tokenizer"],
        training_config_hash=fingerprint(config_hashes),
        weight_file_hashes=weights,
        evaluation_references=[],
        creation_timestamp=timestamp(),
        adapter_only=c.adapter.enabled,
        base_model_required=c.adapter.enabled,
    )
    write_json(output_dir / "checkpoint_manifest.json", checkpoint)


if __name__ == "__main__":
    main()
