import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from ModelArguments import ModelArguments
from DataTrainingArguments import DataTrainingArguments

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_dataset(
    args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
        )

def main():

    model_args = ModelArguments(
        model_name_or_path="gpt2", model_type="gpt2"
    )
    data_args = DataTrainingArguments(
        train_data_file="6_genre_clean_training_data.txt",
        eval_data_file="6_genre_eval_data.txt",
        line_by_line=True,
        block_size=512,
        overwrite_cache=True,
    )
    training_args = TrainingArguments(
        output_dir="story_generator_checkpoint",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=500,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=1,
        save_steps=1000,
    )

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


    set_seed(training_args.seed)


    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<superhero>",
            "<action>",
            "<drama>",
            "<thriller>",
            "<horror>",
            "<sci_fi>",
        ],
    }

    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)


    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Training
    try:
      if training_args.do_train:
          model_path = (
              model_args.model_name_or_path
              if model_args.model_name_or_path is not None
              and os.path.isdir(model_args.model_name_or_path)
              else None
          )
          trainer.train(model_path=model_path)
          trainer.save_model()
          tokenizer.save_pretrained(training_args.output_dir)
    except KeyboardInterrupt:
      print("Saving model that was in the middle of training")
      trainer.save_model()
      tokenizer.save_pretrained(training_args.output_dir)
      return

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results

if __name__ == "__main__":
    main()