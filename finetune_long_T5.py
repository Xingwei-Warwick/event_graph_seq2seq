import json
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from tqdm import tqdm, trange
from datasets import Dataset
import nltk
import numpy as np
from evaluate import load
import random
from itertools import permutations
from argparse import ArgumentParser
import pandas as pd


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True) # many are exceeding the max_length by a large margin

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def add_permutation(train_dict, upper_limit=5):
    # maximum permutation is the upper_limit
    new_train_dict = {}
    #tk_len_list = []
    for doc_id in train_dict:
        count = 0
        origin_target = train_dict[doc_id]['target']
        #tk_len_list.append(len(nltk.word_tokenize(origin_target)))
        edge_list = origin_target[14:-1].split('\n')
        for permu in permutations(edge_list):
            if count >= upper_limit:
                break
            new_train_dict[f"{doc_id}+{count}"] = {
                "document": train_dict[doc_id]['document'],
                "target": "strict graph {\n" + '\n'.join(permu) + "\n}"
            }
            count += 1

    return new_train_dict


if __name__ == "__main__":
    parser = ArgumentParser(description='Finetune models')
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="model name")
    parser.add_argument("--train-path", type=str, default="data/NYT_xml_DOT_pair_new.json", help="training data path")
    parser.add_argument("--test-path", type=str, default="data/NYT_test_DOT_pair_new.json", help="testing data path")
    parser.add_argument("--aug", type=int, default=5, help="number of augmentation")
    parser.add_argument("--batch", type=int, default=2, help="batch size")
    parser.add_argument("--epoch", type=int, default=10, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="batch size")
    parser.add_argument("--load-checkpoint", type=str, default="None", help="train from checkpoint")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    dataset_name = args.train_path.split("/")[-1].split("_")[0]
    
    max_input_length = 2048
    max_target_length = 768

    # get rid of say, told events?????????????????
    with open(args.train_path, 'r') as f:
        train_data = json.loads(f.read())
    with open(args.test_path, 'r') as f:
        test_data = json.loads(f.read())

    
    ################ augmentation
    if args.aug == 0:
        save_name = f"{model_name}-finetuned-{dataset_name}_noaug"
    else:
        save_name = f"{model_name}-finetuned-{dataset_name}_aug{args.aug}"
        train_data = add_permutation(train_data, args.aug)

    print("Training documents:")
    print(len(train_data))
    input_dict = {
        'document': [train_data[d]['document'] for d in train_data],
        'summary': [train_data[d]['target'] for d in train_data],
        'id': [i for i in range(len(train_data.keys()))]
    }
    
    #with open('data/torque_dev_input_output_DOT.json', 'r') as f:
    #    torque_dev = json.loads(f.read())
    print("Validation documents:")
    print(len(test_data))
    dev_input_dict = {
        'document': [test_data[d]['document'] for d in test_data],
        'summary': [test_data[d]['target'] for d in test_data],
        'id': [i for i in range(len(test_data.keys()))]
    }

    metric = load("rouge")  
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    torque_train_dataset = Dataset.from_dict(input_dict)
    #print(torque_train_dataset)
    tokenized_datasets = torque_train_dataset.map(preprocess_function, batched=True)

    dev_dataset = Dataset.from_dict(dev_input_dict)
    #print(torque_train_dataset)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)

    #print(tokenized_datasets)
    model = LongT5ForConditionalGeneration.from_pretrained(args.model)

    
    seq2seq_args = Seq2SeqTrainingArguments(
        save_name,
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epoch,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        report_to="wandb",
        optim="adamw_torch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        seq2seq_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_dev_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.load_checkpoint=="None":
        trainer.train()
    else:
        trainer.train(args.load_checkpoint)

    trainer.save_model(f"{save_name}-final")