import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer 
from para import *
import argparse
import pandas as pd

def tuning(args):
    model_path = backbone_path[args.backbone]
    df = pd.read_csv(f'{args.input}')
    # if args.samples > df.shape[0]:
    #     samples = df.shape[0]
    dataset = load_dataset('csv', data_files = f'{args.input}', split='train')
    # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)  # 获取4位计算的数据类型

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,  # 设置是否以4位加载模型
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,  # 设置4位量化类型
    #     bnb_4bit_compute_dtype=compute_dtype,  # 设置计算时的数据类型
    #     bnb_4bit_use_double_quant=use_nested_quant,  # 设置是否使用嵌套量化
    # )
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )
    model = AutoModelForCausalLM.from_pretrained( 
        model_path,
        device_map = 'auto'
    )
    model.config.use_cache = False  # 设置不使用缓存
    model.config.pretraining_tp = 1  # 设置预训练时间点
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)  # 加载对应模型的分词器
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌为结束令牌
    tokenizer.padding_side = "right"  # 设置填充方式为右侧，解决fp16训练时的溢出问题
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,  # LoRA的alpha参数
        lora_dropout=lora_dropout,  # LoRA层的dropout概率
        r=lora_r,  # LoRA的秩
        bias="none",  # 设置偏置
        task_type="CAUSAL_LM",  # 任务类型为因果语言模型
    )

    ## 设置训练参数
    training_arguments = TrainingArguments(
        output_dir=args.output,  # 输出目录
        num_train_epochs=num_train_epochs,  # 训练周期数
        per_device_train_batch_size = args.batch_size,
        
        gradient_accumulation_steps=gradient_accumulation_steps,  # 梯度累积步数
        optim=optim,  # 优化器
        save_steps=save_steps,  # 保存步数
        logging_steps=logging_steps,  # 记录步数
        learning_rate=learning_rate,  # 学习率
        weight_decay=weight_decay,  # 权重衰减
        fp16=fp16,  # 是否启用fp16
        bf16=bf16,  # 是否启用bf16
        max_grad_norm=max_grad_norm,  # 最大梯度范数
        max_steps=max_steps,  # 最大步数
        warmup_ratio=warmup_ratio,  # 热身比例
        group_by_length=False,  # 是否按长度分组序列
        lr_scheduler_type=lr_scheduler_type,  # 学习率调度器类型
    )

    ## 设置监督式微调参数
    trainer = SFTTrainer(
        model=model,  # 使用的模型
        train_dataset=dataset,  # 训练数据集
        peft_config=peft_config,  # PEFT配置
        dataset_text_field="prompt",  # 数据集文本字段
        max_seq_length=max_seq_length,  # 最大序列长度
        tokenizer=tokenizer,  # 分词器
        args=training_arguments,  # 训练参数
        packing=packing,  # 是否打包
    )

    trainer.train()
    trainer.model.save_pretrained(os.path.join('./tune_model', f'{args.backbone}_{args.new_model}'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./tune_data/base_v4.csv',
                        help='dataset.')
    parser.add_argument('--backbone', type=str, default='galactica6.7b',
                        help='backbone model')
    parser.add_argument('--new_model', type=str, default='base',
                        help='new model name')
    parser.add_argument('--output', type=str, 
                        help='output path')
    parser.add_argument('--config_path', type=str, default='./config.json')
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=8)
    

    args = parser.parse_args()
    args.output = f'./checkpoints/{args.backbone}_{args.new_model}'
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    backbone_path = config['backbone_path']

    tuning(args)