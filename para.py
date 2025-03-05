lora_r = 256  # LoRA的秩，这里设置为64
lora_alpha = 128  # LoRA的缩放参数，这里设置为16
lora_dropout = 0.1  # 设置LoRA层的dropout概率为0.1
use_4bit = True  
bnb_4bit_compute_dtype = "float16" 
bnb_4bit_quant_type = "nf4"  # 设置4位量化类型为nf4
use_nested_quant = False  # 设置不使用嵌套量化

num_train_epochs = 1  # 训练 epochs
fp16 = True  # 启用fp16/bf16训练（如果是A100，将bf16设置为True），设置不使用fp16
bf16 = False  # 设置不使用bf16
per_device_train_batch_size = 188  # 设置每个GPU的训练批量大小为4
per_device_eval_batch_size = 188  # 设置每个GPU的评估批量大小为4
gradient_accumulation_steps = 1 # 累积梯度的更新步数
gradient_checkpointing = True  #启用梯度检查点
max_grad_norm = 0.3  # 最大梯度范数（梯度裁剪）
learning_rate = 1e-4  # 初始学习率（AdamW优化器）
weight_decay = 0.001  # 除偏置/LayerNorm权重外应用的权重衰减
optim = "paged_adamw_32bit" # 使用的优化器
lr_scheduler_type = "constant" # 学习率调度（常数略好于余弦）
max_steps = -1  # 设置训练步数为-1（表示不覆盖）
warmup_ratio = 0.03  # 线性热身的步数比例（从0到学习率）

save_steps = 100000  # 保存步数
logging_steps = 10  # 记录步数
group_by_length = False
max_seq_length = 188
packing = False
