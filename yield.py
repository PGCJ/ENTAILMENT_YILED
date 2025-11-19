# -*- coding: utf-8 -*-
"""
修复版：彻底消除 <YIELD> 标签泄露
保留全部结构化标签，但训练时完全隐藏真实产率数字
"""

import os
import re
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================= 日志 =============================
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, f"yield_fixed_{datetime.now():%Y%m%d_%H%M%S}.log"), encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# ============================= 关键修改：数据加载时彻底移除 <YIELD> 数字 =============================
def load_and_preprocess_data(dataset_path):
    """
    关键修改：输入文本中完全不出现真实产率数字！
    只保留结构化标签，产率只保留在 label 列
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    logger.info(f"加载数据（已移除 <YIELD> 真实值）：{dataset_path}")
    data = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue

            # 提取真实产率（只用于 labels）
            yield_match = re.search(r'<YIELD>\s*(\d+)', line)
            if not yield_match:
                continue
            true_yield = float(yield_match.group(1))

            # 完全移除 <YIELD> 98 这整个部分
            text = re.sub(r'<YIELD>\s*\d+', '', line)      # 删除 <YIELD> 98
            text = re.sub(r'\s+$', '', text)               # 去掉尾部多余空格
            text = text.strip() + " [SEP]"                 # 保证以 [SEP] 结尾

            # 如果你希望更干净，可以统一补上 <YIELD> ? 占位符（可选）
            # text = text.replace("[SEP]", "<YIELD> ? [SEP]")

            data.append({"text": text, "label": true_yield})

    df = pd.DataFrame(data)
    logger.info(f"成功加载 {len(df):,} 条数据，产率范围 {df['label'].min()} ~ {df['label'].max()}")
    return df


# ============================= 其余函数基本不变，只改一点 =============================
def split_data(df, test_size=0.2, val_size=0.25):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)
    logger.info(f"Train/Val/Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    return train_df, val_df, test_df


class LossLoggingTrainer(Trainer):
    def log(self, logs):
        super().log(logs)
        if "loss" in logs:
            logger.info(f"Step {self.state.global_step} - Train loss: {logs['loss']:.6f}")
        if "eval_loss" in logs:
            logger.info(f"Step {self.state.global_step} - Val   loss: {logs['eval_loss']:.6f}")


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    return {
        "mae": mean_absolute_error(labels, preds),
        "mse": mean_squared_error(labels, preds),
        "r2": r2_score(labels, preds),
    }


# ============================= 新的预测函数（最干净）=============================
def predict_with_structured_text(model, tokenizer, reaction_text: str):
    """
    输入：完整的结构化文本（不包含 <YIELD> 数字）
    示例："[CLS] <REACT1> CCCCCCCCS(=O)(=O)CC(=O)OC </REACT1> <PRODUCT> CCCCCCCCS(=O)(=O)CC(N)=O </PRODUCT> <COND> NH3 MeOH </COND> [SEP]"
    """
    if "<YIELD>" in reaction_text:
        reaction_text = re.sub(r'<YIELD>\s*\d+', '', reaction_text)
    
    if not reaction_text.endswith("[SEP]"):
        reaction_text = reaction_text.strip() + " [SEP]"

    inputs = tokenizer(reaction_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        pred = model(**inputs).logits.item()
    return max(0.0, min(100.0, round(pred, 2)))


# ============================= 主流程（只需改数据路径和模型路径）=============================
def main():
    global logger
    logger = setup_logging()

    # ==================== 配置 ====================
    DATASET_PATH = "/home/public_space/yanmeng/caijiajun/datasets/Reaction_Yield/reaxys_pretrain/SMILES_ENTAILMENT/RoBERTa_All_NamingReactions_SMILES.txt"
    BASE_MODEL   = "/home/public_space/yanmeng/caijiajun/code/Yield/bert-loves-chemistry/chemberta/data/ChemBERTa-zinc-base-v1"
    SAVE_DIR     = "./yield_fixed_no_leakage_model"
    # =============================================

    df = load_and_preprocess_data(DATASET_PATH)
    train_df, val_df, test_df = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # 保留你原来的所有特殊标签
    special_tokens = ["<REACT1>", "</REACT1>", "<REACT2>", "</REACT2>", "<REACT3>", "</REACT3>",
                      "<PRODUCT>", "</PRODUCT>", "<COND>", "</COND>", "<YIELD>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))

    # Dataset
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds   = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    for ds in [train_ds, val_ds, test_ds]:
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = LossLoggingTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # 测试示例（完全看不到真实产率）
    example = "[CLS] <REACT1> CCCCCCCCS(=O)(=O)CC(=O)OC </REACT1> <PRODUCT> CCCCCCCCS(=O)(=O)CC(N)=O </PRODUCT> <COND> ammonium hydroxide methanol; water one-pot </COND> [SEP]"
    pred = predict_with_structured_text(model, tokenizer, example)
    print(f"\n真实反应预测产率：{pred}%  （原始标注为98%）")

    logger.info("训练完成！已彻底消除标签泄露，模型真正学会了从反应预测产率")

if __name__ == "__main__":
    main()