import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# 全局设置 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
print(f"设备: {DEVICE}")
SRC_LANG = 'en'
TGT_LANG = 'de'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>']

# 数据加载与预处理
def load_multi30k():
    """从 HuggingFace datasets 加载 Multi30k """
    train_raw = load_dataset("bentrevett/multi30k", split="train")
    val_raw   = load_dataset("bentrevett/multi30k", split="validation")
    test_raw  = load_dataset("bentrevett/multi30k", split="test")
    return train_raw, val_raw, test_raw
def tokenize_en(text):
    # Multi30k英文大多已分词，这里按空格分
    return text.strip().split()

def tokenize_de(text):
    # 德文按空格分
    return text.strip().split()

def build_vocab(sentences, tokenizer):
    """根据句子列表构建词汇映射表"""
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))
    # 按词频排序，保留所有词（小数据集）
    vocab = {word: idx + len(SPECIAL_TOKENS) for idx, (word, _) in enumerate(counter.most_common())}
    # 插入特殊标记
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i
    return vocab

def encode(sentence, tokenizer, vocab):
    """将句子转为索引列表，添加 BOS/EOS"""
    tokens = tokenizer(sentence)
    ids = [BOS_IDX] + [vocab.get(token, UNK_IDX) for token in tokens] + [EOS_IDX]
    return ids

def collate_batch(batch, src_vocab, tgt_vocab):
    """对批次进行填充，返回 src, tgt 张量"""
    src_batch, tgt_batch = [], []
    for item in batch:
        src = encode(item['en'], tokenize_en, src_vocab)
        tgt = encode(item['de'], tokenize_de, tgt_vocab)
        src_batch.append(torch.tensor(src, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt, dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch.to(DEVICE), tgt_batch.to(DEVICE)

def create_dataloaders(batch_size=128):
    """准备训练/验证/测试 DataLoader"""
    train_raw, val_raw, test_raw = load_multi30k()
    # 构建词汇表（只用训练集）
    src_vocab = build_vocab([item['en'] for item in train_raw], tokenize_en)
    tgt_vocab = build_vocab([item['de'] for item in train_raw], tokenize_de)
    print(f"源词汇表大小: {len(src_vocab)}, 目标词汇表大小: {len(tgt_vocab)}")

    train_loader = DataLoader(
        list(train_raw), batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, src_vocab, tgt_vocab)
    )
    val_loader = DataLoader(
        list(val_raw), batch_size=batch_size,
        collate_fn=lambda b: collate_batch(b, src_vocab, tgt_vocab)
    )
    test_loader = DataLoader(
        list(test_raw), batch_size=batch_size,
        collate_fn=lambda b: collate_batch(b, src_vocab, tgt_vocab)
    )
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab

# 模型组件
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """单层编码器，可控制是否使用残差连接"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, residual=True):
        super().__init__()
        self.residual = residual
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力子层
        if self.residual:
            src2 = self.norm1(src)
            src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
            src = src + self.dropout1(src2)
        else:
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
            src = self.norm1(src2)

        # 前馈网络子层
        if self.residual:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = self.norm2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    """单层解码器，同样控制残差连接"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, residual=True):
        super().__init__()
        self.residual = residual
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 自注意力
        if self.residual:
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
            tgt = self.norm1(tgt2)

        # 交叉注意力
        if self.residual:
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.cross_attn(tgt2, memory, memory, attn_mask=memory_mask)
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
            tgt = self.norm2(tgt2)

        # 前馈网络
        if self.residual:
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
            tgt = self.norm3(tgt2)
        return tgt

class Transformer(nn.Module):
    """完整的 Transformer 模型（编码器-解码器），可配置残差"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_layers=3, dim_feedforward=512, dropout=0.1, residual=True):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, residual)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, residual)
            for _ in range(num_layers)
        ])
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """创建解码器自注意力的上三角掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 嵌入 + 位置编码
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        # 编码器
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        # 解码器
        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask, src_mask)

        return self.generator(out)

# 训练与评估 
def train_epoch(model, loader, optimizer, criterion, epoch, epochs):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for src, tgt in loop:
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))
        optimizer.zero_grad()
        pred = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for src, tgt in loader:
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))
        pred = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)

def greedy_decode(model, src, max_len=50, bos_idx=BOS_IDX, eos_idx=EOS_IDX):
    """使用贪心解码生成翻译结果"""
    model.eval()
    src = src.to(DEVICE)
    model = model.to(DEVICE)
    src_emb = model.pos_enc(model.src_embed(src) * math.sqrt(model.d_model))
    memory = src_emb
    for layer in model.encoder_layers:
        memory = layer(memory, src_mask=None)
    ys = torch.ones(src.size(0), 1, dtype=torch.long, device=DEVICE).fill_(bos_idx)
    for i in range(max_len - 1):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1))
        # 对 token IDs 进行嵌入和位置编码
        out = model.pos_enc(model.tgt_embed(ys) * math.sqrt(model.d_model))
        for layer in model.decoder_layers:
            out = layer(out, memory, tgt_mask, memory_mask=None)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        if (next_word == eos_idx).all():
            break
    return ys

def compute_bleu(model, loader, tgt_vocab):
    """计算整个数据集的 BLEU 分数"""
    bleu = BLEU()
    references = []
    hypotheses = []
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    for src, tgt in loader:
        pred_ids = greedy_decode(model, src)
        for i in range(pred_ids.size(0)):
            # 去掉 BOS/EOS，还原为单词
            pred_tokens = [inv_vocab.get(idx, '<unk>') for idx in pred_ids[i].tolist()
                           if idx not in [BOS_IDX, EOS_IDX, PAD_IDX]]
            tgt_tokens = [inv_vocab.get(idx, '<unk>') for idx in tgt[i].tolist()
                          if idx not in [BOS_IDX, EOS_IDX, PAD_IDX]]
            hypotheses.append(' '.join(pred_tokens))
            references.append(' '.join(tgt_tokens))
    return bleu.corpus_score(hypotheses, [references]).score

def run_experiment(exp_name, residual, train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, epochs=20, lr=1e-4):
    """进行一次完整实验，返回训练/验证 loss 历史"""
    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=256, nhead=8, num_layers=3, dim_feedforward=512,
        dropout=0.1, residual=residual
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = None

    print(f"\n=== 实验: {exp_name} ===")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, epochs)
        val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    # 测试集上的最终表现
    model = best_model
    bleu = compute_bleu(model, test_loader, tgt_vocab)
    val_ppl = math.exp(best_val_loss)
    print(f"{exp_name} 最佳验证 Loss: {best_val_loss:.4f}, PPL: {val_ppl:.2f}, BLEU: {bleu:.2f}")
    return train_losses, val_losses, {'val_loss': best_val_loss, 'ppl': val_ppl, 'bleu': bleu}

# 主程序 
if __name__ == "__main__":
    # 准备数据 (为了加快演示，可调小 batch_size)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(batch_size=64)

    # 实验参数
    EPOCHS = 20  # 快速查看趋势，可改为 20 以上获得更好效果

    # 实验 1: Baseline (有残差连接)
    bl_train_loss, bl_val_loss, bl_metrics = run_experiment(
        "Baseline (with residual)",
        residual=True,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        epochs=EPOCHS
    )

    # 实验 2: 无残差连接
    nores_train_loss, nores_val_loss, nores_metrics = run_experiment(
        "No Residual (without residual)",
        residual=False,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        epochs=EPOCHS
    )

    # 绘制损失曲线对比
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, EPOCHS+1)
    plt.plot(epochs_range, bl_train_loss, 'b-', label='Baseline Train Loss')
    plt.plot(epochs_range, bl_val_loss, 'b--', label='Baseline Val Loss')
    plt.plot(epochs_range, nores_train_loss, 'r-', label='NoResidual Train Loss')
    plt.plot(epochs_range, nores_val_loss, 'r--', label='NoResidual Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.title('Impact of Residual Connections on Transformer Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('residual_impact.png')
    plt.show()

    print("\n===== 最终结果对比 =====")
    print(f"Baseline - Val Loss: {bl_metrics['val_loss']:.4f}, PPL: {bl_metrics['ppl']:.2f}, BLEU: {bl_metrics['bleu']:.2f}")
    print(f"NoRes    - Val Loss: {nores_metrics['val_loss']:.4f}, PPL: {nores_metrics['ppl']:.2f}, BLEU: {nores_metrics['bleu']:.2f}")