基于你澄清后的需求，我设计了一套文档级多标签分类的混合架构。这个架构的核心是：将大标签空间转化为检索排序问题，利用LLM构建数据飞轮，用多任务学习聚合1000个问答对的文档信号，并通过持续学习迭代进化。

---

一、架构总览

```
                    ┌─────────────────────────────────────────────────┐
                    │              数据飞轮（LLM + RAG）              │
                    │  · 种子文档 → LLM 生成更多文档 + 标签          │
                    │  · 标签描述向量化 → FAISS 索引                  │
                    └─────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         多任务文档编码器                             │
│  ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────┐ │
│  │ 问答对编码   │ → │ 池化聚合  │ → │ 共享文档  │   │               │ │
│  │ (BERT/RoBERTa)│  │ (Attn/Mean)│  │ 表示 h    │   │               │ │
│  └─────────────┘   └──────────┘   └────┬─────┘   │               │ │
│                                        │         │               │ │
│         ┌────────────────┬─────────────┼─────────┬─────────────┐ │ │
│         ▼                ▼             ▼         ▼             ▼ │ │
│  [小类别softmax头]  [小类别softmax头]  [大类别]  [大类别]  ...   │ │
│   cat1 (≤20值)       cat2 (≤20值)    cat3 (≈1000值)  cat4 ...   │ │
│                                       │                          │ │
│                            ┌──────────▼──────────┐               │ │
│                            │ 双塔检索 + 交叉排序  │               │ │
│                            │ h → ANN 检索 Top-K   │               │ │
│                            │ → Cross-Encoder精排  │               │ │
│                            └─────────────────────┘               │ │
└─────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        决策输出 + 持续学习                           │
│  · 直接输出各类别预测标签值                                          │
│  · 低置信度 → LLM 兜底校验                                          │
│  · 反馈池 → 增量微调（LoRA）                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

二、分层设计理由

1. 文档级编码：问答对聚合

为什么必须聚合1000个问答对？
一篇文档的标签由整篇内容决定，但1000个问答对如果全部拼接会超过BERT长度上限（512）。因此我们采用独立编码 + 池化聚合：

· 每个问答对拼接为 [CLS] 问题 [SEP] 答案 [SEP] 原文上下文片段 [SEP]，独立通过BERT得到 [CLS] 向量。
· 将所有1000个问答对的向量用注意力池化（或简单平均）聚合为一个文档向量 h。
    理由：注意力池化可以自动忽略无关问答对，聚焦与分类任务最相关的部分。

2. 处理大小标签空间的两套机制

小类别（≤20个值）直接用softmax分类头
理由：标签值数量小，直接分类简单有效，训练稳定。

大类别（≈1000个值）用“检索→重排序”
理由：

· 1000个输出节点的softmax在小数据下极易过拟合，且新增标签值时需要重新训练整个头。
· 双塔检索：将文档向量 h 与所有标签值的语义描述向量做相似度检索，召回Top-K（如K=10~50）候选。
· 交叉编码器精排：把文档片段+候选标签描述拼在一起，用一个小型Transformer精排，提升准确率。
    这样即使标签值增加到2000个，也只需更新索引，无需重训分类头。

3. LLM驱动的数据飞轮

为什么必须用LLM生成数据？
只有2个文档，无法训练深度学习模型。LLM可以用种子文档的风格、主题、标签模式，批量生成成百上千篇类似文档（每篇含问答对），并附带标签，快速构建训练集。

具体流程：

1. 将2个种子文档的摘要、标签结构、问答对样例构造为提示词。
2. 让LLM批量生成新文档，并自己为文档打标（用Chain-of-Thought + self-consistency提高质量）。
3. 用种子文档的分布做一致性校验，过滤明显偏移的生成样本。
4. 对生成的数据做人工抽检或LLM互审，形成 silver_train 数据集。

4. RAG在架构中的角色

RAG不仅是标签检索，还用于为LLM提供上下文：

· 在LLM兜底校验时，将Top-K候选标签及其描述、原文相关问答对作为上下文，让LLM做出更精准的最终裁决。
· 在数据生成时，RAG可检索类似文档的标签，辅助LLM生成更一致的标签。

5. 持续学习迭代

线上运行后，收集模型低置信度样本和用户修正反馈，定期用回放缓冲区做参数高效微调（如LoRA），避免灾难性遗忘，让模型越用越准。

---

三、关键伪代码

数据增强阶段

```python
def generate_synthetic_docs(seed_docs, schema, llm, n=200):
    synthetic = []
    for i in range(n):
        # 随机选一个种子文档作为风格参考
        seed = random.choice(seed_docs)
        prompt = f"""
        你是一个文档生成器。参考下面文档的主题、结构和标签模式，生成一篇全新的、主题相似但内容不同的文档。
        文档应包含约1000个问答对，标签需覆盖以下类别：{schema.descriptions}
        种子文档摘要：{seed.summary}
        标签示例：{seed.labels}
        请生成新文档和对应的标签值。
        """
        new_doc = llm.generate(prompt)
        # 抽取问答对（如果是结构化输出）
        qa_pairs = extract_qa_pairs(new_doc)
        labels = new_doc.labels
        synthetic.append({'qa_pairs': qa_pairs, 'labels': labels})
    return synthetic
```

文档编码与聚合

```python
class DocumentEncoder(nn.Module):
    def __init__(self, bert_model, hidden_size):
        self.bert = bert_model
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, qa_inputs):  # qa_inputs: (num_pairs, seq_len)
        # 批量编码问答对
        cls_embeddings = self.bert(**qa_inputs).pooler_output  # (num_pairs, hidden)
        # 注意力池化
        attn_scores = self.attention_pool(cls_embeddings).squeeze(-1)  # (num_pairs)
        attn_weights = torch.softmax(attn_scores, dim=0)
        doc_vec = torch.sum(attn_weights.unsqueeze(-1) * cls_embeddings, dim=0)
        return doc_vec
```

多任务模型（含检索分支）

```python
class DocTagger(nn.Module):
    def __init__(self, encoder, small_heads, large_indices, cross_encoder):
        self.encoder = encoder
        self.small_heads = nn.ModuleDict(small_heads)  # {cat: Linear}
        self.large_indices = large_indices  # FAISS indices for each large cat
        self.large_label_texts = ...  # 标签描述列表
        self.cross_encoder = cross_encoder  # 精排模型

    def forward(self, qa_inputs):
        doc_vec = self.encoder(qa_inputs)
        preds = {}
        for cat_name, head in self.small_heads.items():
            logits = head(doc_vec)
            preds[cat_name] = torch.argmax(logits, dim=-1)

        for cat_name in self.large_indices:
            index = self.large_indices[cat_name]
            # 检索Top-K
            scores, ids = index.search(doc_vec.unsqueeze(0).cpu().numpy(), k=10)
            candidates = [self.large_label_texts[cat_name][i] for i in ids[0]]
            # 交叉编码器精排（取最高分）
            best_label = self.cross_encoder.rank(doc_vec, candidates)
            preds[cat_name] = best_label
        return preds
```

训练与评估（留一文档交叉验证）

```python
# 因为只有2个原始标注文档，采用留一法：
for test_doc in seed_docs:
    # 用LLM基于另一个文档生成大量增强数据作为训练集
    train_data = generate_synthetic_docs([other_doc], schema, llm, n=200)
    # 训练模型
    model = train_model(train_data)
    # 在真实测试文档上评估
    metrics = evaluate(model, test_doc)
    print(metrics)

# 同时，在增强数据内部划分90%/10%做常规验证，监控训练过程。
```

持续学习闭环

```python
# 线上推理
def predict_with_fallback(qa_inputs, model, llm, threshold=0.7):
    preds, probs = model(qa_inputs, return_probs=True)
    for cat in probs:
        if max(probs[cat]) < threshold:
            # 用LLM + RAG重新判断该类别
            llm_label = llm_verify(qa_inputs, cat, top_k_candidates)
            preds[cat] = llm_label
    return preds

# 收集反馈
feedback_pool.append((qa_inputs, model_pred, user_correction))

# 定时增量微调
def continual_update(model, feedback_pool, replay_buffer):
    for batch in replay_buffer:
        loss = compute_loss(model, batch)
        # 仅更新LoRA参数
        lora_update(loss)
```

---

四、评估策略

评估维度 方法 理由
类别级准确率 对每个类别分别计算精确率、召回率、F1 不同类别难易度不同，需独立诊断
大类别检索质量 Recall@1, MRR 反映检索+精排的真实效果
文档级整体匹配 全部10个类别完全匹配的 Exact Match Ratio 最严格的端到端要求
泛化能力 留一文档交叉验证（2-fold） 只有2个文档，最充分利用数据
人工评估 定期抽取线上样本人工评分 LLM生成数据的噪声需要人工锚定

---

五、总结：为什么这套架构能工作

1. 冷启动：LLM数据增强解决了“只有2个文档”的致命数据短缺。
2. 大标签空间：检索+排序避开了1000-way softmax的不稳定性，且易于扩展。
3. 文档级决策：问答对聚合池化充分利用了1000个细粒度信号，而非只看摘要。
4. 鲁棒性：LLM兜底机制为小模型补上了安全网，同时为持续学习提供反馈。
5. 成本可控：日常推理只需小模型，LLM只在低置信度时异步调用。

下一步你可以从构造标签描述和用LLM生成第一篇增强文档开始，快速验证整个数据飞轮是否能产出合理质量的训练样本。