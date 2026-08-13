[English](README.md) | [简体中文](README.zh-CN.md)

# Amazon VOC Pipeline

基于云端的亚马逊客户之声（VOC）分析管道，将非结构化的亚马逊评论转化为结构化、面向业务的分析洞察。

项目整合了数据工程、混合 NLP、BigQuery 维度建模以及 Power BI，构建端到端的分析管道。

---

## 架构

![系统架构](docs/architecture.png)

---

## NLP 2.0

NLP 2.0 采用混合分类方法，结合确定性规则与语义相似度。

管道提取以下内容：

- 场景（Scenes）
- 摩擦（Frictions）
- 动机（Motivations）
- 时间（Time of Day）
- 地点（Locations）
- 关键词（Keywords）

语义组件使用 `SentenceTransformer` 和 `all-MiniLM-L6-v2`。

分类结果保留决策信号，如 `Source`、`Score` 和 `Margin`。

---

## 数据模型

### 原始层

`voc_raw.review_raw`

存储由 ETL 管道生成的标准化亚马逊评论数据。

数据通过 `WRITE_APPEND` 增量加载，`Review_Key` 作为原始层的业务键。

### NLP 层

```text
voc_features
├── review_processing
├── review_features
├── scene_bridge
├── friction_bridge
├── motivation_bridge
├── time_bridge
└── location_bridge
```

使用独立的桥接表来处理多值 NLP 维度，以保留关系并避免笛卡尔积问题。

### 分析模型

```text
AsinProduct (1)
      │
      ▼
dim_asinreview (*)
      │
      ▼
dim_review (1)
      │
      ├── scene_bridge (*)
      ├── friction_bridge (*)
      ├── motivation_bridge (*)
      ├── time_bridge (*)
      └── location_bridge (*)
```

分析模型是以 `dim_review` 为中心的星型模型。多值 NLP 标签通过关联到 `Review_ID` 的桥接表解析，从而在 Power BI 中实现干净的筛选、分组和下钻，而不会产生笛卡尔积。

---

## 仓库结构

```text
amazon-voc-pipeline/
├── amazon-voc-etl/
├── amazon-voc-nlp/
├── docs/
├── .gitignore
└── README.md
```

- `amazon-voc-etl` — 数据摄取与标准化
- `amazon-voc-nlp` — NLP 分类与特征生成
- `docs` — 项目文档和架构图

---

## 技术栈

### 数据工程

Python · pandas · openpyxl · PyArrow

### NLP

spaCy · Sentence Transformers · `all-MiniLM-L6-v2` · 正则表达式 · 基于规则的分类

### 云端

Google Cloud Storage · BigQuery · Cloud Run Jobs · Docker

### 商业智能

Microsoft Power BI

---

## 处理策略

NLP 管道具备模型版本感知能力。

当前模型版本：

`nlp_v2.0_rule_v1`

在当前模型版本下已处理的评论会被跳过，无论处理状态如何。这意味着成功或失败处理的评论都视为当前模型版本已完成。

若在修复规则或数据问题后需要重新处理评论，请将 `MODEL_VERSION` 更新为新值。这样既能保留处理历史，又能进行干净的重新处理。

---

## 项目状态

### 已实现

- 亚马逊评论摄取
- 云端 ETL
- BigQuery 原始层和特征层
- 混合 NLP 2.0
- 多标签桥接表
- 产品-评论维度模型
- Power BI 分析模型
- 基于 Git 的版本控制

### 计划中

- NLP 评估框架
- 分类体系优化
- 管道监控
- 进一步性能优化
- Power BI 仪表板扩展

---

## 项目目标

构建一个可复用的 VOC 分析框架，将大量非结构化客户评论转化为结构化、面向业务的洞察。