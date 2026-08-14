[English](README.md) | [简体中文](README.zh-CN.md)

# Amazon VOC Pipeline

基于云端的亚马逊客户之声（VOC）分析管道，将非结构化的亚马逊评论转化为结构化、面向业务的分析洞察。

项目整合了数据工程、混合 NLP、BigQuery 维度建模以及 Power BI 语义层，构建端到端的分析管道。

---

## System Architecture

```mermaid
flowchart TD

A[Amazon Reviews<br/>ASIN-based Review Data<br/>ASIN Sources]

B[Google Cloud Storage<br/>Raw Data Landing Zone]

C[Cloud Run Job<br/>ETL Pipeline<br/>Cleaning & Transformation]

D[BigQuery<br/>Raw Layer<br/>review_raw]

E[Cloud Run Job<br/>NLP Processing<br/>Hybrid NLP Engine]

F[BigQuery<br/>Features Layer<br/>NLP Features<br/>Bridge Tables<br/>Dimension Tables]

G[Power BI<br/>Semantic Model<br/>Star Schema & DAX Measures]

H[VOC Analytics Dashboard<br/>Interactive Insights]


A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H


style A fill:#f9f,stroke:#333,stroke-width:2px
style E fill:#ffe6cc,stroke:#333,stroke-width:2px
style H fill:#dfd,stroke:#333,stroke-width:2px
```

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

分析模型采用带桥接表的维度建模方法，结合星型模型与桥接表。

多值 NLP 属性通过关联到 `Review_ID` 的桥接表解析，从而在 Power BI 中实现干净的筛选、分组和下钻，而不会产生笛卡尔积。

该设计避免了语义模型中模糊的筛选传播，并保持了一对多的分析关系。

---

## Power BI 语义模型

![Power BI 语义模型](docs/powerbi-model.png)

语义层在 Microsoft Power BI 中实现，使用基于 BigQuery 维度表的语义模型。

BigQuery 作为分析数据仓库并提供维度表，而 Power BI 通过关系建模、DAX 度量和交互式分析体验来实现语义层。

语义模型包括：

- 遵循维度建模原则的 Tabular 模型设计
- 使用 DAX 的显式度量层
- 带桥接表的星型模型建模
- 维度表和桥接表关系
- 评论覆盖率、评分分布、摩擦分析和动机分析的计算逻辑
- 跨 NLP 维度的上下文感知筛选
- 评论级下钻分析

关键建模技术包括：

- 用于多值 NLP 维度的桥接表
- 评论级关系保留
- 使用 `TREATAS` 和基于集合的筛选进行虚拟关系传播
- 跨多个分析维度的筛选上下文管理
- 针对场景、摩擦、动机分析的动态度量

跨维度分析的 DAX 示例：

```dax
Motivation Friction Reviews =
CALCULATE(
    [All Reviews],
    TREATAS(
        INTERSECT(
            VALUES(bMotivation[Review_ID]),
            VALUES(bFriction[Review_ID])
        ),
        dReviews[Review_ID]
    )
)
```

该度量值通过虚拟关系计算同时属于所选动机和摩擦类别的评论，并在保持评论级粒度的前提下实现跨维度分析。

### 分析能力

Power BI 语义模型支持：

- 跨场景、摩擦、动机、时间和地点维度的交叉分析
- 评论覆盖率分析，用于衡量 NLP 分类覆盖情况
- 评分分布和变异性分析
- 通过 NLP 驱动维度发现客户摩擦点
- 从聚合洞察下钻到单条客户反馈的评论级下钻

PBIP 以基于文本的文件存储报表和语义模型定义，支持通过 Git 进行源代码控制和变更追踪。

---

## 仓库结构

```text
amazon-voc-pipeline/
├── amazon-voc-etl/
│   └── 数据摄取与标准化管道
│
├── amazon-voc-nlp/
│   └── 混合 NLP 分类管道
│
├── sql/
│   └── BigQuery 特征层架构与数据建模脚本
│
├── PowerBI/
│   └── 包含语义模型和报表定义的 Power BI PBIP 项目
│
├── docs/
│   └── 架构图和语义模型图
│
├── .gitignore
├── README.md
└── README.zh-CN.md
```

- `amazon-voc-etl` — 数据摄取与标准化
- `amazon-voc-nlp` — NLP 分类与特征生成
- `sql` — BigQuery 特征层架构定义与分析数据建模脚本
- `PowerBI` — 包含语义模型和报表定义的 Power BI PBIP 项目
- `docs` — 项目文档、架构图和语义模型图

---

## 技术栈

### 数据工程

Python · pandas · openpyxl · PyArrow

### NLP

spaCy · Sentence Transformers · `all-MiniLM-L6-v2` · 正则表达式 · 基于规则的分类

### 云端

Google Cloud Storage · BigQuery · Cloud Run Jobs · Docker

### 商业智能

Microsoft Power BI · Tabular 语义模型 · DAX · Power Query · TMDL · PBIP 版本控制

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
- 基于 DAX 分析计算层的 Power BI 语义模型
- 基于 PBIP 的报表和语义模型版本控制
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