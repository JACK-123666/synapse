"""Synapse 意图识别评测测试集。

100 条人工标注的中英文查询，覆盖三类意图：
    knowledge_retrieval  知识检索 / 概念解释 / 文档查询
    summarize            摘要 / 总结 / 概括
    small_talk           问候 / 闲聊 / 情感表达

字段说明：
    text   用户输入
    intent 金标准意图（人工标注）
    hard   是否为「模糊难例」——关键词路难以命中、需要语义理解才能正确分类的样本。
           用于单独评估「三路融合在难例上的表现」，这是融合设计的价值主张。

使用方式：被 tools/eval_intent.py 导入。欢迎补充/修正样本（注意保持标注一致性）。
"""

TEST_SET = [
    # ---------- knowledge_retrieval · 清晰样本 ----------
    {"text": "什么是向量数据库？", "intent": "knowledge_retrieval", "hard": False},
    {"text": "解释一下RAG是什么", "intent": "knowledge_retrieval", "hard": False},
    {"text": "Redis和MySQL有什么区别？", "intent": "knowledge_retrieval", "hard": False},
    {"text": "帮我查一下多Agent架构的原理", "intent": "knowledge_retrieval", "hard": False},
    {"text": "什么是LoRA微调？", "intent": "knowledge_retrieval", "hard": False},
    {"text": "Docker和Kubernetes的区别", "intent": "knowledge_retrieval", "hard": False},
    {"text": "如何部署FastAPI服务？", "intent": "knowledge_retrieval", "hard": False},
    {"text": "帮我找关于机器学习的资料", "intent": "knowledge_retrieval", "hard": False},
    {"text": "检索一下相关的技术文档", "intent": "knowledge_retrieval", "hard": False},
    {"text": "什么是Transformer架构？", "intent": "knowledge_retrieval", "hard": False},
    {"text": "了解prompt工程的基本概念", "intent": "knowledge_retrieval", "hard": False},
    {"text": "what is a vector database?", "intent": "knowledge_retrieval", "hard": False},
    {"text": "How does authentication work in this system?", "intent": "knowledge_retrieval", "hard": False},
    {"text": "what is the difference between Redis and Memcached", "intent": "knowledge_retrieval", "hard": False},
    {"text": "explain chain of thought prompting", "intent": "knowledge_retrieval", "hard": False},
    {"text": "这个技术方案是什么意思？", "intent": "knowledge_retrieval", "hard": False},
    {"text": "查一下最新的AI新闻", "intent": "knowledge_retrieval", "hard": False},
    {"text": "我想了解嵌入式系统", "intent": "knowledge_retrieval", "hard": False},
    {"text": "讲解一下NLP的发展历程", "intent": "knowledge_retrieval", "hard": False},
    {"text": "什么是大语言模型？", "intent": "knowledge_retrieval", "hard": False},
    # ---------- knowledge_retrieval · 模糊难例（关键词路难以命中） ----------
    {"text": "那个东西的原理是什么？", "intent": "knowledge_retrieval", "hard": True},
    {"text": "这个功能怎么用？", "intent": "knowledge_retrieval", "hard": True},
    {"text": "我上次看的那篇文章在讲什么来着？", "intent": "knowledge_retrieval", "hard": True},
    {"text": "有没有关于这个话题的资料？", "intent": "knowledge_retrieval", "hard": True},
    {"text": "能再讲详细一点吗？", "intent": "knowledge_retrieval", "hard": True},
    {"text": "解释下为什么需要Rerank", "intent": "knowledge_retrieval", "hard": True},
    {"text": "how does that work?", "intent": "knowledge_retrieval", "hard": True},
    {"text": "can you explain that concept to me?", "intent": "knowledge_retrieval", "hard": True},
    {"text": "请介绍一下这个项目", "intent": "knowledge_retrieval", "hard": True},
    {"text": "我想知道这个系统的架构是怎么设计的", "intent": "knowledge_retrieval", "hard": True},
    # ---------- summarize · 清晰样本 ----------
    {"text": "帮我总结一下这段对话", "intent": "summarize", "hard": False},
    {"text": "把这篇文档压缩成摘要", "intent": "summarize", "hard": False},
    {"text": "概括一下会议要点", "intent": "summarize", "hard": False},
    {"text": "总结一下这篇文章的主要内容", "intent": "summarize", "hard": False},
    {"text": "提炼一下重点", "intent": "summarize", "hard": False},
    {"text": "简要概括", "intent": "summarize", "hard": False},
    {"text": "归纳一下这几个观点", "intent": "summarize", "hard": False},
    {"text": "帮我写个摘要", "intent": "summarize", "hard": False},
    {"text": "这段文字讲了什么，简短说", "intent": "summarize", "hard": False},
    {"text": "summarize the key points of this article", "intent": "summarize", "hard": False},
    {"text": "give me a brief summary of the document", "intent": "summarize", "hard": False},
    {"text": "can you condense this text?", "intent": "summarize", "hard": False},
    {"text": "把聊天记录整理成要点", "intent": "summarize", "hard": False},
    {"text": "帮我总结PDF内容", "intent": "summarize", "hard": False},
    {"text": "总结一下市场报告", "intent": "summarize", "hard": False},
    # ---------- summarize · 模糊难例 ----------
    {"text": "太长了我没时间看，帮我概括一下", "intent": "summarize", "hard": True},
    {"text": "说了这么多，中心思想是什么？", "intent": "summarize", "hard": True},
    {"text": "用三句话概括", "intent": "summarize", "hard": True},
    {"text": "帮我提炼核心观点", "intent": "summarize", "hard": True},
    {"text": "这段内容信息量太大，压缩一下", "intent": "summarize", "hard": True},
    # ---------- small_talk · 清晰样本 ----------
    {"text": "你好", "intent": "small_talk", "hard": False},
    {"text": "嗨", "intent": "small_talk", "hard": False},
    {"text": "谢谢", "intent": "small_talk", "hard": False},
    {"text": "再见", "intent": "small_talk", "hard": False},
    {"text": "早上好", "intent": "small_talk", "hard": False},
    {"text": "今天天气怎么样", "intent": "small_talk", "hard": False},
    {"text": "你是谁", "intent": "small_talk", "hard": False},
    {"text": "哈哈", "intent": "small_talk", "hard": False},
    {"text": "好的", "intent": "small_talk", "hard": False},
    {"text": "辛苦了", "intent": "small_talk", "hard": False},
    {"text": "hello", "intent": "small_talk", "hard": False},
    {"text": "hi there", "intent": "small_talk", "hard": False},
    {"text": "thanks a lot", "intent": "small_talk", "hard": False},
    {"text": "how are you", "intent": "small_talk", "hard": False},
    {"text": "拜拜", "intent": "small_talk", "hard": False},
    {"text": "晚上好", "intent": "small_talk", "hard": False},
    {"text": "你叫什么名字", "intent": "small_talk", "hard": False},
    {"text": "在吗", "intent": "small_talk", "hard": False},
    # ---------- small_talk · 模糊难例 ----------
    {"text": "然后呢？", "intent": "small_talk", "hard": True},
    {"text": "继续", "intent": "small_talk", "hard": True},
    {"text": "说说看", "intent": "small_talk", "hard": True},
    {"text": "哦是吗", "intent": "small_talk", "hard": True},
    {"text": "就这样吧", "intent": "small_talk", "hard": True},
    {"text": "你在干嘛", "intent": "small_talk", "hard": True},
    {"text": "真不错", "intent": "small_talk", "hard": True},
    {"text": "有意思", "intent": "small_talk", "hard": True},
    # ---------- 补充：knowledge_retrieval ----------
    {"text": "什么是检索增强生成", "intent": "knowledge_retrieval", "hard": False},
    {"text": "帮我查一下prompt里temperature参数", "intent": "knowledge_retrieval", "hard": False},
    {"text": "ChromaDB和Milvus怎么选", "intent": "knowledge_retrieval", "hard": False},
    {"text": "解释一下attention机制", "intent": "knowledge_retrieval", "hard": False},
    {"text": "了解了解微调的原理", "intent": "knowledge_retrieval", "hard": False},
    {"text": "介绍一下Redis的持久化方式", "intent": "knowledge_retrieval", "hard": False},
    {"text": "SQL和NoSQL的区别", "intent": "knowledge_retrieval", "hard": False},
    {"text": "什么是余弦相似度", "intent": "knowledge_retrieval", "hard": False},
    {"text": "帮我找找相关的论文", "intent": "knowledge_retrieval", "hard": False},
    {"text": "什么是多模态模型", "intent": "knowledge_retrieval", "hard": False},
    {"text": "解释一下Z-score异常检测", "intent": "knowledge_retrieval", "hard": False},
    # ---------- 补充：summarize ----------
    {"text": "帮我概括这段代码的作用", "intent": "summarize", "hard": False},
    {"text": "简述一下会议纪要", "intent": "summarize", "hard": False},
    {"text": "把这份报告浓缩成一段话", "intent": "summarize", "hard": False},
    {"text": "归纳总结我的需求", "intent": "summarize", "hard": False},
    {"text": "简短总结一下背景", "intent": "summarize", "hard": False},
    {"text": "帮我整理一下读书笔记要点", "intent": "summarize", "hard": False},
    {"text": "总结一下这周的进度", "intent": "summarize", "hard": False},
    # ---------- 补充：small_talk ----------
    {"text": "哈喽", "intent": "small_talk", "hard": False},
    {"text": "谢谢你的帮助", "intent": "small_talk", "hard": False},
    {"text": "太棒了", "intent": "small_talk", "hard": False},
    {"text": "嗯嗯", "intent": "small_talk", "hard": False},
    {"text": "晚安", "intent": "small_talk", "hard": False},
    {"text": "哈哈好的", "intent": "small_talk", "hard": False},
]


def distribution() -> dict:
    """返回测试集的意图分布统计。"""
    dist: dict = {}
    hard: dict = {}
    for item in TEST_SET:
        i = item["intent"]
        dist[i] = dist.get(i, 0) + 1
        if item["hard"]:
            hard[i] = hard.get(i, 0) + 1
    return {"total": len(TEST_SET), "dist": dist, "hard": hard, "hard_total": sum(hard.values())}
