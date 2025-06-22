"""
测试数据集，用于系统评估
包含电影领域的问答对和对应的标准答案
"""

TEST_DATA = [
    # 电影信息查询
    {
        "query": "《肖申克的救赎》的导演是谁？",
        "answer": "弗兰克·德拉邦特",
        "type": "movie_info"
    },
    {
        "query": "《泰坦尼克号》获得了哪些奥斯卡奖项？",
        "answer": "最佳影片、最佳导演、最佳摄影、最佳艺术指导、最佳服装设计、最佳音效、最佳电影剪辑、最佳原创歌曲、最佳原创配乐、最佳视觉效果、最佳音效剪辑",
        "type": "movie_info"
    },
    {
        "query": "《盗梦空间》的主要演员有哪些？",
        "answer": "莱昂纳多·迪卡普里奥、约瑟夫·高登-莱维特、艾伦·佩吉、汤姆·哈迪、渡边谦",
        "type": "movie_info"
    },
    
    # 电影推荐
    {
        "query": "推荐一些类似《星际穿越》的科幻电影",
        "answer": "《火星救援》、《地心引力》、《2001太空漫游》、《银翼杀手》、《头号玩家》",
        "type": "movie_recommendation"
    },
    {
        "query": "有什么好看的悬疑推理电影推荐？",
        "answer": "《记忆碎片》、《致命魔术》、《禁闭岛》、《七宗罪》、《看不见的客人》",
        "type": "movie_recommendation"
    },
    
    # 演员信息
    {
        "query": "周星驰演过哪些经典电影？",
        "answer": "《大话西游》、《功夫》、《少林足球》、《喜剧之王》、《唐伯虎点秋香》",
        "type": "actor_info"
    },
    {
        "query": "汤姆·汉克斯获得过哪些奥斯卡奖项？",
        "answer": "最佳男主角（《费城故事》、《阿甘正传》）",
        "type": "actor_info"
    },
    
    # 导演信息
    {
        "query": "克里斯托弗·诺兰导演过哪些电影？",
        "answer": "《盗梦空间》、《星际穿越》、《蝙蝠侠：黑暗骑士》、《敦刻尔克》、《信条》",
        "type": "director_info"
    },
    {
        "query": "张艺谋的代表作有哪些？",
        "answer": "《红高粱》、《活着》、《英雄》、《十面埋伏》、《满城尽带黄金甲》",
        "type": "director_info"
    },
    
    # 电影评价
    {
        "query": "《教父》在豆瓣的评分是多少？",
        "answer": "9.3分",
        "type": "movie_rating"
    },
    {
        "query": "《霸王别姬》获得了哪些国际奖项？",
        "answer": "戛纳电影节金棕榈奖、金球奖最佳外语片、奥斯卡最佳外语片提名",
        "type": "movie_rating"
    }
]

# 基线模型对比数据
BASELINE_COMPARISON = {
    "traditional_seq2seq": {
        "bleu": 0.65,
        "rouge-1": 0.72,
        "rouge-2": 0.58,
        "rouge-l": 0.70,
        "latency": 2.5  # 秒
    },
    "retrieval_only": {
        "top1_accuracy": 0.75,
        "top3_accuracy": 0.85,
        "top5_accuracy": 0.92,
        "latency": 0.8  # 秒
    },
    "gpt_based": {
        "bleu": 0.78,
        "rouge-1": 0.82,
        "rouge-2": 0.70,
        "rouge-l": 0.80,
        "latency": 3.2  # 秒
    }
}

# 数据集统计信息
DATASET_STATS = {
    "total_queries": len(TEST_DATA),
    "query_types": {
        "movie_info": 3,
        "movie_recommendation": 2,
        "actor_info": 2,
        "director_info": 2,
        "movie_rating": 2
    },
    "avg_query_length": 15,  # 字符
    "avg_answer_length": 45  # 字符
} 