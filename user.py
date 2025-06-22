# -*- coding: utf-8 -*-
import os
import streamlit as st
import ner_model as zwk
import pickle
from transformers import BertTokenizer
import torch
import py2neo
import random
import re
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import unicodedata

# 缓存装饰器
def cache_result(expire_seconds: int = 300):
    cache = {}
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            now = datetime.now()
            
            # 检查缓存是否存在且未过期
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if now - timestamp < timedelta(seconds=expire_seconds):
                    return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[cache_key] = (result, now)
            return result
        return wrapper
    return decorator

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background():
    bin_str = get_base64_of_bin_file('D:/RAGQnASystem/img/user_background.jpg')
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.45)), url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }}

    .stApp > header {{
        background-color: transparent !important;
    }}

    .main .block-container {{
        padding: 1rem;
        max-width: 100%;
        height: calc(100vh - 80px);
        display: flex;
        flex-direction: column;
    }}

    /* 标签页容器样式 */
    .stTabs {{
        flex: 1;
        display: flex;
        flex-direction: column;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* 标签页内容区域样式 */
    .stTabs [data-baseweb="tab-panel"] {{
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
    }}

    /* 聊天界面样式 */
    [data-testid="stChatMessageContainer"] {{
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 80px;
        display: flex;
        flex-direction: column;
    }}

    .stChatInputContainer {{
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100% - 350px);
        padding: 0 !important;
        z-index: 1000;
        background: transparent !important;
    }}

    /* 聊天输入框样式 */
    .stChatInput {{
        background: rgba(32, 33, 35, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        margin: 0 !important;
        transition: all 0.3s ease;
    }}

    /* 输入框内部容器 */
    .stChatInput > div {{
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* 实际输入区域 */
    .stChatInput input {{
        background: transparent !important;
        color: rgba(255, 255, 255, 0.9) !important;
        padding: 12px 16px !important;
        height: 48px !important;
        font-size: 16px !important;
        line-height: 24px !important;
        border: none !important;
        box-shadow: none !important;
        width: 100% !important;
        margin: 0 !important;
    }}

    /* 输入框占位符 */
    .stChatInput input::placeholder {{
        color: rgba(255, 255, 255, 0.4) !important;
        font-size: 16px !important;
    }}

    /* 输入框悬停状态 */
    .stChatInput:hover {{
        background: rgba(32, 33, 35, 0.98) !important;
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }}

    /* 输入框焦点状态 */
    .stChatInput:focus-within {{
        background: rgba(32, 33, 35, 1) !important;
        border-color: rgba(64, 149, 255, 0.6) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4),
                   0 0 0 2px rgba(64, 149, 255, 0.2) !important;
    }}

    /* 发送按钮容器 */
    .stChatInput button {{
        margin-right: 8px !important;
        opacity: 0.8;
        transition: opacity 0.3s ease;
    }}

    /* 发送按钮悬停状态 */
    .stChatInput button:hover {{
        opacity: 1;
    }}

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px);
        height: 100vh;
        overflow-y: auto;
    }}

    /* 电影推荐界面样式 */
    .filter-section {{
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .movie-card {{
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .movie-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}

    .movie-type-tag {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .movie-synopsis {{
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* 按钮和控件样式 */
    .stButton button {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
    }}

    .stButton button:hover {{
        background-color: rgba(255, 255, 255, 0.1) !important;
    }}

    /* 滑块和选择器样式 */
    .stSlider, .stSelectbox {{
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }}

    /* 扩展器样式 */
    .streamlit-expanderHeader {{
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }}

    /* 滚动条样式 */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255, 255, 255, 0.15);
    }}

    /* 确保所有组件可见 */
    .stButton button, 
    .stSelectbox select,
    .stSlider,
    .stTextInput input {{
        opacity: 1 !important;
        visibility: visible !important;
    }}

    /* 侧边栏按钮统一样式 */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        min-height: 45px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem 0 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }}

    [data-testid="stSidebar"] .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* 删除按钮特殊样式 */
    [data-testid="stSidebar"] .stButton > button[aria-label*="delete"] {{
        min-width: 40px !important;
        padding: 0.5rem !important;
    }}

    /* 活动窗口按钮样式 */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: linear-gradient(45deg, rgba(64, 149, 255, 0.15), rgba(94, 114, 235, 0.15)) !important;
        border-color: rgba(64, 149, 255, 0.3) !important;
    }}

    /* 聊天消息气泡样式优化 */
    [data-testid="stChatMessage"] {{
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        animation: fadeInUp 0.5s ease-out !important;
    }}

    /* 用户消息特殊样式 */
    [data-testid="stChatMessage"][data-testid="user"] {{
        background: linear-gradient(135deg, rgba(64, 149, 255, 0.1), rgba(94, 114, 235, 0.1)) !important;
        border-color: rgba(64, 149, 255, 0.2) !important;
        margin-left: 2rem !important;
    }}

    /* 助手消息特殊样式 */
    [data-testid="stChatMessage"][data-testid="assistant"] {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)) !important;
        margin-right: 2rem !important;
    }}

    /* 消息内容样式 */
    [data-testid="stChatMessage"] p {{
        margin: 0 !important;
        line-height: 1.6 !important;
    }}

    /* 标签页样式优化 */
    .stTabs {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px) !important;
    }}

    /* 标签按钮样式 */
    .stTabs [role="tab"] {{
        background: transparent !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        margin: 0 0.25rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }}

    /* 标签按钮悬停效果 */
    .stTabs [role="tab"]:hover {{
        background: rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.9) !important;
    }}

    /* 选中的标签按钮 */
    .stTabs [role="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(64, 149, 255, 0.2), rgba(94, 114, 235, 0.2)) !important;
        color: white !important;
        font-weight: 500 !important;
    }}

    /* 标签内容区域 */
    .stTabs [role="tabpanel"] {{
        padding: 1rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin-top: 0.5rem !important;
    }}

    /* 加载动画 */
    @keyframes pulse {{
        0% {{ opacity: 0.6; transform: scale(0.98); }}
        50% {{ opacity: 0.8; transform: scale(1); }}
        100% {{ opacity: 0.6; transform: scale(0.98); }}
    }}

    /* 消息加载动画 */
    .stMarkdown div[data-testid="stMarkdownContainer"] {{
        animation: fadeInUp 0.5s ease-out;
    }}

    /* 页面切换动画 */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* 扩展器样式优化 */
    .streamlit-expanderHeader {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }}

    .streamlit-expanderHeader:hover {{
        background: rgba(255, 255, 255, 0.05) !important;
    }}

    /* 代码块样式优化 */
    pre {{
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        overflow-x: auto !important;
    }}

    code {{
        color: #a6e22e !important;
        font-family: 'Fira Code', monospace !important;
    }}

    /* 滚动条美化 */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        transition: all 0.3s ease;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255, 255, 255, 0.2);
    }}

    /* 链接样式优化 */
    a {{
        color: rgb(64, 149, 255) !important;
        text-decoration: none !important;
        transition: all 0.3s ease !important;
    }}

    a:hover {{
        color: rgb(94, 114, 235) !important;
        text-decoration: underline !important;
    }}

    /* 工具提示样式 */
    [data-tooltip]:hover:before {{
        content: attr(data-tooltip);
        position: absolute;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        white-space: nowrap;
        z-index: 1000;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        pointer-events: none;
        opacity: 0;
        animation: fadeIn 0.3s ease-out forwards;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# 添加Qwen API配置
QWEN_API_KEY = "sk-0064b09d78a04f50a4d3fa4b944eaaa2"  # 阿里云百炼API密钥
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云百炼API基础URL
# 添加DeepSeek API配置
DEEPSEEK_API_KEY = "sk-a5aefc754e8549a3914e25b62816647f"  # 阿里云百炼API密钥
DEEPSEEK_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云百炼API基础URL

@cache_result(expire_seconds=600)
def get_movie_recommendations(user_preferences: Dict[str, Any], client) -> List[Dict[str, Any]]:
    """
    增强版基于用户偏好的电影推荐算法
    """
    try:
        recommendations = []
        base_rating = float(user_preferences.get('rating', 0.0))
        
        # 1. 基于类型的推荐
        if 'genre' in user_preferences:
            genre = data_validator.clean_text(user_preferences['genre'])
            query = """
                MATCH (m:电影)-[:属于]->(g:类型)
                WHERE g.名称 CONTAINS $genre AND toFloat(m.豆瓣评分) >= $rating
                WITH m, g
                OPTIONAL MATCH (m)-[:属于]->(og:类型)
                WITH m, collect(og.名称) as genres
                OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                WITH m, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                WITH m, genres, directors, collect(a.名称)[..3] as actors
                RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                       m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                       m.语言 as 语言,
                       genres as 电影类型, directors as 导演, actors as 主演,
                       m.评价人数 as 评价人数, m.又名 as 又名
                ORDER BY toFloat(m.豆瓣评分) DESC, toInteger(m.评价人数) DESC
            """
            result = client.run(query, genre=genre, rating=base_rating)
            recommendations.extend([dict(record) for record in result])
            
            # 如果没有找到结果，尝试使用更宽松的匹配
            if not recommendations:
                genre_alt = {
                    '喜剧': ['搞笑', '幽默', '喜剧'],
                    '动作': ['动作', '武打', '功夫'],
                    '科幻': ['科幻', '未来', '科技'],
                    '恐怖': ['恐怖', '惊悚', '悬疑'],
                    '爱情': ['爱情', '浪漫', '情感']
                }.get(genre, [genre])
                for alt in genre_alt:
                    query = """
                        MATCH (m:电影)-[:属于]->(g:类型)
                        WHERE (g.名称 CONTAINS $genre OR g.名称 CONTAINS $genre_alt)
                        AND toFloat(m.豆瓣评分) >= $rating
                        WITH m, g
                        OPTIONAL MATCH (m)-[:属于]->(og:类型)
                        WITH m, collect(og.名称) as genres
                        OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                        WITH m, genres, collect(d.名称) as directors
                        OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                        WITH m, genres, directors, collect(a.名称)[..3] as actors
                        RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                               m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                               m.语言 as 语言,
                               genres as 电影类型, directors as 导演, actors as 主演,
                               m.评价人数 as 评价人数, m.又名 as 又名
                        ORDER BY toFloat(m.豆瓣评分) DESC, toInteger(m.评价人数) DESC
                    """
                    result = client.run(query, genre=genre, genre_alt=alt, rating=base_rating)
                    recommendations.extend([dict(record) for record in result])
                    if recommendations:
                        break
        
        # 2. 基于导演的推荐
        elif 'director' in user_preferences:
            director = data_validator.clean_text(user_preferences['director'])
            query = """
                MATCH (d:导演 {名称: $director})-[:执导]->(m:电影)
                WHERE toFloat(m.豆瓣评分) >= $rating
                WITH d, m
                OPTIONAL MATCH (m)-[:属于]->(g:类型)
                WITH m, collect(g.名称) as genres
                OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                WITH m, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                WITH m, genres, directors, collect(a.名称)[..3] as actors
                RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                       m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                       m.语言 as 语言,
                       genres as 电影类型, directors as 导演, actors as 主演,
                       m.评价人数 as 评价人数, m.又名 as 又名
                ORDER BY toFloat(m.豆瓣评分) DESC, toInteger(m.评价人数) DESC
            """
            result = client.run(query, director=director, rating=base_rating)
            recommendations.extend([dict(record) for record in result])
        
        # 3. 基于相似电影的推荐
        elif 'similar_to' in user_preferences:
            movie = data_validator.clean_text(user_preferences['similar_to'])
            query = """
                MATCH (m1:电影 {名称: $movie})
                MATCH (m1)-[:属于]->(g:类型)
                MATCH (m2:电影)-[:属于]->(g)
                WHERE m2.名称 <> m1.名称 AND toFloat(m2.豆瓣评分) >= $rating
                WITH m1, m2, count(g) as common_genres,
                     collect(g.名称) as genres
                OPTIONAL MATCH (m2)<-[:执导]-(d:导演)
                WITH m1, m2, common_genres, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m2)<-[:出演]-(a:演员)
                WITH m2, common_genres, genres, directors,
                     collect(a.名称)[..3] as actors
                RETURN m2.名称 as 电影名称, m2.豆瓣评分 as 评分,
                       m2.上映日期 as 年份, m2.剧情简介 as 简介,
                       m2.片长 as 时长, m2.制片国家 as 国家地区,
                       m2.语言 as 语言,
                       genres as 电影类型, directors as 导演,
                       actors as 主演, m2.评价人数 as 评价人数,
                       common_genres as 相似度, m2.又名 as 又名
                ORDER BY common_genres DESC, toFloat(m2.豆瓣评分) DESC
            """
            result = client.run(query, movie=movie, rating=base_rating)
            recommendations.extend([dict(record) for record in result])
        
        # 4. 基于用户综合偏好的智能推荐
        else:
            base_query = """
                MATCH (m:电影)
                WHERE toFloat(m.豆瓣评分) >= $rating
            """
            
            conditions = []
            params = {'rating': base_rating}
            
            # 添加年份范围条件
            if 'year' in user_preferences:
                min_year, max_year = user_preferences['year']
                conditions.append("""
                    CASE 
                        WHEN m.上映日期 IS NOT NULL THEN 
                            CASE 
                                WHEN m.上映日期 CONTAINS '-' THEN 
                                    toInteger(SUBSTRING(m.上映日期, 0, 4)) >= $min_year AND 
                                    toInteger(SUBSTRING(m.上映日期, 0, 4)) <= $max_year
                                ELSE 
                                    toInteger(m.上映日期) >= $min_year AND 
                                    toInteger(m.上映日期) <= $max_year
                            END
                        ELSE true
                    END
                """)
                params.update({'min_year': min_year, 'max_year': max_year})
            
            # 添加时长范围条件
            if 'duration' in user_preferences:
                min_duration, max_duration = user_preferences['duration']
                conditions.append("""
                    CASE 
                        WHEN m.片长 IS NOT NULL THEN 
                            CASE 
                                WHEN m.片长 CONTAINS '分钟' THEN 
                                    toInteger(replace(m.片长, '分钟', '')) >= $min_duration AND 
                                    toInteger(replace(m.片长, '分钟', '')) <= $max_duration
                                ELSE 
                                    toInteger(m.片长) >= $min_duration AND 
                                    toInteger(m.片长) <= $max_duration
                            END
                        ELSE true
                    END
                """)
                params.update({'min_duration': min_duration, 'max_duration': max_duration})
            
            # 添加国家地区条件
            if 'country' in user_preferences:
                country = data_validator.clean_text(user_preferences['country'])
                # 特殊处理中国的情况
                if country == '中国':
                    conditions.append("""
                        CASE 
                            WHEN m.制片国家 IS NOT NULL THEN 
                                ANY(country IN SPLIT(m.制片国家, '/') WHERE 
                                    country CONTAINS '中国' OR 
                                    country CONTAINS '香港' OR 
                                    country CONTAINS '台湾' OR 
                                    country CONTAINS '澳门')
                            ELSE true
                        END
                    """)
                else:
                    conditions.append("""
                        CASE 
                            WHEN m.制片国家 IS NOT NULL THEN 
                                ANY(country IN SPLIT(m.制片国家, '/') WHERE country CONTAINS $country)
                            ELSE true
                        END
                    """)
                params.update({'country': country})
            
            # 组合所有条件
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            # 添加排序和聚合逻辑
            query = base_query + """
                WITH m
                OPTIONAL MATCH (m)-[:属于]->(g:类型)
                WITH m, collect(g.名称) as genres
                OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                WITH m, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                WITH m, genres, directors, collect(a.名称)[..3] as actors
                RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分,
                       m.上映日期 as 年份, m.剧情简介 as 简介,
                       m.片长 as 时长, m.制片国家 as 国家地区,
                       m.语言 as 语言,
                       genres as 电影类型, directors as 导演,
                       actors as 主演, m.评价人数 as 评价人数, m.又名 as 又名
                ORDER BY toFloat(m.豆瓣评分) DESC, toInteger(m.评价人数) DESC
            """
            result = client.run(query, **params)
            recommendations.extend([dict(record) for record in result])
        
        # 数据清洗和验证
        clean_recommendations = []
        for rec in recommendations:
            if not rec.get('电影名称'):  # 跳过没有电影名称的记录
                continue
                
            clean_rec = {}
            for key, value in rec.items():
                if key == '评分':
                    try:
                        clean_value = float(value) if value else 0.0
                    except (ValueError, TypeError):
                        clean_value = 0.0
                elif key in ['年份', '上映日期']:
                    # 保留原始字符串
                    clean_value = value if value else "未知"
                elif key == '时长':
                    clean_value = data_validator.validate_duration(value)
                    if clean_value is None:
                        clean_value = 0
                elif key == '评价人数':
                    try:
                        clean_value = int(value) if value else 0
                    except (ValueError, TypeError):
                        clean_value = 0
                elif key == '电影类型':
                    if isinstance(value, list):
                        clean_value = [str(v).strip() for v in value if v]
                    else:
                        clean_value = [str(value).strip()] if value else []
                elif key == '语言':
                    if isinstance(value, str):
                        clean_value = [lang.strip() for lang in value.split('/') if lang.strip()]
                    elif isinstance(value, list):
                        clean_value = [str(lang).strip() for lang in value if lang]
                    else:
                        clean_value = []
                elif key == '又名':
                    clean_value = value if value else ""
                else:
                    clean_value = data_validator.clean_text(str(value)) if value else ""
                
                clean_rec[key] = clean_value
            
            if clean_rec.get('电影名称') and clean_rec.get('评分', 0) > 0:
                clean_recommendations.append(clean_rec)
        
        # 去重并按评分和评价人数排序
        seen = set()
        unique_recommendations = []
        for rec in clean_recommendations:
            if rec['电影名称'] not in seen:
                seen.add(rec['电影名称'])
                unique_recommendations.append(rec)
        
        # 最终排序：优先考虑评分，其次是评价人数
        unique_recommendations.sort(
            key=lambda x: (float(x.get('评分', 0)), int(x.get('评价人数', 0))), 
            reverse=True
        )
        # 只返回全部，不截断
        return unique_recommendations
    except Exception as e:
        st.error(f"推荐系统错误: {str(e)}")
        return []

@st.cache_resource
def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #加载ChatGLM模型
    # glm_tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b-128k", trust_remote_code=True)
    # glm_model = AutoModel.from_pretrained("model/chatglm3-6b-128k",trust_remote_code=True,device=device)
    # glm_model.eval()
    glm_model = None
    glm_tokenizer= None
    #加载Bert模型
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()
    model_name = 'model/chinese-roberta-wwm-ext'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = zwk.Bert_Model(model_name, hidden_size=64, tag_num=len(tag2idx), bi=True)
    bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt', map_location='cpu'))
    
    bert_model = bert_model.to(device)
    bert_model.eval()
    return glm_tokenizer,glm_model,bert_tokenizer,bert_model,idx2tag,rule,tfidf_r,device

def get_model_client(choice):
    choice = choice.lower()  # 统一转换为小写
    if choice == "qwen-plus":
        return OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_BASE,
        )
    elif choice == "deepseek-v3":
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
        )
    else:
        raise ValueError(f"Unsupported model: {choice}")

def Intent_Recognition(query, choice):
    prompt = f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。

**查询类别**
- "查询电影基本信息"  # 包含电影名称、评分、年份等基本信息
- "查询导演信息"  # 导演相关查询
- "查询演员信息"  # 演员相关查询
- "查询电影类型"  # 电影类型/题材查询
- "查询电影评分"  # 豆瓣评分相关查询
- "查询上映年份"  # 电影上映时间查询
- "查询评价人数"  # 评价人数相关查询
- "查询国家地区"  # 制片国家/地区查询
- "查询语言信息"  # 电影语言查询
- "查询剧情简介"  # 电影剧情介绍查询
- "查询电影时长"  # 电影片长查询
- "电影推荐"  # 基于用户偏好的电影推荐
- "类型推荐"  # 基于类型的电影推荐
- "相似电影推荐"  # 基于相似度的电影推荐

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中
- 确保最终的输出列表包含了所有与用户问题相关的类别描述

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！

现在，请你解决下面这个问题并将结果输出！
问题输入："{query}"
"""
    client = get_model_client(choice)
    response = client.chat.completions.create(
        model=choice.lower(),  # 直接使用选择的模型
        messages=[
            {"role": "system", "content": prompt}
        ],
    )
    rec_result = response.choices[0].message.content
    print(f'意图识别结果:{rec_result}')
    return rec_result

def add_shuxing_prompt(entity: str, shuxing: str, client) -> str:
    """
    添加属性查询提示
    """
    add_prompt = ""
    try:
        entity = data_validator.clean_text(entity)
        if shuxing == '基本信息':
            # 查询多个基本属性
            sql_q = f"""
            MATCH (a:电影) 
            WHERE a.名称 CONTAINS '{entity}' 
            RETURN a.名称, a.导演, a.主演, a.类型, a.上映日期, 
                   a.制片国家, a.语言, a.片长, a.豆瓣评分, a.评价人数
            """
            res = client.run(sql_q).data()
            if len(res) > 0:
                movie = res[0]
                info = []
                for key, value in movie.items():
                    if value and str(value).strip():
                        # 数据验证和清洗
                        clean_value = value
                        if '评分' in key:
                            clean_value = data_validator.validate_rating(value)
                        elif '年份' in key or '日期' in key:
                            clean_value = data_validator.validate_year(value)
                        elif '片长' in key:
                            clean_value = data_validator.validate_duration(value)
                        else:
                            clean_value = data_validator.clean_text(str(value))
                        
                        if clean_value is not None:
                            info.append(f"{key.split('.')[-1]}: {clean_value}")
                add_prompt += "、".join(info)
            else:
                add_prompt += "图谱中无信息，查找失败。"
        else:
            # 普通属性查询
            sql_q = f"MATCH (a:电影) WHERE a.名称 CONTAINS '{entity}' RETURN a.{shuxing}"
            res = client.run(sql_q).data()
            if len(res) > 0:
                values = [str(r[f'a.{shuxing}']) for r in res if r[f'a.{shuxing}']]
                clean_values = []
                for value in values:
                    if '评分' in shuxing:
                        clean_value = data_validator.validate_rating(value)
                    elif '年份' in shuxing or '日期' in shuxing:
                        clean_value = data_validator.validate_year(value)
                    elif '片长' in shuxing:
                        clean_value = data_validator.validate_duration(value)
                    else:
                        clean_value = data_validator.clean_text(value)
                    
                    if clean_value is not None:
                        clean_values.append(str(clean_value))
                
                add_prompt += "、".join(clean_values) if clean_values else "暂无相关信息"
            else:
                add_prompt += "图谱中无信息，查找失败。"
        
        return f"<提示>用户对{entity}的{shuxing}查询结果如下：{add_prompt}</提示>"
    except Exception as e:
        st.error(f"查询错误: {str(e)}")
        return f"<提示>查询出错: {str(e)}</提示>"

def add_lianxi_prompt(entity: str, lianxi: str, target: str, client) -> str:
    """
    添加关系查询提示
    """
    add_prompt = ""
    try:
        entity = data_validator.clean_text(entity)
        
        # 直接查询电影名称
        sql_q = f"""
        MATCH (a:电影)
        WHERE a.名称 CONTAINS '{entity}'
        RETURN a.名称 as name
        """
        res = client.run(sql_q).data()
        
        if not res:
            return f"<提示>未找到电影：{entity}</提示>"
            
        movie_name = res[0]['name']
        
        # 构建关系查询
        if lianxi in ["导演", "演员", "编剧"]:
            relation_map = {
                "导演": "执导",
                "演员": "出演",
                "编剧": "编剧"
            }
            # 从人物指向电影的关系
            sql_q = f"""
            MATCH (b:{target})-[r:{relation_map[lianxi]}]->(a:电影)
            WHERE a.名称 = '{movie_name}'
            RETURN DISTINCT b.名称 as name
            """
        else:
            # 从电影指向其他实体的关系
            sql_q = f"""
            MATCH (a:电影)-[r:{lianxi}]->(b:{target})
            WHERE a.名称 = '{movie_name}'
            RETURN DISTINCT b.名称 as name
            """
        
        res = client.run(sql_q).data()
        if res:
            names = [r['name'] for r in res]
            add_prompt = "、".join(names)
        else:
            add_prompt = "暂无相关信息"
        
        return f"<提示>用户对{entity}的{lianxi}查询结果如下：{add_prompt}</提示>"
    except Exception as e:
        st.error(f"查询错误: {str(e)}")
        return f"<提示>查询出错: {str(e)}</提示>"

def generate_prompt(response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag):
    entities = zwk.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
    yitu = []
    prompt = "<指令>你是一个电影问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥。如果根据提示无法给出答案，立刻回答\"根据已知信息无法回答该问题\"。</指令>"
    prompt += "<指令>请你仅针对电影类问题提供简洁和专业的回答。如果问题不是电影相关的，你一定要回答\"我只能回答电影相关的问题\"，以明确告知你的回答限制。</指令>"
    
    # 定义查询映射，只包含数据集中实际存在的内容
    query_mappings = {
        "基本信息": ("基本信息", "shuxing"),
        "导演": ("导演", "lianxi"),
        "演员": ("演员", "lianxi"),
        "编剧": ("编剧", "lianxi"),
        "类型": ("类型", "lianxi"),
        "评分": ("豆瓣评分", "shuxing"),
        "年份": ("上映日期", "shuxing"),
        "评价人数": ("评价人数", "shuxing"),
        "国家": ("制片国家", "shuxing"),
        "语言": ("语言", "shuxing"),
        "剧情": ("剧情简介", "shuxing"),
        "时长": ("片长", "shuxing")
    }
    
    # 处理所有查询类型
    if '电影' in entities:
        for query_type, (field, query_method) in query_mappings.items():
            if query_type in response:
                if query_method == "shuxing":
                    prompt += add_shuxing_prompt(entities['电影'], field, client)
                else:
                    prompt += add_lianxi_prompt(entities['电影'], field, field.title(), client)
                yitu.append(f'查询{query_type}')
    
    # 处理电影推荐
    if "推荐" in response:
        user_preferences = {}
        if '类型' in entities:
            user_preferences['genre'] = entities['类型']
            prompt += f"<提示>用户想要{entities['类型']}类型的电影推荐。</提示>"
            yitu.append('类型推荐')
        elif '电影' in entities:
            user_preferences['similar_to'] = entities['电影']
            prompt += f"<提示>用户想要与{entities['电影']}相似的电影推荐。</提示>"
            yitu.append('相似电影推荐')
        else:
            prompt += f"<提示>用户想要电影推荐。</提示>"
            yitu.append('电影推荐')
    
        # 获取电影推荐结果
        recommendations = get_movie_recommendations(user_preferences, client)
        if recommendations:
            prompt += "<提示>为您推荐以下电影：\n"
            for i, movie in enumerate(recommendations, 1):
                prompt += f"{i}. {movie['电影名称']}（{movie['年份']}年）\n"
                prompt += f"   评分：{movie['评分']}分\n"
                prompt += f"   类型：{', '.join(movie.get('电影类型', []))}\n"
                prompt += f"   简介：{movie.get('简介', '暂无简介')}\n\n"
            prompt += "</提示>"
        else:
            prompt += "<提示>抱歉，未能找到符合要求的电影推荐。</提示>"
    
    # 检查是否有查询结果（修改判断逻辑）
    has_content = False
    for tag in ["<提示>", "</提示>"]:
        if tag in prompt:
            has_content = True
            break
    
    if not has_content:
        prompt += f"<提示>提示：知识库异常，没有相关信息！请你直接回答\"根据已知信息无法回答该问题\"！</提示>"
    
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += f"<注意>现在你已经知道给定的\"<提示></提示>\"和\"<用户问题></用户问题>\"了,你要极其认真的判断提示里是否有用户问题所需的信息，如果没有相关信息，你必须直接回答\"根据已知信息无法回答该问题\"。</注意>"
    prompt += f"<注意>你一定要再次检查你的回答是否完全基于\"<提示></提示>\"的内容，不可产生提示之外的答案！换而言之，你起到的作用仅仅是整合提示的功能，你一定不可以利用自身已经存在的知识进行回答，你必须从提示中找到问题的答案！</注意>"
    prompt += f"<注意>你必须充分的利用提示中的知识，不可将提示中的任何信息遗漏，你必须做到对提示信息的充分整合。你回答的任何一句话必须在提示中有所体现！如果根据提示无法给出答案，你必须回答\"根据已知信息无法回答该问题\"。</注意>"
    
    return prompt, "、".join(yitu), entities

def ans_stream(prompt):
    
    result = ""
    for res,his in glm_model.stream_chat(glm_tokenizer, prompt, history=[]):
        yield res

def main(is_admin, usname):
    # 设置背景图片
    set_background()
    
    # 定义模型缓存名称
    cache_model = 'best_roberta_gru_model_ent_aug'
    
    # 创建侧边栏
    with st.sidebar:
        st.markdown(f'<img src="data:image/jpg;base64,{get_base64_of_bin_file(os.path.join("img", "movie_logo.jpg"))}" style="width: 80%; display: block; margin: 0 auto; border-radius: 15px; border: 2px solid rgba(255, 255, 255, 0.2); padding: 10px; background-color: rgba(255, 255, 255, 0.1); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
        st.markdown(f"""<div style="text-align: center; margin-top: 1.5rem; margin-bottom: 1rem; color: rgba(255, 255, 255, 0.8); font-size: 1rem;">欢迎您，{'管理员 ' if is_admin else '用户 '}{usname}！</div>""", unsafe_allow_html=True)
        
        # 对话窗口管理
        st.markdown("### 💭 对话窗口")
        
        # 使用容器来固定高度
        window_container = st.container()
        with window_container:
            if st.button('➕ 新建对话窗口', use_container_width=True):
                if 'chat_windows' not in st.session_state:
                    st.session_state.chat_windows = [{"id": 1, "messages": []}]
                    st.session_state.active_window = 1
                    st.session_state.next_window_id = 2
                else:
                    new_window = {
                        "id": st.session_state.next_window_id,
                        "messages": []
                    }
                    st.session_state.chat_windows.append(new_window)
                    st.session_state.active_window = new_window["id"]
                    st.session_state.next_window_id += 1
                st.experimental_rerun()

            # 显示现有对话窗口
            if 'chat_windows' in st.session_state:
                for window in st.session_state.chat_windows:
                    cols = st.columns([0.85, 0.15])
                    with cols[0]:
                        button_type = "primary" if window['id'] == st.session_state.get('active_window') else "secondary"
                        if st.button(
                            f"💬 对话 {window['id']}",
                            key=f"window_{window['id']}",
                            help="点击切换到此窗口",
                            use_container_width=True,
                            type=button_type
                        ):
                            st.session_state.active_window = window['id']
                            st.experimental_rerun()
                    
                    if len(st.session_state.chat_windows) > 1:
                        with cols[1]:
                            if st.button(
                                "🗑️",
                                key=f"delete_{window['id']}",
                                help="删除此窗口",
                                use_container_width=True
                            ):
                                st.session_state.chat_windows = [w for w in st.session_state.chat_windows if w['id'] != window['id']]
                                if window['id'] == st.session_state.active_window:
                                    st.session_state.active_window = st.session_state.chat_windows[0]['id']
                                st.experimental_rerun()

        st.markdown("---")

        selected_option = st.selectbox(
            label='选择大语言模型:',
            options=['通义千问-Plus', 'DeepSeek-v3']
        )
        choice = 'qwen-plus' if selected_option == '通义千问-Plus' else 'deepseek-v3'

        # 初始化调试选项
        show_ent = False
        show_int = False
        show_prompt = False
        
        if is_admin:
            st.markdown("### ⚙️ 调试选项")
            show_ent = st.checkbox("显示实体识别结果")
            show_int = st.checkbox("显示意图识别结果")
            show_prompt = st.checkbox("显示查询的知识库信息")
            if st.button('🔧 修改知识图谱'):
                st.markdown('[点击这里修改知识图谱](http://127.0.0.1:7474/)', unsafe_allow_html=True)

        if st.button("🚪 退出登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.experimental_rerun()

    # 加载模型和数据库连接
    glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(cache_model)
    client = py2neo.Graph(uri="bolt://localhost:7687", auth=("neo4j", "11111111"))

    # 页面标题
    st.title("🎬 泡泡Dragon电影助手")

    # 创建标签页
    tab1, tab2 = st.tabs(["💬 智能问答", "🎯 电影推荐"])

    # 显示标签页内容
    with tab1:
        chat_interface(is_admin, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag, choice, show_ent, show_int, show_prompt)

    with tab2:
        recommendation_interface(client)

def chat_interface(is_admin, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag, choice, show_ent, show_int, show_prompt):
    """
    聊天界面主函数
    """
    # 初始化对话窗口状态
    if 'chat_windows' not in st.session_state:
        st.session_state.chat_windows = [{"id": 1, "messages": []}]
        st.session_state.active_window = 1
        st.session_state.next_window_id = 2

    # 获取当前活动窗口的消息
    current_window = next(w for w in st.session_state.chat_windows if w['id'] == st.session_state.active_window)
    current_messages = current_window['messages']

    # 创建一个容器来包含所有内容
    main_container = st.container()
    
    # 创建输入框容器（固定在底部）
    input_container = st.container()
    
    # 在底部显示输入框
    with input_container:
        if query := st.chat_input("问我任何问题!", key=f"chat_input_{st.session_state.active_window}"):
            # 将用户输入添加到消息历史
            current_messages.append({"role": "user", "content": query})
            
            # 在主容器中显示消息历史和回复
            with main_container:
                # 显示历史消息
                for message in current_messages[:-1]:  # 显示除了最新消息之外的所有历史消息
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant":
                            if show_ent:
                                with st.expander("实体识别结果"):
                                    st.write(message.get("ent", ""))
                            if show_int:
                                with st.expander("意图识别结果"):
                                    st.write(message.get("yitu", ""))
                            if show_prompt:
                                with st.expander("点击显示知识库信息"):
                                    st.write(message.get("prompt", ""))

                # 显示最新的用户消息
                with st.chat_message("user"):
                    st.markdown(query)

                # 创建助手消息容器
                with st.chat_message("assistant"):
                    # 创建状态占位符
                    status_placeholder = st.empty()
                    
                    # 生成回复
                    status_placeholder.write("正在进行意图识别...")
                    response = Intent_Recognition(query, choice)
                    
                    prompt, yitu, entities = generate_prompt(response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag)

                    # 获取对应模型的客户端
                    model_client = get_model_client(choice)
                    
                    status_placeholder.write("正在生成回答...")
                    
                    # 提取知识库内容
                    knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
                    zhishiku_content = "\n".join([f"提示{idx + 1}, {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])
                    
                    # 使用一个占位符来显示生成的回答
                    message_placeholder = st.empty()
                    last = ""
                    
                    for chunk in model_client.chat.completions.create(
                            model=choice.lower(),
                            messages=[{'role': 'user', 'content': prompt}],
                            stream=True
                    ):
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            last += content
                            message_placeholder.markdown(last)
                    
                    # 清空状态提示
                    status_placeholder.empty()
                    
                    # 最终显示完整的回答
                    message_placeholder.markdown(last)
                    
                    # 显示调试信息
                    if show_ent:
                        with st.expander("实体识别结果"):
                            st.write(str(entities))
                    if show_int:
                        with st.expander("意图识别结果"):
                            st.write(yitu)
                    if show_prompt:
                        with st.expander("点击显示知识库信息"):
                            st.write(zhishiku_content)
                
                # 将助手的回复添加到消息历史
                current_messages.append({
                    "role": "assistant", 
                    "content": last, 
                    "yitu": yitu, 
                    "prompt": zhishiku_content, 
                    "ent": str(entities)
                })
        else:
            # 仅显示历史消息
            with main_container:
                for message in current_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant":
                            if show_ent:
                                with st.expander("实体识别结果"):
                                    st.write(message.get("ent", ""))
                            if show_int:
                                with st.expander("意图识别结果"):
                                    st.write(message.get("yitu", ""))
                            if show_prompt:
                                with st.expander("点击显示知识库信息"):
                                    st.write(message.get("prompt", ""))

def recommendation_interface(client):
    """
    电影推荐界面
    """
    # ========== 新增：全局可用的海报查找函数 ==========
    def get_movie_poster_path(movie):
        posters_dir = "data/posters"
        poster_file = movie.get("海报路径", "")
        candidates = []
        if poster_file:
            candidates.append(poster_file)
        # 中文名
        movie_name = movie.get("电影名称", "").split(' ')[0]
        candidates.append(movie_name)
        # 全名
        full_name = movie.get("电影名称", "")
        candidates.append(full_name)
        # 外文名
        if len(full_name.split(' ')) > 1:
            foreign_name = full_name.split(' ', 1)[1]
            candidates.append(foreign_name)

        if os.path.exists(posters_dir):
            files = os.listdir(posters_dir)
            norm_files = {normalize_filename(f.rsplit('.',1)[0]): f for f in files}
            # 1. 精确匹配
            for name in candidates:
                for ext in [".jpg", ".jpeg", ".png"]:
                    norm_name = normalize_filename(name)
                    for f in files:
                        if normalize_filename(f.rsplit('.',1)[0]) == norm_name and f.lower().endswith(ext):
                            return os.path.join(posters_dir, f)
                # 精确无扩展名
                norm_name = normalize_filename(name)
                if norm_name in norm_files:
                    return os.path.join(posters_dir, norm_files[norm_name])
            # 2. 模糊匹配（优先最长候选名）
            sorted_candidates = sorted(candidates, key=lambda x: -len(x))
            for name in sorted_candidates:
                norm_name = normalize_filename(name)
                for file_norm, file_real in norm_files.items():
                    if norm_name in file_norm or file_norm in norm_name:
                        return os.path.join(posters_dir, file_real)
        return os.path.join("img", "no_poster.jpg")
    # ========== END ==========

    st.markdown("### 🎬 电影推荐系统")
    
    # 创建三个标签页：搜索、推荐和图片搜索
    search_tab, filter_tab, image_tab = st.tabs(["🔍 电影搜索", "🎯 电影推荐", "🖼️ 图片搜索"])
    
    with image_tab:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0;">海报搜索</h4>
                <p style="margin: 0; opacity: 0.7;">上传电影海报图片，系统将为您找到相似的电影</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 初始化搜索状态
        if 'searching' not in st.session_state:
            st.session_state['searching'] = False

        uploaded_file = st.file_uploader("上传电影海报", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            if not st.session_state['searching']:
                st.session_state['searching'] = True
            # 只在搜索中且还没出结果时显示提示和取消按钮
            if st.session_state['searching']:
                st.info("正在搜索中，请稍候...", icon="🔍")
                if st.button("取消搜索"):
                    st.session_state['searching'] = False
                    st.warning("已取消搜索")
                    st.stop()
            if st.session_state['searching']:
                similar_movies = find_similar_movie_by_poster(uploaded_file, client)
                st.session_state['searching'] = False
                # 下面只显示结果，不再显示"正在搜索中"和"取消搜索"
                if similar_movies:
                    st.markdown(f"<div style='margin: 1rem 0;'>找到 {len(similar_movies)} 个相似结果</div>", unsafe_allow_html=True)
                    for movie_name, similarity in similar_movies:
                        # 获取电影详细信息
                        query = """
                        MATCH (m:电影)
                        WHERE m.名称 = $movie_name
                        WITH m
                        OPTIONAL MATCH (m)-[:属于]->(g:类型)
                        WITH m, collect(g.名称) as genres
                        OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                        WITH m, genres, collect(d.名称) as directors
                        OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                        WITH m, genres, directors, collect(a.名称)[..3] as actors
                        RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                               m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                               m.语言 as 语言, genres as 电影类型, directors as 导演,
                               actors as 主演, m.评价人数 as 评价人数, m.海报路径 as 海报路径, m.又名 as 又名
                        """
                        movie_data = client.run(query, movie_name=movie_name).data()[0]
                        display_movie_with_poster(movie_data)
                        st.markdown(f"<div style='text-align: right; opacity: 0.7;'>相似度: {similarity:.2%}</div>", unsafe_allow_html=True)
                else:
                    st.warning("未找到相似的电影海报或已取消搜索")
    
    with search_tab:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0;">电影搜索</h4>
                <p style="margin: 0; opacity: 0.7;">支持搜索电影名称、导演、演员等信息</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 搜索选项和输入框
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("输入搜索内容", placeholder="电影名称/导演/演员...")
        with col2:
            search_type = st.selectbox("搜索类型", ["电影名称", "导演", "演员"])
        
        if search_query:
            # 构建搜索查询
            if search_type == "电影名称":
                query = """
                MATCH (m:电影)
                WHERE m.名称 CONTAINS $search_term OR m.又名 CONTAINS $search_term
                WITH m
                OPTIONAL MATCH (m)-[:属于]->(g:类型)
                WITH m, collect(g.名称) as genres
                OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                WITH m, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                WITH m, genres, directors, collect(a.名称)[..3] as actors
                RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                       m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                       m.语言 as 语言, genres as 电影类型, directors as 导演,
                       actors as 主演, m.评价人数 as 评价人数, m.海报路径 as 海报路径, m.又名 as 又名
                ORDER BY m.豆瓣评分 DESC
                """
            elif search_type == "导演":
                query = """
                MATCH (d:导演)-[:执导]->(m:电影)
                WHERE d.名称 CONTAINS $search_term
                WITH m
                OPTIONAL MATCH (m)-[:属于]->(g:类型)
                WITH m, collect(g.名称) as genres
                OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                WITH m, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                WITH m, genres, directors, collect(a.名称)[..3] as actors
                RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                       m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                       m.语言 as 语言, genres as 电影类型, directors as 导演,
                       actors as 主演, m.评价人数 as 评价人数, m.海报路径 as 海报路径, m.又名 as 又名
                ORDER BY m.豆瓣评分 DESC
                """
            else:  # 演员搜索
                query = """
                MATCH (a:演员)-[:出演]->(m:电影)
                WHERE a.名称 CONTAINS $search_term
                WITH m
                OPTIONAL MATCH (m)-[:属于]->(g:类型)
                WITH m, collect(g.名称) as genres
                OPTIONAL MATCH (m)<-[:执导]-(d:导演)
                WITH m, genres, collect(d.名称) as directors
                OPTIONAL MATCH (m)<-[:出演]-(a:演员)
                WITH m, genres, directors, collect(a.名称)[..3] as actors
                RETURN m.名称 as 电影名称, m.豆瓣评分 as 评分, m.上映日期 as 年份,
                       m.剧情简介 as 简介, m.片长 as 时长, m.制片国家 as 国家地区,
                       m.语言 as 语言, genres as 电影类型, directors as 导演,
                       actors as 主演, m.评价人数 as 评价人数, m.海报路径 as 海报路径, m.又名 as 又名
                ORDER BY m.豆瓣评分 DESC
                """
            # 执行搜索
            try:
                results = client.run(query, search_term=search_query).data()
                
                if results:
                    st.markdown(f"<div style='margin: 1rem 0;'>找到 {len(results)} 个相关结果</div>", unsafe_allow_html=True)
                    
                    # 清理和显示结果
                    for movie in results:
                        # 处理电影类型
                        movie_types = movie.get('电影类型', [])
                        if isinstance(movie_types, str):
                            movie_types = [t.strip() for t in movie_types.split(',')]
                        elif not isinstance(movie_types, list):
                            movie_types = []
                        
                        # 修正电影名称展示逻辑
                        movie_name = movie['电影名称']
                        # 针对小森林系列特殊处理
                        if '小森林' in movie_name:
                            if '夏秋' in movie_name or '夏・秋' in movie_name:
                                chinese_name = '小森林 夏秋篇'
                                foreign_name = '（リトル・フォレスト 夏・秋）'
                            elif '冬春' in movie_name or '冬・春' in movie_name:
                                chinese_name = '小森林 冬春篇'
                                foreign_name = '（リトル・フォレスト 冬・春）'
                            else:
                                parts = movie_name.split(' ', 1)
                                chinese_name = parts[0]
                                foreign_name = f"（{parts[1]}）" if len(parts) > 1 else ""
                        else:
                            parts = movie_name.split(' ', 1)
                            chinese_name = parts[0]
                            foreign_name = f"（{parts[1]}）" if len(parts) > 1 else ""
                        
                        # 处理语言信息
                        languages = movie.get('语言', '未知')
                        if isinstance(languages, list):
                            languages = '、'.join(languages)
                        
                        # 构建类型标签HTML
                        type_tags = ''.join([f'<span class="movie-type-tag">{t}</span>' for t in movie_types])
                        
                        # 获取海报路径，复用推荐部分的查找逻辑
                        poster_path = get_movie_poster_path(movie)
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown("<div style='margin-right: 0px; padding-right: 0px;'>", unsafe_allow_html=True)
                            img_base64 = get_img_base64(poster_path)
                            if img_base64:
                                st.markdown(
                                    f"<img src='data:image/jpg;base64,{img_base64}' style='width:200px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); display:block;'>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<div style='width:200px;height:280px;background:#eee;border-radius:8px;'></div>",
                                    unsafe_allow_html=True
                                )
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            # 修正时长展示，避免出现两次"分钟"
                            duration = str(movie.get('时长', ''))
                            if duration.endswith('分钟'):
                                duration_str = duration
                            else:
                                duration_str = duration + '分钟' if duration else ''
                            # foreign_name 后加又名
                            aka = f" / {movie['又名']}" if movie.get('又名') else ""
                            st.markdown(f"""
                                <div class="movie-card" style="margin:0; width:100%; max-width:100%; box-sizing:border-box; display:flex; flex-direction:column;">
                                    <div class="movie-title">{chinese_name}</div>
                                    <div class="movie-alias">{foreign_name}{aka}</div>
                                    <div class="movie-info">
                                        <div class="movie-info-item">⭐ 评分：{movie['评分']}</div>
                                        <div class="movie-info-item">📅 年份：{movie['年份']}</div>
                                        <div class="movie-info-item">⏱️ 时长：{duration_str}</div>
                                        <div class="movie-info-item">🌍 国家：{movie['国家地区']}</div>
                                        <div class="movie-info-item">🗣️ 语言：{languages}</div>
                                    </div>
                                    <div class="movie-types">{type_tags}</div>
                                    <div class="movie-synopsis">
                                        <strong>📝 剧情简介：</strong><br>
                                        {movie.get('简介', '暂无简介')}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning('未找到与"' + search_query + '"相关的' + search_type + '信息')
            except Exception as e:
                st.error(f"搜索出错: {str(e)}")
    
    with filter_tab:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0;">电影推荐</h4>
                <p style="margin: 0; opacity: 0.7;">根据您的偏好推荐电影</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 创建过滤器
        with st.expander("推荐过滤器", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_rating = st.slider("最低评分", 0.0, 10.0, 0.0, 0.1)
                year_range = st.slider("年份范围", 1900, 2024, (1900, 2024))
            
            with col2:
                duration_range = st.slider("时长范围(分钟)", 60, 240, (60, 240))
                genres = ["全部", "动作", "喜剧", "剧情", "科幻", "恐怖", "爱情", "动画", "悬疑", "犯罪", "战争"]
                selected_genre = st.selectbox("电影类型", genres)
            
            with col3:
                countries = ["全部", "中国", "美国", "日本", "韩国", "英国", "法国", "德国", "印度"]
                selected_country = st.selectbox("国家地区", countries)
                sort_by = st.selectbox("排序方式", ["评分", "上映日期", "评价人数"])

        # 构建推荐参数
        user_preferences = {}
        if min_rating > 0:
            user_preferences['rating'] = min_rating
        if year_range != (1900, 2024):
            user_preferences['year'] = year_range
        if duration_range != (60, 240):
            user_preferences['duration'] = duration_range
        if selected_genre != "全部":
            user_preferences['genre'] = selected_genre
        if selected_country != "全部":
            user_preferences['country'] = selected_country

        # 获取推荐结果
        recommendations = get_movie_recommendations(user_preferences, client)

        # 分页加载逻辑
        if 'shown_count' not in st.session_state or st.session_state.get('last_filter', None) != str(user_preferences):
            st.session_state['shown_count'] = 20
            st.session_state['last_filter'] = str(user_preferences)
        shown_count = st.session_state['shown_count']
        total_count = len(recommendations)
        show_recommendations = recommendations[:shown_count]

        # 在过滤器下方显示总数
        st.markdown(f"<div style='margin: 1rem 0;'>共找到 <b>{total_count}</b> 部符合条件的电影</div>", unsafe_allow_html=True)

        if show_recommendations:
            # 根据选择的方式排序
            if sort_by == "评分":
                show_recommendations.sort(key=lambda x: float(x['评分']), reverse=True)
            elif sort_by == "上映日期":
                show_recommendations.sort(key=lambda x: extract_earliest_date(str(x['年份'])), reverse=True)
            elif sort_by == "评价人数":
                show_recommendations.sort(key=lambda x: int(x.get('评价人数', 0)), reverse=True)

            for movie in show_recommendations:
                # 处理电影类型
                movie_types = movie.get('电影类型', [])
                if isinstance(movie_types, str):
                    movie_types = [t.strip() for t in movie_types.split(',')]
                elif not isinstance(movie_types, list):
                    movie_types = []
                
                # 修正电影名称展示逻辑
                movie_name = movie['电影名称']
                if '小森林' in movie_name:
                    if '夏秋' in movie_name or '夏・秋' in movie_name:
                        chinese_name = '小森林 夏秋篇'
                        foreign_name = '（リトル・フォレスト 夏・秋）'
                    elif '冬春' in movie_name or '冬・春' in movie_name:
                        chinese_name = '小森林 冬春篇'
                        foreign_name = '（リトル・フォレスト 冬・春）'
                    else:
                        parts = movie_name.split(' ', 1)
                        chinese_name = parts[0]
                        foreign_name = f"（{parts[1]}）" if len(parts) > 1 else ""
                else:
                    parts = movie_name.split(' ', 1)
                    chinese_name = parts[0]
                    foreign_name = f"（{parts[1]}）" if len(parts) > 1 else ""
                
                # 处理语言信息
                languages = movie.get('语言', '未知')
                if isinstance(languages, list):
                    languages = '、'.join(languages)
                
                # 构建类型标签HTML
                type_tags = ''.join([f'<span class="movie-type-tag">{t}</span>' for t in movie_types])
                
                # 使用st.columns布局，左侧显示海报，右侧显示文字信息，缩小间距
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("<div style='margin-right: 0px; padding-right: 0px;'>", unsafe_allow_html=True)
                    img_base64 = get_img_base64(get_movie_poster_path(movie))
                    if img_base64:
                        st.markdown(
                            f"<img src='data:image/jpg;base64,{img_base64}' style='width:200px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); display:block;'>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='width:200px;height:280px;background:#eee;border-radius:8px;'></div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    # 修正时长展示，避免出现两次"分钟"
                    duration = str(movie.get('时长', ''))
                    if duration.endswith('分钟'):
                        duration_str = duration
                    else:
                        duration_str = duration + '分钟' if duration else ''
                    # foreign_name 后加又名
                    aka = f" / {movie['又名']}" if movie.get('又名') else ""
                    st.markdown(f"""
                        <div class="movie-card" style="margin:0; width:100%; max-width:100%; box-sizing:border-box; display:flex; flex-direction:column;">
                            <div class="movie-title">{chinese_name}</div>
                            <div class="movie-alias">{foreign_name}{aka}</div>
                            <div class="movie-info">
                                <div class="movie-info-item">⭐ 评分：{movie['评分']}</div>
                                <div class="movie-info-item">📅 年份：{movie['年份']}</div>
                                <div class="movie-info-item">⏱️ 时长：{duration_str}</div>
                                <div class="movie-info-item">🌍 国家：{movie['国家地区']}</div>
                                <div class="movie-info-item">🗣️ 语言：{languages}</div>
                            </div>
                            <div class="movie-types">{type_tags}</div>
                            <div class="movie-synopsis">
                                <strong>📝 剧情简介：</strong><br>
                                {movie.get('简介', '暂无简介')}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            if shown_count < total_count:
                if st.button("查看更多"):
                    st.session_state['shown_count'] += 20
                    st.experimental_rerun()
            else:
                st.markdown("<div style='text-align:center;color:#aaa;margin:1.5em 0;'>没有更多啦~</div>", unsafe_allow_html=True)
        else:
            st.warning("抱歉，未找到符合条件的电影推荐。请尝试调整过滤条件。")

    st.markdown('</div>', unsafe_allow_html=True)

# 数据验证和清洗工具
class DataValidator:
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本数据"""
        if not isinstance(text, str):
            return ""
        # 移除特殊字符
        text = re.sub(r'[\\/:*?"<>|]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def validate_year(year: str) -> Optional[int]:
        """验证年份格式"""
        if not year:
            return None
        match = re.search(r'\d{4}', str(year))
        if match:
            year_int = int(match.group())
            if 1900 <= year_int <= datetime.now().year:
                return year_int
        return None

    @staticmethod
    def validate_rating(rating: Any) -> Optional[float]:
        """验证评分格式"""
        try:
            rating_float = float(rating)
            if 0 <= rating_float <= 10:
                return round(rating_float, 1)
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def validate_duration(duration: Any) -> Optional[int]:
        """验证时长格式"""
        try:
            duration_str = str(duration)
            # 提取数字
            match = re.search(r'\d+', duration_str)
            if match:
                duration_int = int(match.group())
                if 0 < duration_int < 1000:  # 合理的电影时长范围
                    return duration_int
        except (ValueError, TypeError):
            pass
        return None

class TextSimilarity:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            analyzer='char',
            ngram_range=(2, 3)
        )
        self.vectors = None
        self.texts = None

    def fit(self, texts: List[str]):
        """训练文本向量化模型"""
        self.texts = texts
        self.vectors = self.vectorizer.fit_transform(texts)
        return self

    def find_similar(self, query: str, threshold: float = 0.5, top_k: int = 5) -> List[Tuple[str, float]]:
        """查找相似文本"""
        if not self.vectors or not self.texts:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # 获取相似度大于阈值的结果
        similar_indices = np.where(similarities >= threshold)[0]
        similar_scores = similarities[similar_indices]
        
        # 按相似度排序
        sorted_indices = np.argsort(similar_scores)[::-1][:top_k]
        
        return [(self.texts[similar_indices[i]], similar_scores[i]) 
                for i in sorted_indices]

# 初始化全局工具类实例
data_validator = DataValidator()
text_similarity = TextSimilarity()

def get_image_features(img_path):
    """提取图片特征"""
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def find_similar_movie_by_poster(uploaded_file, client):
    """通过海报图片查找相似电影"""
    # 加载缓存特征
    features = np.load("posters_features.npy")
    names = np.load("posters_names.npy")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    temp_path = "temp_poster.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    img = image.load_img(temp_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    query_feat = model.predict(x).flatten().reshape(1, -1)
    sims = cosine_similarity(query_feat, features)[0]
    top_idx = sims.argsort()[::-1][:5]
    os.remove(temp_path)
    # 取出电影名
    result = []
    for idx in top_idx:
        # 检查是否被取消
        if 'searching' in st.session_state and not st.session_state['searching']:
            return []
        poster_name = str(names[idx])  # 强制转为Python str
        movie = client.run("MATCH (m:电影) WHERE m.海报路径 = $p RETURN m.名称", p=poster_name).data()
        if movie:
            result.append((movie[0]['m.名称'], sims[idx]))
    return result

def display_movie_info(movie_data):
    """生成卡片格式的电影详细信息HTML，风格与搜索/推荐页面一致"""
    def get_any(keys, default="未知"):
        for k in keys:
            v = movie_data.get(k)
            if v and v != "未知":
                return v
        return default

    movie_name = get_any(['电影名称', '电影名', '名称'])
    movie_types = get_any(['类型', '电影类型'], [])
    if isinstance(movie_types, str):
        movie_types = [t.strip() for t in movie_types.split(',') if t.strip()]
    elif not isinstance(movie_types, list):
        movie_types = []
    directors = get_any(['导演'], [])
    if isinstance(directors, str):
        directors = [d.strip() for d in str(directors).split(',') if d.strip()]
    elif not isinstance(directors, list):
        directors = []
    actors = get_any(['主演'], [])
    if isinstance(actors, str):
        actors = [a.strip() for a in str(actors).split(',') if a.strip()]
    elif not isinstance(actors, list):
        actors = []
    country = get_any(['地区', '国家地区', '制片国家'])
    year = get_any(['年份', '上映日期'])
    rating = get_any(['评分', '豆瓣评分'])
    duration = get_any(['时长'])
    languages = get_any(['语言'], '未知')
    if isinstance(languages, list):
        languages = '、'.join(languages)
    synopsis = get_any(['简介', '剧情简介'], '暂无简介')
    aka = get_any(['又名'], '')

    type_tags = ''.join([f'<span class="movie-type-tag">{t}</span>' for t in movie_types])
    director_str = '、'.join(directors) if directors else '未知'
    actor_str = '、'.join(actors) if actors else '未知'
    aka_str = f" / {aka}" if aka else ""
    duration_str = f"{duration}分钟" if duration and not str(duration).endswith('分钟') else str(duration)

    html = f'''
    <div class="movie-card" style="margin-left: 0;">
        <div class="movie-title">{movie_name}</div>
        <div class="movie-alias">{aka_str}</div>
        <div class="movie-info">
            <div class="movie-info-item">⭐ 评分：{rating}</div>
            <div class="movie-info-item">📅 年份：{year}</div>
            <div class="movie-info-item">⏱️ 时长：{duration_str}</div>
            <div class="movie-info-item">🌍 国家：{country}</div>
            <div class="movie-info-item">🗣️ 语言：{languages}</div>
            <div class="movie-info-item">🎬 导演：{director_str}</div>
            <div class="movie-info-item">👤 主演：{actor_str}</div>
        </div>
        <div class="movie-types">{type_tags}</div>
        <div class="movie-synopsis">
            <strong>📝 剧情简介：</strong><br>
            {synopsis}
        </div>
    </div>
    '''
    return html

def display_movie_with_poster(movie_data):
    """显示电影信息及其海报（左右结构）"""
    poster_path = os.path.join("data", "posters", movie_data.get("海报路径", ""))
    col1, col2 = st.columns([1, 2])
    with col1:
        img_base64 = get_img_base64(poster_path) if os.path.exists(poster_path) else None
        if img_base64:
            st.markdown(f"<img src='data:image/jpg;base64,{img_base64}' style='width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);display:block;'>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='width:200px;height:280px;background:#eee;border-radius:8px;'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(display_movie_info(movie_data), unsafe_allow_html=True)

def normalize_filename(name):
    name = unicodedata.normalize('NFKC', name)
    name = ''.join(e for e in name if e.isalnum()).lower()
    return name

def get_img_base64(img_path):
    try:
        with open(img_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

def extract_earliest_date(date_str):
    if not date_str:
        return 99999999  # 排到最后
    # 匹配所有形如 2005-09-02 的日期
    dates = re.findall(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if not dates:
        # 只年
        years = re.findall(r'(\d{4})', date_str)
        if years:
            return int(years[0] + '0101')
        return 99999999
    # 转为整数比较
    min_date = min(int(y + m + d) for y, m, d in dates)
    return min_date
