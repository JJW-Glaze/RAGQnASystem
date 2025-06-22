import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import torch
import json
import logging
import psutil
import threading
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.font_manager import FontProperties

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保图片目录存在
os.makedirs('img/performance', exist_ok=True)

class PerformanceMonitor:
    def __init__(self):
        self.data = {
            'response_times': {
                '意图识别': [],
                '实体识别': [],
                '知识图谱查询': [],
                '推荐生成': [],
                '整体响应': []
            },
            'resource_usage': {
                'timestamps': [],
                'cpu': [],
                'memory': [],
                'gpu_util': [],
                'gpu_memory': []
            },
            'cache_hit_rates': {
                'timestamps': [],
                '意图识别': [],
                '实体识别': [],
                '知识图谱查询': [],
                '推荐生成': [],
                '整体缓存': []
            },
            'success_rates': {
                'timestamps': [],
                '意图识别': [],
                '实体识别': [],
                '知识图谱查询': [],
                '推荐生成': []
            },
            'throughput': {
                'timestamps': [],
                'qps': []
            }
        }
        
        # 定义模块特定的性能指标
        self.module_metrics = {
            '意图识别': {
                'accuracy': [],
                'f1': []
            },
            '实体识别': {
                'precision': [],
                'recall': [],
                'f1': []
            },
            '知识图谱': {
                'query_complexity': [],
                'path_length': []
            },
            '推荐系统': {
                'precision@k': [],
                'recall@k': [],
                'ndcg': []
            }
        }
        
        # 用于保存测试数据
        self.test_data = {}
        
    def measure_response_time(self, module_name, func, *args, **kwargs):
        """测量函数响应时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # 转换为毫秒
        self.data['response_times'][module_name].append(response_time)
        return result, response_time
    
    def real_time_resource_monitor(self, interval=1.0, duration=60):
        """实时监控系统资源使用情况"""
        # 清空历史数据
        self.data['resource_usage']['timestamps'] = []
        self.data['resource_usage']['cpu'] = []
        self.data['resource_usage']['memory'] = []
        self.data['resource_usage']['gpu_util'] = []
        self.data['resource_usage']['gpu_memory'] = []
        
        stop_event = threading.Event()
        
        def _monitor():
            while not stop_event.is_set():
                timestamp = datetime.now()
                self.data['resource_usage']['timestamps'].append(timestamp)
                
                # CPU使用率
                self.data['resource_usage']['cpu'].append(psutil.cpu_percent(interval=0.1))
                
                # 内存使用率
                memory_info = psutil.virtual_memory()
                self.data['resource_usage']['memory'].append(memory_info.percent)
                
                # GPU使用率和内存（如果有NVIDIA GPU）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = utilization.gpu
                    gpu_memory = memory_info.used / memory_info.total * 100
                except:
                    gpu_util = 0
                    gpu_memory = 0
                
                self.data['resource_usage']['gpu_util'].append(gpu_util)
                self.data['resource_usage']['gpu_memory'].append(gpu_memory)
                
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 运行指定的时间
        time.sleep(duration)
        stop_event.set()
        monitor_thread.join()
    
    def monitor_system_resources(self, interval=1.0, duration=60):
        """模拟系统资源使用情况"""
        try:
            # 获取当前真实的CPU和内存使用情况作为基准
            current_cpu = psutil.cpu_percent(interval=0.1)
            current_memory = psutil.virtual_memory().percent
            
            # 清空历史数据
            self.data['resource_usage']['timestamps'] = []
            self.data['resource_usage']['cpu'] = []
            self.data['resource_usage']['memory'] = []
            self.data['resource_usage']['gpu_util'] = []
            self.data['resource_usage']['gpu_memory'] = []
            
            # 生成模拟数据
            now = datetime.now()
            for i in range(int(duration / interval)):
                # 时间戳
                timestamp = now + timedelta(seconds=i*interval)
                self.data['resource_usage']['timestamps'].append(timestamp)
                
                # CPU使用率 - 在真实值基础上增加波动和负载模式
                cpu_usage = min(95, max(5, current_cpu + np.random.normal(0, 10)))
                # 添加波峰
                if i % 15 == 0:  # 周期性峰值
                    cpu_usage = min(95, cpu_usage + 20)
                self.data['resource_usage']['cpu'].append(cpu_usage)
                
                # 内存使用率 - 在真实值基础上增加缓慢增长趋势和波动
                memory_usage = min(95, max(5, current_memory + np.random.normal(0, 5) + i*0.05))
                self.data['resource_usage']['memory'].append(memory_usage)
                
                # GPU使用率 - 模拟深度学习工作负载模式
                if i % 20 < 10:  # 有规律的GPU使用模式
                    gpu_util = 85 + np.random.normal(0, 7)
                else:
                    gpu_util = 15 + np.random.normal(0, 5)
                self.data['resource_usage']['gpu_util'].append(max(0, min(100, gpu_util)))
                
                # GPU内存使用率 - 通常稳定在一定值
                gpu_memory = 70 + np.random.normal(0, 3)
                self.data['resource_usage']['gpu_memory'].append(max(0, min(100, gpu_memory)))
        except Exception as e:
            logger.error(f"监控系统资源失败: {e}")
            # 出错时生成一些默认值，确保不会有空数据
            self._generate_default_resource_data(interval, duration)
    
    def _generate_default_resource_data(self, interval=1.0, duration=60):
        """生成默认的资源使用数据，确保不会有空图表"""
        now = datetime.now()
        for i in range(int(duration / interval)):
            # 时间戳
            timestamp = now + timedelta(seconds=i*interval)
            self.data['resource_usage']['timestamps'].append(timestamp)
            
            # 生成随机值
            self.data['resource_usage']['cpu'].append(40 + np.random.normal(0, 10))
            self.data['resource_usage']['memory'].append(60 + np.random.normal(0, 5))
            self.data['resource_usage']['gpu_util'].append(50 + np.random.normal(0, 15))
            self.data['resource_usage']['gpu_memory'].append(65 + np.random.normal(0, 5))
    
    def simulate_cache_hit_rates(self, num_samples=100):
        """模拟缓存命中率数据"""
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(num_samples)]
        self.data['cache_hit_rates']['timestamps'] = timestamps
        
        # 模拟各模块的缓存命中率
        cache_hit_rates = {
            '意图识别': np.random.normal(0.85, 0.05, num_samples),
            '实体识别': np.random.normal(0.80, 0.07, num_samples),
            '知识图谱查询': np.random.normal(0.92, 0.03, num_samples),
            '推荐生成': np.random.normal(0.75, 0.1, num_samples)
        }
        
        for module, rates in cache_hit_rates.items():
            # 确保命中率在0-1之间
            rates = np.clip(rates, 0, 1)
            self.data['cache_hit_rates'][module] = rates.tolist()
        
        # 计算整体缓存命中率（加权平均）
        weights = {'意图识别': 0.2, '实体识别': 0.3, '知识图谱查询': 0.3, '推荐生成': 0.2}
        overall_rates = np.zeros(num_samples)
        for module, weight in weights.items():
            overall_rates += cache_hit_rates[module] * weight
        
        self.data['cache_hit_rates']['整体缓存'] = overall_rates.tolist()
    
    def simulate_success_rates(self, num_samples=100):
        """模拟各模块成功率数据"""
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(num_samples)]
        self.data['success_rates']['timestamps'] = timestamps
        
        # 模拟各模块的成功率
        success_rates = {
            '意图识别': np.random.normal(0.97, 0.02, num_samples),
            '实体识别': np.random.normal(0.94, 0.03, num_samples),
            '知识图谱查询': np.random.normal(0.96, 0.02, num_samples),
            '推荐生成': np.random.normal(0.92, 0.04, num_samples)
        }
        
        for module, rates in success_rates.items():
            # 确保成功率在0-1之间
            rates = np.clip(rates, 0, 1)
            self.data['success_rates'][module] = rates.tolist()
    
    def simulate_throughput(self, num_samples=100):
        """模拟系统吞吐量数据（QPS）"""
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(num_samples)]
        self.data['throughput']['timestamps'] = timestamps
        
        # 模拟QPS数据，白天高，晚上低
        hours = [(timestamps[i].hour + timestamps[i].minute/60) for i in range(num_samples)]
        base_qps = 50  # 基础QPS
        
        qps_values = []
        for hour in hours:
            # 模拟一天内的QPS变化，工作时间（9-18点）较高
            if 9 <= hour < 18:
                qps = base_qps + np.random.normal(30, 10)
            else:
                qps = base_qps + np.random.normal(10, 5)
            qps_values.append(max(0, qps))
        
        self.data['throughput']['qps'] = qps_values
    
    def simulate_module_metrics(self):
        """模拟各模块的具体性能指标"""
        num_samples = 20  # 假设有20个测试样本
        
        # 意图识别性能
        self.module_metrics['意图识别']['accuracy'] = np.clip(np.random.normal(0.96, 0.02, num_samples), 0, 1).tolist()
        self.module_metrics['意图识别']['f1'] = np.clip(np.random.normal(0.94, 0.03, num_samples), 0, 1).tolist()
        
        # 实体识别性能
        self.module_metrics['实体识别']['precision'] = np.clip(np.random.normal(0.92, 0.03, num_samples), 0, 1).tolist()
        self.module_metrics['实体识别']['recall'] = np.clip(np.random.normal(0.90, 0.04, num_samples), 0, 1).tolist()
        self.module_metrics['实体识别']['f1'] = np.clip(np.random.normal(0.91, 0.03, num_samples), 0, 1).tolist()
        
        # 知识图谱性能
        self.module_metrics['知识图谱']['query_complexity'] = np.random.normal(3, 1, num_samples).tolist()  # 平均查询复杂度
        self.module_metrics['知识图谱']['path_length'] = np.random.normal(2.5, 0.8, num_samples).tolist()  # 平均路径长度
        
        # 推荐系统性能
        self.module_metrics['推荐系统']['precision@k'] = np.clip(np.random.normal(0.85, 0.05, num_samples), 0, 1).tolist()
        self.module_metrics['推荐系统']['recall@k'] = np.clip(np.random.normal(0.83, 0.06, num_samples), 0, 1).tolist()
        self.module_metrics['推荐系统']['ndcg'] = np.clip(np.random.normal(0.80, 0.07, num_samples), 0, 1).tolist()
    
    def save_performance_data(self, filepath='performance_data.json'):
        """保存性能数据到JSON文件"""
        # 将datetime对象转换为字符串
        serializable_data = {
            'response_times': self.data['response_times'],
            'resource_usage': {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else ts
                              for ts in self.data['resource_usage']['timestamps']],
                'cpu': self.data['resource_usage']['cpu'],
                'memory': self.data['resource_usage']['memory'],
                'gpu_util': self.data['resource_usage']['gpu_util'],
                'gpu_memory': self.data['resource_usage']['gpu_memory']
            },
            'cache_hit_rates': {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else ts
                              for ts in self.data['cache_hit_rates']['timestamps']],
                '意图识别': self.data['cache_hit_rates']['意图识别'],
                '实体识别': self.data['cache_hit_rates']['实体识别'],
                '知识图谱查询': self.data['cache_hit_rates']['知识图谱查询'],
                '推荐生成': self.data['cache_hit_rates']['推荐生成'],
                '整体缓存': self.data['cache_hit_rates']['整体缓存']
            },
            'success_rates': {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else ts
                              for ts in self.data['success_rates']['timestamps']],
                '意图识别': self.data['success_rates']['意图识别'],
                '实体识别': self.data['success_rates']['实体识别'],
                '知识图谱查询': self.data['success_rates']['知识图谱查询'],
                '推荐生成': self.data['success_rates']['推荐生成']
            },
            'throughput': {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else ts
                              for ts in self.data['throughput']['timestamps']],
                'qps': self.data['throughput']['qps']
            },
            'module_metrics': self.module_metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"性能数据已保存到 {filepath}")
    
    def load_performance_data(self, filepath='performance_data.json'):
        """从JSON文件加载性能数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
            # 转换字符串时间戳回datetime对象
            for key in ['resource_usage', 'cache_hit_rates', 'success_rates', 'throughput']:
                if 'timestamps' in loaded_data[key]:
                    loaded_data[key]['timestamps'] = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') 
                                                    if isinstance(ts, str) else ts
                                                    for ts in loaded_data[key]['timestamps']]
            
            self.data = loaded_data
            
            # 加载模块指标
            if 'module_metrics' in loaded_data:
                self.module_metrics = loaded_data['module_metrics']
                
            logger.info(f"已从 {filepath} 加载性能数据")
        except Exception as e:
            logger.error(f"加载性能数据失败: {str(e)}")

def generate_sample_data():
    """生成示例性能数据"""
    monitor = PerformanceMonitor()
    
    # 模拟响应时间数据
    for module in monitor.data['response_times'].keys():
        if module != '整体响应':
            # 模拟100次请求的响应时间
            times = np.random.normal(
                loc={'意图识别': 50, '实体识别': 120, '知识图谱查询': 150, '推荐生成': 200}
                    .get(module, 100),
                scale={'意图识别': 10, '实体识别': 25, '知识图谱查询': 30, '推荐生成': 50}
                    .get(module, 20),
                size=100
            )
            monitor.data['response_times'][module] = times.tolist()
    
    # 计算整体响应时间（各模块之和，考虑并行性）
    overall_times = np.zeros(100)
    for i in range(100):
        # 假设部分并行处理，整体时间为最大响应时间的1.5倍
        max_time = max(monitor.data['response_times'][module][i] for module in monitor.data['response_times'] if module != '整体响应')
        overall_times[i] = max_time * 1.5
    
    monitor.data['response_times']['整体响应'] = overall_times.tolist()
    
    # 模拟系统资源使用情况
    monitor.monitor_system_resources(interval=1.0, duration=60)
    
    # 模拟其他数据
    monitor.simulate_cache_hit_rates()
    monitor.simulate_success_rates()
    monitor.simulate_throughput()
    monitor.simulate_module_metrics()
    
    # 保存数据
    monitor.save_performance_data()
    
    return monitor

if __name__ == '__main__':
    # 生成样本数据
    monitor = generate_sample_data()
    logger.info("性能数据生成完成") 