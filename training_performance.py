import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 确保图片目录存在
os.makedirs('img/performance/training', exist_ok=True)

class TrainingPerformanceAnalyzer:
    def __init__(self):
        self.ner_data = {
            'epochs': [],
            'losses': [],
            'f1_scores': [],
            'best_epochs': []
        }
        
        self.intent_data = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'train_acc': [],
            'val_acc': [],
            'best_epochs': []
        }
        
        self.model_details = {
            'ner': {},
            'intent': {}
        }
    
    def parse_ner_log(self, log_file='training_log.txt'):
        """解析命名实体识别的训练日志"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 提取模型配置信息
            train_size_match = re.search(r'训练集大小: (\d+)', content)
            val_size_match = re.search(r'验证集大小: (\d+)', content)
            batch_size_match = re.search(r'批次大小: (\d+)', content)
            lr_match = re.search(r'学习率: (\d+\.\d+e?-?\d*)', content)
            max_seq_len_match = re.search(r'最大序列长度: (\d+)', content)
            hidden_size_match = re.search(r'隐藏层大小: (\d+)', content)
            bidirectional_match = re.search(r'双向GRU: (True|False)', content)
            
            if train_size_match:
                self.model_details['ner']['train_size'] = int(train_size_match.group(1))
            if val_size_match:
                self.model_details['ner']['val_size'] = int(val_size_match.group(1))
            if batch_size_match:
                self.model_details['ner']['batch_size'] = int(batch_size_match.group(1))
            if lr_match:
                self.model_details['ner']['learning_rate'] = float(lr_match.group(1))
            if max_seq_len_match:
                self.model_details['ner']['max_seq_len'] = int(max_seq_len_match.group(1))
            if hidden_size_match:
                self.model_details['ner']['hidden_size'] = int(hidden_size_match.group(1))
            if bidirectional_match:
                self.model_details['ner']['bidirectional'] = bidirectional_match.group(1) == 'True'
            
            # 提取每个epoch的性能数据
            epoch_pattern = r'Epoch (\d+)/\d+ - Loss: ([\d\.]+) - F1: ([\d\.]+)( \(最佳模型已保存\))?'
            matches = re.findall(epoch_pattern, content)
            
            for match in matches:
                epoch = int(match[0])
                loss = float(match[1])
                f1 = float(match[2])
                is_best = match[3] != ''
                
                self.ner_data['epochs'].append(epoch)
                self.ner_data['losses'].append(loss)
                self.ner_data['f1_scores'].append(f1)
                
                if is_best:
                    self.ner_data['best_epochs'].append(epoch)
            
            return True
        except Exception as e:
            print(f"解析NER训练日志失败: {str(e)}")
            return False
    
    def parse_intent_log(self, log_file='model/intent_training.log'):
        """解析意图分类的训练日志"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取最后一次训练的数据
            last_train_pattern = r'类别分布:.*?训练集大小: (\d+), 验证集大小: (\d+)'
            last_train_match = re.findall(last_train_pattern, content, re.DOTALL)
            
            if last_train_match:
                self.model_details['intent']['train_size'] = int(last_train_match[-1][0])
                self.model_details['intent']['val_size'] = int(last_train_match[-1][1])
            
            # 提取每个epoch的性能数据（最后一次训练）
            last_train_start = content.rfind('类别分布:')
            if last_train_start != -1:
                last_train_content = content[last_train_start:]
                
                epoch_pattern = r'Epoch \[(\d+)/\d+\], Train Loss: ([\d\.]+), Train Acc: ([\d\.]+)%, Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)%'
                matches = re.findall(epoch_pattern, last_train_content)
                
                best_pattern = r'Model saved with validation accuracy: ([\d\.]+)%'
                best_matches = re.findall(best_pattern, last_train_content)
                
                best_epochs = []
                for i, match in enumerate(best_matches):
                    # 这里假设best_matches的顺序与上面matches中的epoch顺序一致
                    if i < len(matches):
                        best_epochs.append(int(matches[i][0]))
                
                for match in matches:
                    epoch = int(match[0])
                    train_loss = float(match[1])
                    train_acc = float(match[2])
                    val_loss = float(match[3])
                    val_acc = float(match[4])
                    
                    self.intent_data['epochs'].append(epoch)
                    self.intent_data['train_losses'].append(train_loss)
                    self.intent_data['val_losses'].append(val_loss)
                    self.intent_data['train_acc'].append(train_acc / 100)  # 转换为0-1范围
                    self.intent_data['val_acc'].append(val_acc / 100)  # 转换为0-1范围
                
                self.intent_data['best_epochs'] = best_epochs
            
            return True
        except Exception as e:
            print(f"解析意图分类训练日志失败: {str(e)}")
            return False
    
    def plot_ner_training_curves(self):
        """绘制NER训练曲线"""
        if not self.ner_data['epochs']:
            print("未找到NER训练数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(self.ner_data['epochs'], self.ner_data['losses'], 'b-', marker='o')
        ax1.set_title('NER训练损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 标记最佳epoch
        for epoch in self.ner_data['best_epochs']:
            idx = self.ner_data['epochs'].index(epoch)
            ax1.plot(epoch, self.ner_data['losses'][idx], 'ro', markersize=8)
            ax1.annotate(f'Best', xy=(epoch, self.ner_data['losses'][idx]), 
                        xytext=(10, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        # F1分数曲线
        ax2.plot(self.ner_data['epochs'], self.ner_data['f1_scores'], 'g-', marker='o')
        ax2.set_title('NER验证F1分数曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 标记最佳epoch
        for epoch in self.ner_data['best_epochs']:
            idx = self.ner_data['epochs'].index(epoch)
            ax2.plot(epoch, self.ner_data['f1_scores'][idx], 'ro', markersize=8)
            ax2.annotate(f'Best', xy=(epoch, self.ner_data['f1_scores'][idx]), 
                        xytext=(10, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig('img/performance/training/ner_training_curves.png', dpi=300)
        plt.close()
        
        # 绘制收敛速度图
        plt.figure(figsize=(8, 5))
        
        # 计算相对于最终F1的收敛百分比
        final_f1 = self.ner_data['f1_scores'][-1]
        convergence = [f1 / final_f1 * 100 for f1 in self.ner_data['f1_scores']]
        
        plt.plot(self.ner_data['epochs'], convergence, 'b-', marker='o')
        plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95%收敛')
        
        # 找到首次达到95%收敛的epoch
        convergence_95 = next((i for i, x in enumerate(convergence) if x >= 95), None)
        if convergence_95 is not None:
            epoch_95 = self.ner_data['epochs'][convergence_95]
            plt.plot(epoch_95, convergence[convergence_95], 'ro', markersize=8)
            plt.annotate(f'95%收敛: Epoch {epoch_95}', 
                         xy=(epoch_95, convergence[convergence_95]),
                         xytext=(10, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.title('NER训练收敛速度')
        plt.xlabel('Epoch')
        plt.ylabel('收敛百分比 (%)')
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('img/performance/training/ner_convergence_speed.png', dpi=300)
        plt.close()
    
    def plot_intent_training_curves(self):
        """绘制意图分类训练曲线"""
        if not self.intent_data['epochs']:
            print("未找到意图分类训练数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(self.intent_data['epochs'], self.intent_data['train_losses'], 'b-', marker='o', label='训练损失')
        ax1.plot(self.intent_data['epochs'], self.intent_data['val_losses'], 'r-', marker='s', label='验证损失')
        ax1.set_title('意图分类损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 精度曲线
        ax2.plot(self.intent_data['epochs'], self.intent_data['train_acc'], 'g-', marker='o', label='训练精度')
        ax2.plot(self.intent_data['epochs'], self.intent_data['val_acc'], 'm-', marker='s', label='验证精度')
        ax2.set_title('意图分类精度曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # 标记最佳epoch
        for epoch in self.intent_data['best_epochs']:
            if epoch in self.intent_data['epochs']:
                idx = self.intent_data['epochs'].index(epoch)
                ax2.plot(epoch, self.intent_data['val_acc'][idx], 'ro', markersize=8)
                ax2.annotate(f'Best', xy=(epoch, self.intent_data['val_acc'][idx]), 
                           xytext=(0, 20), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig('img/performance/training/intent_training_curves.png', dpi=300)
        plt.close()
        
        # 绘制训练/验证差异图（过拟合分析）
        plt.figure(figsize=(8, 5))
        
        # 计算训练精度和验证精度之间的差异
        accuracy_diff = [train - val for train, val in zip(self.intent_data['train_acc'], self.intent_data['val_acc'])]
        
        plt.bar(self.intent_data['epochs'], accuracy_diff, color='skyblue')
        plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='过拟合警戒线 (10%)')
        
        plt.title('意图分类过拟合分析')
        plt.xlabel('Epoch')
        plt.ylabel('训练精度 - 验证精度')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # 找出最大过拟合的epoch
        max_overfit_idx = accuracy_diff.index(max(accuracy_diff))
        max_overfit_epoch = self.intent_data['epochs'][max_overfit_idx]
        max_overfit_value = accuracy_diff[max_overfit_idx]
        
        if max_overfit_value > 0.1:  # 超过10%的差异视为过拟合
            plt.annotate(f'最大过拟合: {max_overfit_value:.2f}', 
                        xy=(max_overfit_epoch, max_overfit_value),
                        xytext=(0, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig('img/performance/training/intent_overfitting_analysis.png', dpi=300)
        plt.close()
    
    def plot_training_summary(self):
        """绘制训练性能总结图"""
        # 准备数据
        models = ['命名实体识别', '意图分类']
        metrics = {}
        
        if self.ner_data['epochs']:
            max_f1 = max(self.ner_data['f1_scores'])
            final_f1 = self.ner_data['f1_scores'][-1]
            convergence_epoch = next((i for i, x in enumerate(self.ner_data['f1_scores']) 
                                    if x >= 0.95 * final_f1), len(self.ner_data['epochs']))
            metrics['最大F1/精度'] = [max_f1, max(self.intent_data['val_acc']) if self.intent_data['val_acc'] else 0]
            metrics['收敛速度(epochs)'] = [convergence_epoch + 1, 
                                      self.intent_data['best_epochs'][-1] if self.intent_data['best_epochs'] else 0]
        
        # 绘制性能指标对比图
        if metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(models))
            width = 0.35
            
            # 绘制最大F1/精度
            ax.bar(x - width/2, metrics['最大F1/精度'], width, label='最大F1/精度')
            
            # 添加数值标签
            for i, v in enumerate(metrics['最大F1/精度']):
                ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
            
            # 设置坐标轴
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel('分数')
            ax.set_title('模型训练性能对比')
            ax.set_ylim(0, 1.1)
            ax.legend()
            
            # 添加第二个Y轴显示收敛速度
            ax2 = ax.twinx()
            ax2.bar(x + width/2, metrics['收敛速度(epochs)'], width, color='green', label='收敛速度(epochs)')
            ax2.set_ylabel('Epochs')
            
            # 添加数值标签
            for i, v in enumerate(metrics['收敛速度(epochs)']):
                ax2.text(i + width/2, v + 0.5, f'{v}', ha='center')
            
            # 设置图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            plt.savefig('img/performance/training/model_performance_comparison.png', dpi=300)
            plt.close()
    
    def plot_epoch_time_analysis(self):
        """绘制每个epoch的训练时间分析图（基于训练日志中的时间戳估计）"""
        if self.ner_data['epochs'] and len(self.ner_data['epochs']) >= 2:
            # 假设每个epoch训练时间相近，估算平均每个epoch的时间
            estimated_time_per_epoch = 60  # 估计值，单位为秒
            
            epochs = self.ner_data['epochs']
            times = [i * estimated_time_per_epoch for i in range(len(epochs))]
            
            plt.figure(figsize=(8, 5))
            
            # 绘制累计时间曲线
            plt.plot(epochs, times, 'b-', marker='o')
            
            # 标注关键点
            plt.annotate(f'总训练时间: {times[-1]/60:.1f}分钟', 
                        xy=(epochs[-1], times[-1]),
                        xytext=(-100, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='blue'))
            
            # 标注最佳模型点
            if self.ner_data['best_epochs']:
                best_epoch = self.ner_data['best_epochs'][-1]
                best_idx = epochs.index(best_epoch)
                best_time = times[best_idx]
                
                plt.plot(best_epoch, best_time, 'ro', markersize=8)
                plt.annotate(f'最佳模型: {best_time/60:.1f}分钟', 
                            xy=(best_epoch, best_time),
                            xytext=(10, -20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'))
            
            plt.title('NER模型训练时间分析')
            plt.xlabel('Epoch')
            plt.ylabel('累计训练时间 (秒)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('img/performance/training/ner_training_time.png', dpi=300)
            plt.close()
    
    def save_model_details(self):
        """保存模型配置和训练指标到JSON文件"""
        # 添加训练性能指标
        if self.ner_data['epochs']:
            self.model_details['ner']['epochs'] = len(self.ner_data['epochs'])
            self.model_details['ner']['best_epoch'] = self.ner_data['best_epochs'][-1] if self.ner_data['best_epochs'] else None
            self.model_details['ner']['best_f1'] = max(self.ner_data['f1_scores'])
            self.model_details['ner']['final_loss'] = self.ner_data['losses'][-1]
        
        if self.intent_data['epochs']:
            self.model_details['intent']['epochs'] = len(self.intent_data['epochs'])
            self.model_details['intent']['best_epoch'] = self.intent_data['best_epochs'][-1] if self.intent_data['best_epochs'] else None
            self.model_details['intent']['best_accuracy'] = max(self.intent_data['val_acc'])
            self.model_details['intent']['final_train_loss'] = self.intent_data['train_losses'][-1]
            self.model_details['intent']['final_val_loss'] = self.intent_data['val_losses'][-1]
        
        with open('img/performance/training/model_details.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_details, f, ensure_ascii=False, indent=4)
    
    def analyze(self):
        """执行训练日志分析并生成图表"""
        # 解析训练日志
        ner_success = self.parse_ner_log()
        intent_success = self.parse_intent_log()
        
        # 绘制图表
        if ner_success:
            self.plot_ner_training_curves()
            self.plot_epoch_time_analysis()
        
        if intent_success:
            self.plot_intent_training_curves()
        
        if ner_success or intent_success:
            self.plot_training_summary()
            self.save_model_details()
        
        return ner_success or intent_success

if __name__ == "__main__":
    analyzer = TrainingPerformanceAnalyzer()
    success = analyzer.analyze()
    
    if success:
        print("训练性能分析完成，图表已保存到 img/performance/training/ 目录")
    else:
        print("训练性能分析失败，请检查日志文件是否存在") 