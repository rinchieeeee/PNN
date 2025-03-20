import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.evaluation import evaluate
import os

class BaseTrainer:
    """
    すべてのトレーナーの基底クラス
    """
    def __init__(self, model, optimizer, device):
        """
        Args:
            model (nn.Module): 学習するモデル
            optimizer (torch.optim.Optimizer): オプティマイザ
            device (torch.device): 使用するデバイス
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_metrics = {
            'recall': [],
            'hit': [],
            'ndcg': []
        }
        self.best_val_recall = 0
        self.best_epoch = 0
        self.best_model_path = None
    
    def train_epoch(self, train_loader, epoch):
        """
        1エポックの学習を実行
        
        Args:
            train_loader (DataLoader): 学習データのローダー
            epoch (int): 現在のエポック数
            
        Returns:
            float: 平均損失
        """
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def validate(self, val_interactions, num_items, k=10):
        """
        検証データセットでモデルを評価
        
        Args:
            val_interactions (dict): 検証用ユーザーインタラクション
            num_items (int): アイテム数
            k (int): 推薦アイテム数
            
        Returns:
            tuple: (recall, hit, ndcg)
        """
        return evaluate(self.model, val_interactions, num_items, self.device, k)
    
    def train(self, train_loader, val_interactions, test_interactions, num_items, epochs, model_name):
        """
        モデルの学習と評価を実行
        
        Args:
            train_loader (DataLoader): 学習データのローダー
            val_interactions (dict): 検証用ユーザーインタラクション
            test_interactions (dict): テスト用ユーザーインタラクション
            num_items (int): アイテム数
            epochs (int): エポック数
            model_name (str): モデル名
            
        Returns:
            dict: 学習結果
        """
        self.best_model_path = f'best_{model_name}_model.pt'
        
        for epoch in range(1, epochs + 1):
            # モデルの学習
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 検証セットでの評価
            val_recall, val_hit, val_ndcg = self.validate(val_interactions, num_items)
            self.val_metrics['recall'].append(val_recall)
            self.val_metrics['hit'].append(val_hit)
            self.val_metrics['ndcg'].append(val_ndcg)
            
            print(f'Validation: Recall@10: {val_recall:.4f}, Hit@10: {val_hit:.4f}, NDCG@10: {val_ndcg:.4f}')
            
            # 最良のモデルを保存
            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.best_model_path)
        
        # 最良のモデルを読み込んでテストセットで評価
        self.model.load_state_dict(torch.load(self.best_model_path))
        test_recall, test_hit, test_ndcg = evaluate(self.model, test_interactions, num_items, self.device)
        
        print(f'\nBest model at epoch {self.best_epoch}:')
        print(f'Test: Recall@10: {test_recall:.4f}, Hit@10: {test_hit:.4f}, NDCG@10: {test_ndcg:.4f}')
        
        # 学習曲線をプロット
        self.plot_learning_curves(model_name, epochs)
        
        return {
            'model': self.model,
            'best_epoch': self.best_epoch,
            'test_recall': test_recall,
            'test_hit': test_hit,
            'test_ndcg': test_ndcg,
            'train_losses': self.train_losses,
            'val_recalls': self.val_metrics['recall'],
            'val_hits': self.val_metrics['hit'],
            'val_ndcgs': self.val_metrics['ndcg']
        }
    
    def plot_learning_curves(self, model_name, epochs):
        """
        学習曲線をプロット
        
        Args:
            model_name (str): モデル名
            epochs (int): エポック数
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), self.train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), self.val_metrics['recall'], label='Recall@10')
        plt.plot(range(1, epochs + 1), self.val_metrics['hit'], label='Hit@10')
        plt.plot(range(1, epochs + 1), self.val_metrics['ndcg'], label='NDCG@10')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存先ディレクトリの確認
        if not os.path.exists('results'):
            os.makedirs('results')
            
        plt.savefig(f'results/{model_name}_learning_curves.png')
