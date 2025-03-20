import torch
import random
from tqdm import tqdm
from src.trainers.base_trainer import BaseTrainer
import matplotlib.pyplot as plt

class PNNTrainer(BaseTrainer):
    """
    PNN-BPRMFモデルのトレーナー
    """
    def __init__(self, model, optimizer, device):
        """
        Args:
            model (PNNBPRMF): PNN-BPRMFモデル
            optimizer (torch.optim.Optimizer): オプティマイザ
            device (torch.device): 使用するデバイス
        """
        super(PNNTrainer, self).__init__(model, optimizer, device)
        self.pnn_metrics = {
            'bpr_loss': [],
            'pnn_loss': [],
            'constrain_loss': [],
            'uniform_loss': [],
            'rank_loss': []
        }
    
    def train_epoch(self, train_loader, epoch, full_user_interactions, num_items, unlabeled_size=32):
        """
        1エポックの学習を実行
        
        Args:
            train_loader (DataLoader): 学習データのローダー
            epoch (int): 現在のエポック数
            full_user_interactions (dict): 全ユーザーのポジティブインタラクション
            num_items (int): アイテム数
            unlabeled_size (int): バッチごとに使用するラベルなしデータの数
            
        Returns:
            float: 平均損失
        """
        self.model.train()
        total_loss = 0
        total_bpr_loss = 0
        total_pnn_loss = 0
        total_constrain_loss = 0
        total_uniform_loss = 0
        total_rank_loss = 0
        
        for batch_idx, (user_id, pos_item_id, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            user_id = user_id.to(self.device)
            pos_item_id = pos_item_id.unsqueeze(1).to(self.device)  # バッチ x 1 の形状に変更
            
            # 各ユーザーのポジティブアイテムをすべて取得
            batch_pos_items = []
            for u in user_id.cpu().numpy():
                user_pos_items = full_user_interactions[u]
                batch_pos_items.append(user_pos_items)
            
            # ラベルなしデータを生成
            batch_unlabeled_items = []
            for u in user_id.cpu().numpy():
                user_pos_items = set(full_user_interactions[u])
                unlabeled = []
                while len(unlabeled) < unlabeled_size:
                    item = random.randint(0, num_items - 1)
                    if item not in user_pos_items and item not in unlabeled:
                        unlabeled.append(item)
                batch_unlabeled_items.append(unlabeled)
            
            # テンソルに変換
            full_pos_items = [torch.tensor(items, dtype=torch.long) for items in batch_pos_items]
            max_pos_len = max(len(items) for items in full_pos_items)
            padded_pos_items = torch.zeros(len(full_pos_items), max_pos_len, dtype=torch.long)
            for i, items in enumerate(full_pos_items):
                padded_pos_items[i, :len(items)] = items
            padded_pos_items = padded_pos_items.to(self.device)
            
            unlabeled_items = torch.tensor(batch_unlabeled_items, dtype=torch.long).to(self.device)
            
            # 勾配をリセット
            self.optimizer.zero_grad()
            
            # BPR損失を計算（バッチ内のアイテムから負サンプルを選択）
            neg_item_id = []
            for u, p in zip(user_id.cpu().numpy(), pos_item_id.cpu().numpy()):
                user_pos_items = set(full_user_interactions[u])
                # バッチ内のハードネガティブサンプリング（式9）
                max_score = -float('inf')
                best_neg = None
                for _ in range(10):  # 10個のサンプルから最適なものを選択
                    candidate = random.randint(0, num_items - 1)
                    if candidate not in user_pos_items:
                        with torch.no_grad():
                            score = self.model.forward(torch.tensor([u]).to(self.device), 
                                                      torch.tensor([candidate]).to(self.device)).item()
                        if score > max_score:
                            max_score = score
                            best_neg = candidate
                neg_item_id.append(best_neg if best_neg is not None else random.randint(0, num_items - 1))
            
            neg_item_id = torch.tensor(neg_item_id).to(self.device)
            bpr_loss = self.model.bpr_loss(user_id, pos_item_id.squeeze(), neg_item_id)
            
            # ユーザーアウェアアテンションによるλを計算（式10, 11）
            lambda_val = self.model.user_aware_attention(user_id, padded_pos_items)
            
            # PNN損失を計算
            pnn_loss, constrain_loss, uniform_loss, rank_loss = self.model.pnn_loss(
                user_id, padded_pos_items, unlabeled_items, lambda_val
            )
            
            # 最終的な損失を計算（式7）
            loss = (1 - lambda_val) * bpr_loss + lambda_val * pnn_loss
            loss = loss.mean()
            
            # バックプロパゲーション
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_pnn_loss += pnn_loss.item()
            total_constrain_loss += constrain_loss.item()
            total_uniform_loss += uniform_loss.item()
            total_rank_loss += rank_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_bpr_loss = total_bpr_loss / len(train_loader)
        avg_pnn_loss = total_pnn_loss / len(train_loader)
        avg_constrain_loss = total_constrain_loss / len(train_loader)
        avg_uniform_loss = total_uniform_loss / len(train_loader)
        avg_rank_loss = total_rank_loss / len(train_loader)
        
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.6f}, BPR Loss: {avg_bpr_loss:.6f}, PNN Loss: {avg_pnn_loss:.6f}')
        print(f'         Constrain Loss: {avg_constrain_loss:.6f}, Uniform Loss: {avg_uniform_loss:.6f}, Rank Loss: {avg_rank_loss:.6f}')
        
        # 各損失をリストに追加
        self.pnn_metrics['bpr_loss'].append(avg_bpr_loss)
        self.pnn_metrics['pnn_loss'].append(avg_pnn_loss)
        self.pnn_metrics['constrain_loss'].append(avg_constrain_loss)
        self.pnn_metrics['uniform_loss'].append(avg_uniform_loss)
        self.pnn_metrics['rank_loss'].append(avg_rank_loss)
        
        return avg_loss
    
    def train(self, train_loader, val_interactions, test_interactions, num_items, epochs, model_name, 
              full_user_interactions, unlabeled_size=32):
        """
        モデルの学習と評価を実行（PNN固有の引数を追加）
        
        Args:
            train_loader (DataLoader): 学習データのローダー
            val_interactions (dict): 検証用ユーザーインタラクション
            test_interactions (dict): テスト用ユーザーインタラクション
            num_items (int): アイテム数
            epochs (int): エポック数
            model_name (str): モデル名
            full_user_interactions (dict): 全ユーザーのポジティブインタラクション
            unlabeled_size (int): バッチごとに使用するラベルなしデータの数
            
        Returns:
            dict: 学習結果
        """
        self.best_model_path = f'best_{model_name}_model.pt'
        
        for epoch in range(1, epochs + 1):
            # モデルの学習
            train_loss = self.train_epoch(
                train_loader, epoch, full_user_interactions, num_items, unlabeled_size
            )
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
        test_recall, test_hit, test_ndcg = self.validate(test_interactions, num_items)
        
        print(f'\nBest model at epoch {self.best_epoch}:')
        print(f'Test: Recall@10: {test_recall:.4f}, Hit@10: {test_hit:.4f}, NDCG@10: {test_ndcg:.4f}')
        
        # 学習曲線をプロット
        self.plot_learning_curves(model_name, epochs)
        self.plot_pnn_metrics(model_name, epochs)
        
        return {
            'model': self.model,
            'best_epoch': self.best_epoch,
            'test_recall': test_recall,
            'test_hit': test_hit,
            'test_ndcg': test_ndcg,
            'train_losses': self.train_losses,
            'val_recalls': self.val_metrics['recall'],
            'val_hits': self.val_metrics['hit'],
            'val_ndcgs': self.val_metrics['ndcg'],
            'pnn_metrics': self.pnn_metrics
        }
    
    def plot_pnn_metrics(self, model_name, epochs):
        """
        PNN特有のメトリクスをプロット
        
        Args:
            model_name (str): モデル名
            epochs (int): エポック数
        """
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(range(1, epochs + 1), self.pnn_metrics['bpr_loss'], label='BPR Loss')
        plt.plot(range(1, epochs + 1), self.pnn_metrics['pnn_loss'], label='PNN Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('BPR vs PNN Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(range(1, epochs + 1), self.pnn_metrics['constrain_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Constrain Loss')
        
        plt.subplot(2, 2, 3)
        plt.plot(range(1, epochs + 1), self.pnn_metrics['uniform_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Uniform Loss')
        
        plt.subplot(2, 2, 4)
        plt.plot(range(1, epochs + 1), self.pnn_metrics['rank_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Ranking Loss')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_pnn_metrics.png')