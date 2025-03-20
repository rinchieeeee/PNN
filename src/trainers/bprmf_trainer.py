import torch
from tqdm import tqdm
from src.trainers.base_trainer import BaseTrainer

class BPRMFTrainer(BaseTrainer):
    """
    BPR-MFモデルのトレーナー
    """
    def __init__(self, model, optimizer, device):
        """
        Args:
            model (BPRMF): BPR-MFモデル
            optimizer (torch.optim.Optimizer): オプティマイザ
            device (torch.device): 使用するデバイス
        """
        super(BPRMFTrainer, self).__init__(model, optimizer, device)
    
    def train_epoch(self, train_loader, epoch):
        """
        1エポックの学習を実行
        
        Args:
            train_loader (DataLoader): 学習データのローダー
            epoch (int): 現在のエポック数
            
        Returns:
            float: 平均損失
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (user_id, pos_item_id, neg_items) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            user_id = user_id.to(self.device)
            pos_item_id = pos_item_id.to(self.device)
            
            # neg_itemsの形状を確認して適切に処理
            # DataLoaderの戻り値のneg_itemsは形状が[batch_size, neg_sample_size]であるはず
            if isinstance(neg_items, list):
                # リストの場合はテンソルに変換
                neg_item_id = torch.tensor(neg_items, dtype=torch.long).to(self.device)
            else:
                # すでにテンソルの場合はそのままデバイスに移動
                neg_item_id = neg_items.to(self.device)
            
            # ネガティブサンプルが複数ある場合は最初のものだけを使用
            if neg_item_id.dim() > 1:
                neg_item_id = neg_item_id[:, 0]
            
            # 勾配をリセット
            self.optimizer.zero_grad()
            
            # 損失を計算
            loss = self.model.bpr_loss(user_id, pos_item_id, neg_item_id)
            
            # バックプロパゲーション
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.6f}')
        return avg_loss
