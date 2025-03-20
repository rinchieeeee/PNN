import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRMF(nn.Module):
    """
    BPR損失を使用した行列分解モデル
    
    Attributes:
        user_embedding (nn.Embedding): ユーザー埋め込み層
        item_embedding (nn.Embedding): アイテム埋め込み層
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        """
        Args:
            num_users (int): ユーザー数
            num_items (int): アイテム数
            embedding_dim (int): 埋め込みの次元数
        """
        super(BPRMF, self).__init__()
        
        # ユーザーとアイテムの埋め込み層
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 埋め込みの初期化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_id, item_id=None):
        """
        ユーザーとアイテムの埋め込みの内積を計算
        
        Args:
            user_id (torch.Tensor): ユーザーIDのテンソル
            item_id (torch.Tensor, optional): アイテムIDのテンソル。Noneの場合は全アイテムのスコアを計算。
            
        Returns:
            torch.Tensor: 予測スコア
        """
        if item_id is None:
            # 全アイテムのスコアを計算
            users = self.user_embedding(user_id)
            items = self.item_embedding.weight
            return torch.matmul(users, items.t())
        
        # 特定のユーザーとアイテムのスコアを計算
        users = self.user_embedding(user_id)
        items = self.item_embedding(item_id)
        return (users * items).sum(dim=1)
    
    def bpr_loss(self, user_id, pos_item_id, neg_item_id):
        """
        BPR損失を計算 (式8)
        L_BPR = −ln[σ(s(e_u, e_i) - s(e_u, e_j))]
        
        Args:
            user_id (torch.Tensor): ユーザーIDのテンソル
            pos_item_id (torch.Tensor): ポジティブアイテムIDのテンソル
            neg_item_id (torch.Tensor): ネガティブアイテムIDのテンソル
            
        Returns:
            torch.Tensor: BPR損失
        """
        # ポジティブアイテムのスコア
        pos_score = self.forward(user_id, pos_item_id)
        
        # ネガティブアイテムのスコア
        neg_score = self.forward(user_id, neg_item_id)
        
        # BPR損失
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10)
        
        return loss.mean()
