import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bprmf import BPRMF

class PNNBPRMF(BPRMF):
    """
    PNN損失を使用した行列分解モデル
    BPRMF拡張して、ポジティブ-中立-ネガティブ学習を実装
    
    Attributes:
        user_embedding (nn.Embedding): ユーザー埋め込み層
        item_embedding (nn.Embedding): アイテム埋め込み層
        W1 (nn.Parameter): アテンションモデル用のパラメータ
        W2 (nn.Parameter): アテンションモデル用のパラメータ
        b (nn.Parameter): アテンションモデル用のバイアス
        alpha (float): L_constrainの重み
        beta (float): L_uniformの重み
    """
    def __init__(self, num_users, num_items, embedding_dim=64, alpha=0.5, beta=0.5):
        """
        Args:
            num_users (int): ユーザー数
            num_items (int): アイテム数
            embedding_dim (int): 埋め込みの次元数
            alpha (float): L_constrainの重み
            beta (float): L_uniformの重み
        """
        super(PNNBPRMF, self).__init__(num_users, num_items, embedding_dim)
        
        # アテンションモデル用のパラメータ (式10, 11)
        self.W1 = nn.Parameter(torch.Tensor(1, embedding_dim))
        self.W2 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.b = nn.Parameter(torch.Tensor(embedding_dim, 1))
        
        # パラメータの初期化
        nn.init.normal_(self.W1, std=0.01)
        nn.init.normal_(self.W2, std=0.01)
        nn.init.zeros_(self.b)
        
        self.alpha = alpha  # L_constrainの重み (式17)
        self.beta = beta    # L_uniformの重み (式17)
    
    def user_aware_attention(self, user_id, pos_item_ids):
        """
        ユーザーアウェアアテンションモデル (式10, 11)
        λ = σ[W1 (∑i∈Σu_pos α^attr_i e_i)]
        
        Args:
            user_id (torch.Tensor): ユーザーIDのテンソル [batch_size]
            pos_item_ids (torch.Tensor): ポジティブアイテムIDのテンソル [batch_size, num_pos_items]
            
        Returns:
            torch.Tensor: λ値 [batch_size]
        """
        user_emb = self.user_embedding(user_id)
        pos_item_embs = self.item_embedding(pos_item_ids)
        
        # アテンションスコアを計算 (式11)
        beta_attr = torch.tanh(torch.matmul(pos_item_embs, self.W2) + self.b.transpose(0, 1))
        beta_attr = torch.matmul(user_emb.unsqueeze(1), beta_attr.transpose(1, 2)).squeeze(1)
        
        # マスクを適用（パディングアイテムを無視）
        mask = (pos_item_ids != 0).float()
        beta_attr = beta_attr * mask
        
        # Softmaxでアテンションの重みを正規化
        alpha_attr = F.softmax(beta_attr, dim=1)
        
        # 重み付き和でアイテム埋め込みを集約
        weighted_sum = torch.sum(alpha_attr.unsqueeze(2) * pos_item_embs, dim=1)
        
        # λ値を計算 (式10)
        lambda_val = torch.sigmoid(torch.matmul(self.W1, weighted_sum.transpose(0, 1))).squeeze(0)
        
        return lambda_val
    
    def classify_unlabeled_data(self, user_id, unlabeled_items, num_neg):
        """
        ラベルなしデータを中立クラスと負クラスに分類 (式12)
        
        Args:
            user_id (torch.Tensor): ユーザーIDのテンソル [batch_size]
            unlabeled_items (torch.Tensor): ラベルなしアイテムIDのテンソル [batch_size, num_unlabeled]
            num_neg (int): 負クラスとして選択するアイテム数
            
        Returns:
            tuple: (neu_indices, neg_indices) - 中立クラスと負クラスのアイテムインデックス
        """
        with torch.no_grad():
            # すべてのラベルなしアイテムのスコアを計算
            user_emb = self.user_embedding(user_id)
            item_embs = self.item_embedding(unlabeled_items)
            scores = torch.sum(user_emb.unsqueeze(1) * item_embs, dim=2)
            
            # スコアの低い順にアイテムを並べ替え
            _, indices = torch.sort(scores, dim=1)
            
            # 上位num_neg個のアイテムを負クラスとして選択
            neg_indices = torch.gather(unlabeled_items, 1, indices[:, :num_neg])
            
            # 残りのアイテムを中立クラスとして選択
            neu_indices = torch.gather(unlabeled_items, 1, indices[:, num_neg:])
            
        return neu_indices, neg_indices
    
    def uniform_loss(self, unlabeled_items):
        """
        一様性損失を計算 (式13)
        L_uniform = log E_{i,i'∈E_u} e^{-2||e_i - e_i'||^2}
        
        Args:
            unlabeled_items (torch.Tensor): ラベルなしアイテムIDのテンソル [batch_size, num_unlabeled]
            
        Returns:
            torch.Tensor: 一様性損失
        """
        item_embs = self.item_embedding(unlabeled_items)
        
        # バッチ内の各ペアの距離を計算
        x = item_embs.unsqueeze(2)
        y = item_embs.unsqueeze(1)
        
        # ユークリッド距離の2乗を計算
        dist = torch.sum((x - y) ** 2, dim=3)
        
        # マスク: 対角要素を除外
        mask = torch.eye(dist.size(1), device=dist.device).unsqueeze(0).expand_as(dist)
        
        # 損失を計算
        loss = torch.log(torch.sum(torch.exp(-2.0 * dist) * (1 - mask)) / (dist.size(1) * (dist.size(1) - 1)))
        
        return loss.mean()
    
    def centroid_ranking(self, user_id, pos_items, neu_items, neg_items):
        """
        セントロイドランキングアプローチ (式14-16)
        L_rank = -ln[σ(s(e_u, e_i+) - s(e_u, e_i_neu))] - ln[σ(s(e_u, e_i_neu) - s(e_u, e_i-))]
        
        Args:
            user_id (torch.Tensor): ユーザーIDのテンソル [batch_size]
            pos_items (torch.Tensor): ポジティブアイテムIDのテンソル [batch_size, num_pos]
            neu_items (torch.Tensor): 中立アイテムIDのテンソル [batch_size, num_neu]
            neg_items (torch.Tensor): 負アイテムIDのテンソル [batch_size, num_neg]
            
        Returns:
            tuple: (L_constrain, L_rank) - 拘束損失とランキング損失
        """
        user_emb = self.user_embedding(user_id)
        
        # 各クラスのセントロイドを計算
        # ポジティブアイテムのマスク（パディングを考慮）
        pos_mask = (pos_items != 0).float()
        pos_mask_sum = pos_mask.sum(dim=1, keepdim=True)
        pos_mask_sum = torch.clamp(pos_mask_sum, min=1.0)  # ゼロ除算を防止
        
        # 中立アイテムのマスク
        neu_mask = (neu_items != 0).float()
        neu_mask_sum = neu_mask.sum(dim=1, keepdim=True)
        neu_mask_sum = torch.clamp(neu_mask_sum, min=1.0)  # ゼロ除算を防止
        
        # 負アイテムのマスク
        neg_mask = (neg_items != 0).float()
        neg_mask_sum = neg_mask.sum(dim=1, keepdim=True)
        neg_mask_sum = torch.clamp(neg_mask_sum, min=1.0)  # ゼロ除算を防止
        
        # 埋め込みを取得
        pos_embs = self.item_embedding(pos_items)
        neu_embs = self.item_embedding(neu_items)
        neg_embs = self.item_embedding(neg_items)
        
        # セントロイドの計算（マスク適用）
        pos_centroid = torch.sum(pos_embs * pos_mask.unsqueeze(-1), dim=1) / pos_mask_sum
        neu_centroid = torch.sum(neu_embs * neu_mask.unsqueeze(-1), dim=1) / neu_mask_sum
        neg_centroid = torch.sum(neg_embs * neg_mask.unsqueeze(-1), dim=1) / neg_mask_sum
        
        # クランプ埋め込みを生成 (式15)
        # ポジティブ方向のクランプ埋め込み
        delta_pos = torch.rand(user_emb.size(0), user_emb.size(1), device=user_emb.device) * 0.1
        sign_pos = (pos_centroid - neu_centroid) / (torch.norm(pos_centroid - neu_centroid, dim=1, keepdim=True) + 1e-10)
        pos_clamp = neu_centroid + delta_pos * sign_pos
        
        # ネガティブ方向のクランプ埋め込み
        delta_neg = torch.rand(user_emb.size(0), user_emb.size(1), device=user_emb.device) * 0.1
        sign_neg = (neg_centroid - neu_centroid) / (torch.norm(neg_centroid - neu_centroid, dim=1, keepdim=True) + 1e-10)
        neg_clamp = neu_centroid + delta_neg * sign_neg
        
        # クランプ拘束損失 (式16)
        L_constrain = torch.norm(pos_clamp - neg_clamp, dim=1).mean()
        
        # ランキング損失 (式14)
        pos_score = torch.sum(user_emb * pos_centroid, dim=1)
        neu_score = torch.sum(user_emb * neu_centroid, dim=1)
        neg_score = torch.sum(user_emb * neg_centroid, dim=1)
        
        L_rank_pos_neu = -torch.log(torch.sigmoid(pos_score - neu_score) + 1e-10)
        L_rank_neu_neg = -torch.log(torch.sigmoid(neu_score - neg_score) + 1e-10)
        
        L_rank = L_rank_pos_neu + L_rank_neu_neg
        
        return L_constrain, L_rank.mean()
    
    def pnn_loss(self, user_id, pos_items, unlabeled_items, lambda_val):
        """
        PNN損失の計算 (式7, 式17)
        L_PNN = αL_constrain + βL_uniform + L_rank
        
        Args:
            user_id (torch.Tensor): ユーザーIDのテンソル [batch_size]
            pos_items (torch.Tensor): ポジティブアイテムIDのテンソル [batch_size, num_pos]
            unlabeled_items (torch.Tensor): ラベルなしアイテムIDのテンソル [batch_size, num_unlabeled]
            lambda_val (torch.Tensor): λ値 [batch_size]
            
        Returns:
            tuple: (L_PNN, L_constrain, L_uniform, L_rank) - PNN損失とその構成要素
        """
        batch_size = user_id.size(0)
        
        # 各ユーザーのポジティブアイテム数を計算（パディングを除く）
        pos_mask = (pos_items != 0)
        num_pos_per_user = pos_mask.sum(dim=1)
        
        # ラベルなしデータを中立と負に分類
        neu_items, neg_items = self.classify_unlabeled_data(
            user_id, unlabeled_items, 
            num_neg=torch.max(torch.ones_like(num_pos_per_user), num_pos_per_user)
        )
        
        # 一様性損失
        L_uniform = self.uniform_loss(unlabeled_items)
        
        # セントロイドランキングによる拘束損失とランキング損失
        L_constrain, L_rank = self.centroid_ranking(user_id, pos_items, neu_items, neg_items)
        
        # PNN損失 (式17)
        L_PNN = self.alpha * L_constrain + self.beta * L_uniform + L_rank
        
        return L_PNN, L_constrain, L_uniform, L_rank