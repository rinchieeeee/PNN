import torch
import numpy as np
from tqdm import tqdm

# グローバル変数
train_interactions = {}

def evaluate(model, test_interactions, num_items, device, k=10):
    """
    モデルの評価を行い、Recall@k, Hit@k, NDCG@kを計算する
    
    Args:
        model (nn.Module): 評価するモデル
        test_interactions (dict): テスト用ユーザーインタラクション
        num_items (int): アイテム数
        device (torch.device): 使用するデバイス
        k (int): トップkアイテムを評価
        
    Returns:
        tuple: (avg_recall, avg_hit, avg_ndcg)
    """
    global train_interactions
    
    model.eval()
    
    recalls = []
    hits = []
    ndcgs = []
    
    with torch.no_grad():
        for user_id, test_items in tqdm(test_interactions.items(), desc='Evaluating'):
            # ユーザーIDをテンソルに変換
            user_tensor = torch.tensor([user_id], device=device)
            
            # 予測スコアを取得
            scores = model(user_tensor).cpu().numpy().flatten()
            
            # トップkのアイテムを取得（学習済みアイテムをマスク）
            if user_id in train_interactions:
                user_train_items = train_interactions[user_id]
                for item in user_train_items:
                    scores[item] = -float('inf')  # 学習済みアイテムを除外
            
            # トップkのアイテムを取得
            top_k_items = np.argsort(-scores)[:k]
            
            # 評価指標を計算
            hit = len(set(top_k_items) & set(test_items)) > 0
            recall = len(set(top_k_items) & set(test_items)) / len(test_items) if len(test_items) > 0 else 0
            
            # NDCG@kの計算
            dcg = 0
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(test_items), k))])
            
            for i, item in enumerate(top_k_items):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            
            recalls.append(recall)
            hits.append(hit)
            ndcgs.append(ndcg)
    
    avg_recall = np.mean(recalls)
    avg_hit = np.mean(hits)
    avg_ndcg = np.mean(ndcgs)
    
    return avg_recall, avg_hit, avg_ndcg