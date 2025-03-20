import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def load_movielens_data(file_path='ml-1m/ratings.dat', test_size=0.1, preprocessing=False):
    """
    MovieLensデータセットを読み込み、トレーニング、検証、テストセットに分割する
    
    Args:
        file_path (str): MovieLensデータファイルのパス
        test_size (float): テストと検証セットの割合（合計）
        
    Returns:
        tuple: (train_interactions, val_interactions, test_interactions, 
               num_users, num_items, user_id_map, item_id_map)
    """
    # データの読み込み
    ratings = pd.read_csv(file_path, sep='::', 
                          names=['user_id', 'item_id', 'rating', 'timestamp'],
                          engine='python')
    
    # インタラクションデータを作成（評価値が0より大きいものをポジティブとする）
    interactions = ratings[ratings['rating'] > 0].copy()
    
    # ユーザーとアイテムのマッピングを作成
    user_ids = interactions['user_id'].unique()
    item_ids = interactions['item_id'].unique()
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(item_ids)}
    
    # IDをマッピング
    interactions['user_id'] = interactions['user_id'].map(user_id_map)
    interactions['item_id'] = interactions['item_id'].map(item_id_map)
    
    # ユーザーごとのインタラクションを取得
    user_interactions = defaultdict(list)
    for row in interactions.itertuples():
        user_interactions[row.user_id].append(row.item_id)
    
    # トレーニング、検証、テストセットに分割
    train_interactions = {}
    val_interactions = {}
    test_interactions = {}
    
    for user_id, items in user_interactions.items():
        # 少なくとも5つのインタラクションがあるユーザーのみ
        if preprocessing:
            if len(items) >= 5:
                train_items, test_val_items = train_test_split(items, test_size=test_size*2, random_state=42)
                val_items, test_items = train_test_split(test_val_items, test_size=0.5, random_state=42)
                
                train_interactions[user_id] = train_items
                val_interactions[user_id] = val_items
                test_interactions[user_id] = test_items
        else:
            train_items, test_val_items = train_test_split(items, test_size=test_size*2, random_state=42)
            val_items, test_items = train_test_split(test_val_items, test_size=0.5, random_state=42)
            
            train_interactions[user_id] = train_items
            val_interactions[user_id] = val_items
            test_interactions[user_id] = test_items
    
    return (train_interactions, val_interactions, test_interactions, 
            len(user_ids), len(item_ids), user_id_map, item_id_map)


class CFDataset(Dataset):
    """協調フィルタリング用のデータセット"""
    def __init__(self, user_interactions, num_users, num_items, negative_sample_size=1):
        """
        Args:
            user_interactions (dict): ユーザーごとのポジティブアイテムリスト
            num_users (int): ユーザー数
            num_items (int): アイテム数
            negative_sample_size (int): 各ポジティブサンプルに対するネガティブサンプルの数
        """
        self.user_interactions = user_interactions
        self.users = list(user_interactions.keys())
        self.num_users = num_users
        self.num_items = num_items
        self.negative_sample_size = negative_sample_size
        
        # ユーザーごとのポジティブアイテムの集合を作成
        self.user_positive_items = {user: set(items) for user, items in user_interactions.items()}
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (user_id, pos_item_id, neg_items)
        """
        user_id = self.users[idx]
        pos_items = self.user_interactions[user_id]
        
        # ポジティブサンプルをランダムに選択
        pos_item_id = random.choice(pos_items)
        
        # ネガティブサンプルを取得（ユーザーがインタラクションしていないアイテム）
        neg_items = []
        for _ in range(self.negative_sample_size):
            neg_item_id = random.randint(0, self.num_items - 1)
            while neg_item_id in self.user_positive_items[user_id]:
                neg_item_id = random.randint(0, self.num_items - 1)
            neg_items.append(neg_item_id)
        
        return user_id, pos_item_id, neg_items