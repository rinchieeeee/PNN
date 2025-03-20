import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import matplotlib
matplotlib.use('Agg')  # GUIなしでプロットを生成
import matplotlib.pyplot as plt

from src.data.dataset import load_movielens_data, CFDataset
from src.models.bprmf import BPRMF
from src.models.pnn import PNNBPRMF
from src.trainers.bprmf_trainer import BPRMFTrainer
from src.trainers.pnn_trainer import PNNTrainer
from src.utils.evaluation import evaluate

# グローバル変数をモジュールレベルで定義
train_interactions = {}

# 再現性のために乱数シードを設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='PNN実験の実行')
    parser.add_argument('--data_path', type=str, default='ml-1m/ratings.dat',
                        help='MovieLensデータセットのパス')
    parser.add_argument('--model', type=str, choices=['bprmf', 'pnn', 'both'], default='pnn',
                        help='使用するモデル（bprmf, pnn, both）')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='埋め込みの次元数')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=50,
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学習率')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='PNNのα（拘束損失の重み）')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='PNNのβ（一様性損失の重み）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='重み減衰（L2正則化）')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='結果を保存するディレクトリ')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード')
    parser.add_argument('--preprocessing', type=bool, default=False, help='前処理をするかのフラグ')
    
    return parser.parse_args()

def run_bprmf_experiment(train_interactions, val_interactions, test_interactions,
                         num_users, num_items, args, device):
    """
    BPR-MFモデルの実験を実行
    
    Args:
        train_interactions (dict): トレーニング用ユーザーインタラクション
        val_interactions (dict): 検証用ユーザーインタラクション
        test_interactions (dict): テスト用ユーザーインタラクション
        num_users (int): ユーザー数
        num_items (int): アイテム数
        args (argparse.Namespace): コマンドライン引数
        device (torch.device): 使用するデバイス
        
    Returns:
        dict: 実験結果
    """
    print("\nBPR-MFモデルの実験を開始...")
    
    # データローダーの設定
    train_dataset = CFDataset(train_interactions, num_users, num_items)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # モデルの初期化
    model = BPRMF(num_users, num_items, args.embedding_dim).to(device)
    
    # オプティマイザーの設定
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # トレーナーの初期化
    trainer = BPRMFTrainer(model, optimizer, device)
    
    # 学習と評価
    results = trainer.train(train_loader, val_interactions, test_interactions, 
                          num_items, args.epochs, 'bprmf')
    
    # 結果を保存
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    save_path = os.path.join(args.save_dir, 'bprmf_results.pt')
    torch.save(results, save_path)
    
    return results

def run_pnn_experiment(train_interactions, val_interactions, test_interactions,
                       num_users, num_items, args, device):
    """
    PNN-BPRMFモデルの実験を実行
    
    Args:
        train_interactions (dict): トレーニング用ユーザーインタラクション
        val_interactions (dict): 検証用ユーザーインタラクション
        test_interactions (dict): テスト用ユーザーインタラクション
        num_users (int): ユーザー数
        num_items (int): アイテム数
        args (argparse.Namespace): コマンドライン引数
        device (torch.device): 使用するデバイス
        
    Returns:
        dict: 実験結果
    """
    print("\nPNNモデルの実験を開始...")
    
    # データローダーの設定
    train_dataset = CFDataset(train_interactions, num_users, num_items)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # モデルの初期化
    model = PNNBPRMF(num_users, num_items, args.embedding_dim, args.alpha, args.beta).to(device)
    
    # オプティマイザーの設定
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # トレーナーの初期化
    trainer = PNNTrainer(model, optimizer, device)
    
    # 学習と評価
    results = trainer.train(train_loader, val_interactions, test_interactions, 
                           num_items, args.epochs, 'pnn', train_interactions)
    
    # 結果を保存
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    save_path = os.path.join(args.save_dir, 'pnn_results.pt')
    torch.save(results, save_path)
    
    return results

def compare_results(bprmf_results, pnn_results):
    """
    BPR-MFとPNNの結果を比較
    
    Args:
        bprmf_results (dict): BPR-MFの結果
        pnn_results (dict): PNNの結果
    """
    print("\n結果の比較:")
    print(f"BPR-MF - Recall@10: {bprmf_results['test_recall']:.4f}, Hit@10: {bprmf_results['test_hit']:.4f}, NDCG@10: {bprmf_results['test_ndcg']:.4f}")
    print(f"PNN    - Recall@10: {pnn_results['test_recall']:.4f}, Hit@10: {pnn_results['test_hit']:.4f}, NDCG@10: {pnn_results['test_ndcg']:.4f}")
    
    improvement_recall = (pnn_results['test_recall'] - bprmf_results['test_recall']) / bprmf_results['test_recall'] * 100
    improvement_hit = (pnn_results['test_hit'] - bprmf_results['test_hit']) / bprmf_results['test_hit'] * 100
    improvement_ndcg = (pnn_results['test_ndcg'] - bprmf_results['test_ndcg']) / bprmf_results['test_ndcg'] * 100
    
    print(f"改善率 - Recall@10: {improvement_recall:.2f}%, Hit@10: {improvement_hit:.2f}%, NDCG@10: {improvement_ndcg:.2f}%")
    
    # 比較グラフの描画
    plt.figure(figsize=(10, 6))
    
    metrics = ['Recall@10', 'Hit@10', 'NDCG@10']
    bprmf_values = [bprmf_results['test_recall'], bprmf_results['test_hit'], bprmf_results['test_ndcg']]
    pnn_values = [pnn_results['test_recall'], pnn_results['test_hit'], pnn_results['test_ndcg']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, bprmf_values, width, label='BPR-MF')
    plt.bar(x + width/2, pnn_values, width, label='PNN')
    
    plt.xlabel('メトリクス')
    plt.ylabel('スコア')
    plt.title('BPR-MFとPNNの比較')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存先ディレクトリの確認
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.savefig('results/comparison_results.png')

def main():
    """
    メイン実行関数
    """
    # コマンドライン引数の解析
    args = parse_args()
    
    # 乱数シードの設定
    set_seed(args.seed)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データの読み込み
    global train_interactions
    print("MovieLensデータセットの読み込み中...")
    start_time = time.time()
    
    train_interactions_local, val_interactions, test_interactions, num_users, num_items, user_id_map, item_id_map = load_movielens_data(file_path=args.data_path, preprocessing=args.preprocessing)
    
    # グローバル変数に格納（評価時に使用するため）
    train_interactions.update(train_interactions_local)
    
    # src/utils/evaluation.py でも同じ変数を使用するための設定
    import src.utils.evaluation
    src.utils.evaluation.train_interactions = train_interactions
    
    print(f"データセット読み込み完了: {time.time() - start_time:.2f}秒")
    print(f"データセット情報:")
    print(f"ユーザー数: {num_users}, アイテム数: {num_items}")
    print(f"トレーニングインタラクション: {sum(len(items) for items in train_interactions.values())}")
    print(f"検証インタラクション: {sum(len(items) for items in val_interactions.values())}")
    print(f"テストインタラクション: {sum(len(items) for items in test_interactions.values())}")
    
    # 実験の実行
    bprmf_results = None
    pnn_results = None
    
    if args.model == 'bprmf' or args.model == 'both':
        bprmf_results = run_bprmf_experiment(train_interactions, val_interactions, test_interactions, 
                                           num_users, num_items, args, device)
    
    if args.model == 'pnn' or args.model == 'both':
        pnn_results = run_pnn_experiment(train_interactions, val_interactions, test_interactions, 
                                        num_users, num_items, args, device)
    
    # 結果の比較
    if args.model == 'both' and bprmf_results is not None and pnn_results is not None:
        compare_results(bprmf_results, pnn_results)

if __name__ == "__main__":
    main()
