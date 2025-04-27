import numpy as np
import torch
import torch.nn as nn
import os
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import glob

class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    lr = 0.01
    epochs = 20
    seq_vocab = "AUCG"
    coord_dims = 7  # 7个骨架点
    hidden_dim = 128
    k_neighbors = 5  # 每个节点的近邻数

class GNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征编码
        self.encoder = nn.Sequential(
            nn.Linear(7*3, Config.hidden_dim),
            nn.ReLU()
        )

        # GNN层
        self.conv1 = GCNConv(Config.hidden_dim, Config.hidden_dim)
        self.conv2 = GCNConv(Config.hidden_dim, Config.hidden_dim)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(Config.hidden_dim, len(Config.seq_vocab))
        )

    def forward(self, data):
        # 节点特征编码
        x = self.encoder(data.x)  # [N, hidden]

        # 图卷积
        x = self.conv1(x, data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)
        x = torch.relu(x)

        # 节点分类
        logits = self.cls_head(x)  # [N, 4]
        return logits

class RNAGraphBuilder:
    @staticmethod
    def build_graph(coord, seq):
        """将坐标和序列转换为图结构"""
        num_nodes = coord.shape[0]

        # 节点特征：展平每个节点的7个骨架点坐标
        x = torch.tensor(coord.reshape(num_nodes, -1), dtype=torch.float32)  # [N, 7*3]

        # 边构建：基于序列顺序的k近邻连接
        edge_index = []
        for i in range(num_nodes):
            # 连接前k和后k个节点
            neighbors = list(range(max(0, i-Config.k_neighbors), i)) + \
                       list(range(i+1, min(num_nodes, i+1+Config.k_neighbors)))
            for j in neighbors:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 双向连接

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 节点标签
        y = torch.tensor([Config.seq_vocab.index(c) for c in seq], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

class RNASequenceGenerator:
    def __init__(self, model_path):
        self.model = GNNModel().to(Config.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=Config.device, weights_only=True)
        )
        self.model.eval()

    def generate_sequences(self, coord_data, num_seq=5, temperature=1.0, top_k=3):
        """
        生成候选RNA序列
        :param coord_data: numpy数组 [L, 7, 3]
        :param num_seq: 需要生成的序列数量
        :param temperature: 温度参数控制多样性
        :param top_k: 每个位置只考虑top_k高概率的碱基
        :return: 生成的序列列表
        """
        # 转换为图数据
        graph = self._preprocess_data(coord_data)
        graph = graph.to(Config.device)  # 关键修复：转移数据到模型设备
        # 获取概率分布
        with torch.no_grad():
            logits = self.model(graph)
            probs = F.softmax(logits / temperature, dim=1)  # [L, 4]

        # 生成候选序列
        sequences = set()
        while len(sequences) < num_seq:
            seq = self._sample_sequence(probs, top_k)
            sequences.add(seq)

        return list(sequences)[:num_seq]

    def _preprocess_data(self, coord):
        """预处理坐标数据为图结构"""
        # 创建伪序列（实际不会被使用）
        dummy_seq = "A" * coord.shape[0]
        return RNAGraphBuilder.build_graph(coord, dummy_seq)

    def _sample_sequence(self, probs, top_k):
        """基于概率分布进行采样"""
        seq = []
        for node_probs in probs:
            # 应用top-k筛选
            topk_probs, topk_indices = torch.topk(node_probs, top_k)
            # 重新归一化
            norm_probs = topk_probs / topk_probs.sum()
            # 采样
            chosen = np.random.choice(topk_indices.cpu().numpy(), p=norm_probs.cpu().numpy())
            seq.append(Config.seq_vocab[chosen])
        return "".join(seq)

# 使用示例
if __name__ == "__main__":
    # 加载生成器
    generator = RNASequenceGenerator("best_gnn_model.pth")
    result={
     "pdb_id":[],
     "seq":[]
    }
    # 示例输入（替换为真实骨架坐标）
    npy_files=glob.glob("/saisdata/coords/*.npy")
    for npy in npy_files:
        id_name=os.path.basename(npy).split(".")[0]
        coord = np.load(npy)  # [L, 7, 3]
        coord = np.nan_to_num(coord, nan=0.0)  # 新增行：将NaN替换为0

        # 生成1个候选序列
        candidates = generator.generate_sequences(
            coord,
            num_seq=1,
            temperature=0.8,  # 适度多样性
            top_k=4          # 每个位置考虑前4个可能
        )
        result["pdb_id"].append(id_name)
        result["seq"].append(candidates[0])
    result=pd.DataFrame(result)
    result.to_csv("/saisresult/submit.csv",index=False)


