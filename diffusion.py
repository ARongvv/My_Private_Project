import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, LayerNorm
from torch_geometric.nn import radius_graph
from Bio import SeqIO
import math
import datetime

# 配置参数
class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if torch.cuda.is_available() else 8  # 根据显存调整
    lr = 0.001
    epochs = 50
    seq_vocab = "AUCG"
    coord_dims = 7  
    hidden_dim = 256
    num_layers = 4  # 减少层数防止显存溢出
    k_neighbors = 20  
    dropout = 0.1
    rbf_dim = 16
    num_heads = 4
    amp_enabled = True  # 混合精度训练
    timesteps = 1000  # 添加时间步数配置

# 几何特征生成器
class GeometricFeatures:
    @staticmethod
    def rbf(D, D_min=0., D_max=20., D_count=16):
        device = D.device
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view(*[1]*len(D.shape), -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        return torch.exp(-((D_expand - D_mu)/D_sigma) ** 2)

    @staticmethod
    def dihedrals(X, eps=1e-7):   # 混合精度训练
        X = X.to(torch.float32)
        L = X.shape[0]
        dX = X[1:] - X[:-1]
        U = F.normalize(dX, dim=-1)
        
        # 计算连续三个向量
        u_prev = U[:-2]
        u_curr = U[1:-1]
        u_next = U[2:]

        # 计算法向量
        n_prev = F.normalize(torch.cross(u_prev, u_curr, dim=-1), dim=-1)
        n_curr = F.normalize(torch.cross(u_curr, u_next, dim=-1), dim=-1)

        # 计算二面角
        cosD = (n_prev * n_curr).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_prev * n_curr).sum(-1)) * torch.acos(cosD)

        # 填充处理
        if D.shape[0] < L:
            D = F.pad(D, (0,0,0,L-D.shape[0]), "constant", 0)
        
        return torch.stack([torch.cos(D[:,:5]), torch.sin(D[:,:5])], -1).view(L,-1)

    @staticmethod
    def direction_feature(X):
        dX = X[1:] - X[:-1]
        return F.pad(F.normalize(dX, dim=-1), (0,0,0,1))

# 图构建器
class RNAGraphBuilder:
    @staticmethod
    def build_graph(coord, seq):
        assert coord.shape[1:] == (7,3), f"坐标维度错误: {coord.shape}"
        coord = torch.tensor(coord, dtype=torch.float32)
        
        # 节点特征
        node_feats = [
            coord.view(-1, 7 * 3),  # [L,21], 展平为21维
            GeometricFeatures.dihedrals(coord[:,:6,:]),  # [L,10],
            GeometricFeatures.direction_feature(coord[:,4,:])  # [L,3]
        ]
        x = torch.cat(node_feats, dim=-1)  # [L,34]

        # 边构建
        pos = coord[:,4,:]  # 第五个骨架节点的xyz
        edge_index = radius_graph(pos, r=20.0, max_num_neighbors=Config.k_neighbors)
        
        # 边特征
        row, col = edge_index
        edge_vec = pos[row] - pos[col]  #起点坐标-终点坐标 = 边的向量[num_edges, 3]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True) #边的长度
        edge_feat = torch.cat([
            GeometricFeatures.rbf(edge_dist).squeeze(1),  # [E,16]
            F.normalize(edge_vec, dim=-1)  # [E,3]
        ], dim=-1)  # [E,19]

        # 标签
        y = torch.tensor([Config.seq_vocab.index(c) for c in seq], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_feat, y=y)

#扩散模型
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义扩散过程的参数
        self.timesteps = Config.timesteps  # 使用配置中的时间步数
        self.beta = torch.linspace(0.0001, 0.02, self.timesteps, device=Config.device)  # 修复设备问题
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_start, t, noise):
        """前向扩散过程：添加噪声"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, x, t, pred_noise):
        """反向扩散过程：去噪"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1)

        # 预测去噪后的 x_start
        pred_x_start = (x - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
        pred_x_start = torch.clamp(pred_x_start, -1, 1)

        # 计算反向扩散的均值
        mean = sqrt_alpha_bar * pred_x_start + sqrt_one_minus_alpha_bar * pred_noise
        if t > 0:
            noise = torch.randn_like(x)
            mean += torch.sqrt(beta_t) * noise
        return mean

#扩散模型训练函数
def train_diffusion(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(Config.device)
        optimizer.zero_grad()

        # 随机选择时间步
        t = torch.randint(0, Config.timesteps, (batch.x.size(0),), device=Config.device)
        noise = torch.randn_like(batch.x)

        # 前向传播
        pred_noise = model(batch, t, noise)
        loss = F.mse_loss(pred_noise, noise)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(loader)

#扩散模型采样函数
@torch.no_grad()
def sample_diffusion(model, data, num_samples=1):
    model.eval()
    samples = []

    for _ in range(num_samples):
        x = torch.randn(data.num_nodes, Config.hidden_dim).to(Config.device)

        for t in range(Config.timesteps - 1, -1, -1):
            t_batch = torch.full((x.size(0),), t, device=Config.device, dtype=torch.long)
            pred_noise = model(data, t_batch, None)
            x = model.diffusion.p_sample(x, t_batch, pred_noise)

        # 转换为序列
        logits = model.cls_head(x)
        pred_seq = logits.argmax(dim=-1)
        samples.append("".join([Config.seq_vocab[i] for i in pred_seq.cpu().numpy()]))

    return samples

# 模型架构
class RNAGNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 节点特征编码
        self.feat_encoder = nn.Sequential(
            nn.Linear(34, Config.hidden_dim),
            nn.ReLU(),
            LayerNorm(Config.hidden_dim),
            nn.Dropout(Config.dropout)
        )
        
        # 边特征编码（关键修复）
        self.edge_encoder = nn.Sequential(
            nn.Linear(19, Config.hidden_dim),
            nn.ReLU(),
            LayerNorm(Config.hidden_dim),
            nn.Dropout(Config.dropout)
        )

        # Transformer卷积层
        self.convs = nn.ModuleList([
            TransformerConv(
                Config.hidden_dim,
                Config.hidden_dim // Config.num_heads,
                heads=Config.num_heads,
                edge_dim=Config.hidden_dim,  # 匹配编码后维度
                dropout=Config.dropout
            ) for _ in range(Config.num_layers)
        ])

        # 残差连接
        self.mlp_skip = nn.ModuleList([
            nn.Sequential(
                nn.Linear(Config.hidden_dim, Config.hidden_dim),
                nn.ReLU(),
                LayerNorm(Config.hidden_dim)
            ) for _ in range(Config.num_layers)
        ])

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.ReLU(),
            LayerNorm(Config.hidden_dim),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, len(Config.seq_vocab))
        )

        # 扩散模型
        self.diffusion = DiffusionModel()  # 修复初始化问题

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, data, t=None, noise=None):  # 修改 forward 方法以支持时间步数和噪声
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 边特征编码（关键步骤）
        edge_attr = self.edge_encoder(edge_attr)  # [E,19] -> [E,256]
        
        # 节点编码
        h = self.feat_encoder(x)
        
        # 消息传递
        for i, (conv, skip) in enumerate(zip(self.convs, self.mlp_skip)):
            h_res = conv(h, edge_index, edge_attr=edge_attr)
            h = h + skip(h_res)
            if i < len(self.convs)-1:
                h = F.relu(h)
                h = F.dropout(h, p=Config.dropout, training=self.training)

        # 扩散过程
        if t is not None and noise is not None:
            noisy_x = self.diffusion.q_sample(h, t, noise)
            pred_noise = self.cls_head(noisy_x)
            return pred_noise
        else:
            return self.cls_head(h)

# 数据增强
class CoordTransform:
    @staticmethod
    def random_rotation(coords):
        device = torch.device(Config.device)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        angle = np.random.uniform(0, 2*math.pi)
        rot_mat = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ], device=device)
        return (coords_tensor @ rot_mat.T).cpu().numpy()
    def add_noise(coords, noise_level=0.01):
        noise = np.random.normal(0, noise_level, coords.shape)
        return coords + noise

# 数据集类
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, coords_dir, seqs_dir, augment=False):
        self.samples = []
        self.augment = augment
        
        for fname in os.listdir(coords_dir):
            # 加载坐标
            coord = np.load(os.path.join(coords_dir, fname))
            #  # 创建掩码，标记 NaN 值的位置
            # mask = np.isnan(coord).astype(np.float32)  # NaN 值为 1，其余为 0   

            coord = np.nan_to_num(coord, nan=0.0)    # 将数组中的 NaN（Not a Number）替换为0.0，改
           
            # 数据增强 ，50%的概率对坐标数据进行随机旋转变换
            if self.augment and np.random.rand() > 0.5: 
                coord = CoordTransform.random_rotation(coord)  
                coord = CoordTransform.add_noise(coord)    # 添加噪声,改

            # 加载序列
            seq_id = os.path.splitext(fname)[0]
            seq_path = os.path.join(seqs_dir, f"{seq_id}.fasta")
            seq = str(next(SeqIO.parse(seq_path, "fasta")).seq)
            
            # 构建图
            # graph = RNAGraphBuilder.build_graph(coord, seq)
            # graph.mask = torch.tensor(mask, dtype=torch.float32)  # 将掩码添加到图中
            self.samples.append(RNAGraphBuilder.build_graph(coord, seq))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# 训练函数
def train(model, loader, optimizer, scheduler, criterion):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.amp_enabled)
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(Config.device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=Config.amp_enabled):
            logits = model(batch)
            loss = criterion(logits, batch.y)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    scheduler.step()
    return total_loss / len(loader)

# 评估函数
def evaluate(model, loader):
    model.eval()
    total_correct = total_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(Config.device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch.y).sum().item()
            total_nodes += batch.y.size(0)
    return total_correct / total_nodes

if __name__ == "__main__":
    # 初始化
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
        torch.backends.cudnn.benchmark = True
    
    # 数据集
    train_set = RNADataset(
        "./sais_third_medicine_baseline/RNA_design_public/RNAdesignv1/train/coords",
        "./sais_third_medicine_baseline/RNA_design_public/RNAdesignv1/train/seqs",
        augment=True
    )
    
    # 划分数据集
    train_size = int(0.8 * len(train_set))
    val_size = (len(train_set) - train_size) // 2
    test_size = len(train_set) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        train_set, [train_size, val_size, test_size])
    
    # 数据加载
    train_loader = torch_geometric.loader.DataLoader(
        train_set, 
        batch_size=Config.batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = torch_geometric.loader.DataLoader(val_set, batch_size=Config.batch_size)
    test_loader = torch_geometric.loader.DataLoader(test_set, batch_size=Config.batch_size)
    
    # 模型初始化
    model = RNAGNN().to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_path = "best_model.pth"

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        model = RNAGNN().to(Config.device)  # 确保模型在加载权重之前定义
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))  # 修复设备问题

        # 训练循环
        best_acc = 0
        for epoch in range(Config.epochs):
            train_loss = train(model, train_loader, optimizer, scheduler, criterion)
            val_acc = evaluate(model, val_loader)
            test_acc = evaluate(model, test_loader)

            print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
        
        # 最终测试

        # 添加时间戳到权重文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), f"best_model_{timestamp}.pth")

        model.load_state_dict(torch.load("best_model.pth"))
        test_acc = evaluate(model, test_loader)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")