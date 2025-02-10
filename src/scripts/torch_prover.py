import json
import torch
from torch import nn
from pathlib import Path

from tqdm.auto import tqdm, trange

"""
import torch

# 乱数シードの設定
torch.manual_seed(0)

# パラメータ
n_samples = 100    # サンプル数
n_features = 3     # 特徴量の次元数
b_true = 2.0       # 真のバイアス項

# 真の重みを生成（ランダム）
w_true = torch.randn(n_features, 1)
print(f'True weights: {w_true.squeeze()}')
print(f'True bias: {b_true}')

# 特徴量行列Xの生成 (n_samples x n_features)
X = torch.randn(n_samples, n_features)

# ノイズを追加
noise = torch.randn(n_samples, 1) * 0.1

# 目的変数yの生成 (バイアス項 b_true を含む)
y = X @ w_true + b_true + noise

# バイアス項の列を特徴量行列Xに追加
X_with_bias = torch.cat([X, torch.ones(n_samples, 1)], dim=1)

# 解析的な最小二乗法を使用してパラメータを推定
# 正規方程式: (X^T * X)^-1 * X^T * y
XTX = X_with_bias.T @ X_with_bias
XTX_inv = torch.inverse(XTX)
XTy = X_with_bias.T @ y
w_estimated = XTX_inv @ XTy

# 結果の解析
w_estimated_weights = w_estimated[:-1].squeeze()  # 推定された重み（重みベクトル）
w_estimated_bias = w_estimated[-1].item()         # 推定されたバイアス

# 結果の出力
print(f'Estimated weights: {w_estimated_weights}')
print(f'Estimated bias: {w_estimated_bias}')

"""


class BaseTorchProverModel:
    model_type = "regression"

    @staticmethod
    def add_args(parser):
        return parser
    
    def __init__(self, args, device=None):
        self.args = args
        self.device = device if device is not None else torch.device('cpu')
        
    def set_device(self, device):
        self.device = device
        return self
    
    @torch.no_grad()
    def fit(self, X, y):
        raise NotImplementedError()

    @torch.no_grad()
    def predict(self, X):
        raise NotImplementedError()

    def to(self, device):
        self.device = device
        return self

class TorchLinearRegression(BaseTorchProverModel):
    def __init__(self, device=None):
        super().__init__(device)
        self.W = None
        self.b = None

    @torch.no_grad()
    def fit(self, X, y):
        if type(X) is not torch.Tensor:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        X_with_bias = torch.cat(
            [X, torch.ones(X.size(0), 1, device=self.device)],
            dim=1
        )
        XTX = X_with_bias.T @ X_with_bias
        XTX_inv = torch.inverse(XTX)
        XTy = X_with_bias.T @ y
        w_estimated = XTX_inv @ XTy
        self.W = w_estimated[:-1].squeeze()
        self.b = w_estimated[-1].item()


    @torch.no_grad()
    def predict(self, X):
        return (X @ self.W + self.b)



class TorchPrincipalComponentRegression(BaseTorchProverModel):
    def __init__(self, n_components, device=None):
        super().__init__(device)
        self.n_components = n_components
        self.regression_model = TorchLinearRegression(self.device)
        self.conversion_matrix = None
        
    @torch.no_grad()
    def fit(self, X, y):
        if type(X) is not torch.Tensor:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # 主成分分析
        U, S, V = torch.pca_lowrank(X)
        self.conversion_matrix = V[:, :self.n_components - 1]
        principal_component = torch.matmul(X, self.conversion_matrix).view(-1, self.n_components - 1)
        
        # 線形回帰
        self.regression_model.fit(principal_component, y)

    @torch.no_grad()
    def predict(self, X):
        principal_component = torch.matmul(X, self.conversion_matrix).view(-1, self.n_components - 1)
        return self.regression_model.predict(principal_component)
    
    
class TorchNoneLinearStochasticRegression(BaseTorchProverModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--n_layer", type=int, default=1)
        parser.add_argument("--n_epoch", type=int, default=30)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--act_fn", type=str, default="relu")
        return parser
    
    
    def __init__(
            self,
            args=None,
            device=None
    ):
        super().__init__(args, device)
        self.n_layer = args.n_layer
        self.n_hidden = None
        self.n_epoch = args.n_epoch
        self.lr = args.lr
        self.act_fn = args.act_fn
        self.model = None
        
    def model_setup(self, num_features):
        if self.act_fn == "relu":
            act_fn = nn.ReLU()
        elif self.act_fn == "sigmoid":
            act_fn = nn.Sigmoid()
        elif self.act_fn == "tanh":
            act_fn = nn.Tanh()
        else:
            raise NotImplementedError()
        
        self.n_hidden = num_features

        layers = []
        for i in range(self.n_layer - 1):
            if i == 0:
                layers.append(nn.Linear(num_features, self.n_hidden))
            else:
                layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            layers.append(act_fn)
            
        layers.append(nn.Linear(self.n_hidden, 1))
        self.model = torch.nn.Sequential(*layers)

    def fit(self, X, y):
        if type(X) is not torch.Tensor:
            X = torch.stack(X, dim=0).to(self.device)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        self.model_setup(X.size(1))
        self.model.to(self.device)
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        logs = {
            "loss": []
        }
        for epoch in trange(self.n_epoch):
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = criterion(y_pred.view(-1), y)
            loss.backward()
            optimizer.step()
            logs["loss"].append(loss.item())
        
        self.model.to("cpu")
        return logs

        
    @torch.no_grad()
    def predict(self, X):
        if type(X) is not torch.Tensor:
            X = torch.stack(X, dim=0).to(self.device)
            
        self.model.eval()
        self.model.to(self.device)
        prediction =  self.model(X).view(-1)
        self.model.to("cpu")
        return prediction

            


class TorchNoneLinearStochasticClassifier(BaseTorchProverModel):
    model_type = "classification"
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--n_layer", type=int, default=1)
        parser.add_argument("--n_epoch", type=int, default=30)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--act_fn", type=str, default="relu")
        parser.add_argument("--label_map_path", type=Path, required=True)
        return parser
    
    def __init__(
            self,
            args=None,
            device=None
    ):
        super().__init__(args, device)
        self.n_layer = self.args.n_layer
        self.n_hidden = None
        self.n_epoch = self.args.n_epoch
        self.lr = self.args.lr
        self.act_fn = self.args.act_fn
        with self.args.label_map_path.open("r") as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)
        self.model = None
        
    def model_setup(self, num_features):
        if self.act_fn == "relu":
            act_fn = nn.ReLU()
        elif self.act_fn == "sigmoid":
            act_fn = nn.Sigmoid()
        elif self.act_fn == "tanh":
            act_fn = nn.Tanh()
        else:
            raise NotImplementedError()
        
        self.n_hidden = num_features

        layers = []
        for i in range(self.n_layer - 1):
            if i == 0:
                layers.append(nn.Linear(num_features, self.n_hidden))
            else:
                layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            layers.append(act_fn)
            
        layers.append(nn.Linear(self.n_hidden, self.num_classes))
        self.model = torch.nn.Sequential(*layers)

        
    def fit(self, X, y):
        if type(X) is not torch.Tensor:
            X = torch.stack(X, dim=0).to(self.device)
            
        if type(y) is not list:
            y = y.tolist()
            
        # Convert to labels
        labels = [self.label_map[str(i)] for i in y]
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
                    
        self.model_setup(X.size(1))
        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        logs = {
            "loss": []
        }
        for epoch in trange(self.n_epoch):
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            logs["loss"].append(loss.item())
        self.model.to("cpu")
        
        return logs
        
        
    @torch.no_grad()
    def predict(self, X):
        if type(X) is not torch.Tensor:
            X = torch.stack(X, dim=0).to(self.device)
            
        self.model.eval()
        self.model.to(self.device)
        prediction =  self.model(X)
        prediction = prediction.argmax(dim=-1).view(-1)
        self.model.to("cpu")
        return prediction

