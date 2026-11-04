import torch
import torch.nn as nn

class M3TensorCouplingOperator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_percept = nn.Linear(d_model, d_model)
        self.W_sub = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # H_percept 感知向量提取
        h_percept = torch.mean(torch.relu(self.W_percept(x)), dim=1) # (batch, d_model)
        
        # S_sub 基底動力學提取
        s_sub = torch.mean(torch.sigmoid(self.W_sub(x)), dim=1)      # (batch, d_model)
        
        # 張量耦合與整合係數計算 (確保維度完美對齊)
        phi_integration = torch.tanh(torch.sum(h_percept * s_sub, dim=-1, keepdim=True)) # (batch, 1)
        
        # 最終系統成熟度 H_total 計算
        h_total = torch.mean(h_percept) + torch.mean(s_sub * phi_integration)
        return h_total

# 實測驗證
if __name__ == "__main__":
    d_model = 64
    batch_size = 2
    seq_len = 1024  # 模擬大上下文 O(N) 線性複雜度測試
    x = torch.randn(batch_size, seq_len, d_model)

    model = M3TensorCouplingOperator(d_model)
    output = model(x)
    print("M3 Proxy Execution Successful!")
    print("H_total output value:", output.item())
