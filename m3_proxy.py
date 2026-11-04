import torch
import torch.nn as nn

class M3TensorCouplingOperator(nn.Module):
    """
    M3 Cognitive Architecture - Core Proxy Operator (Project Alpha)
    Mathematical Formula: H_total = H_percept + (S_sub * Phi_integration)
    """
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.W_percept = nn.Linear(d_model, d_model)
        self.W_sub = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)

        # 1. Perceptual Vector (H_percept): aggregate over sequence
        h_percept = torch.mean(torch.relu(self.W_percept(x)), dim=1)  # (batch, d_model)

        # 2. Substrate Dynamics (S_sub): aggregate over sequence
        s_sub = torch.mean(torch.sigmoid(self.W_sub(x)), dim=1)  # (batch, d_model)

        # 3. Integration Coefficient (Phi_integration): scalar per sample
        phi_integration = torch.tanh(
            torch.sum(h_percept * s_sub, dim=-1, keepdim=True)
        )  # (batch, 1)

        # 4. Final system maturity score: preserve feature dimension
        h_total = h_percept + (s_sub * phi_integration)  # (batch, d_model)
        return h_total

if __name__ == "__main__":
    d_model = 64
    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, d_model)
    model = M3TensorCouplingOperator(d_model)
    output = model(x)
    print("M3 Proxy Execution Successful!")
    print("H_total shape:", output.shape)
    print("H_total mean value:", output.mean().item())