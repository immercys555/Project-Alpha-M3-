import torch
import torch.nn as nn
import torch.nn.functional as F

class M3ProxyOperator(nn.Module):
    """
    M3 Cognitive Architecture - Core Proxy Operator (Project Alpha)
    Mathematical Formula: H_total = H_percept + (S_sub * Phi_recurrence)
    """
    def __init__(self, d_model: int = 512, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 1. Perceptual Projection (H_percept)
        self.proj_percept = nn.Linear(d_model, d_model)
        
        # 2. Subconscious Tensor Field Generator (S_sub)
        self.proj_subconscious = nn.Linear(d_model, d_state)
        self.sub_density_gate = nn.Linear(d_model, d_state)
        
        # 3. Non-linear Self-Coupling Recurrence Factor (Phi_recurrence)
        self.phi_recurrence = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.Tanh(),  # Constrains non-linear feedback to [-1, 1]
            nn.Linear(d_state, d_model)
        )
        
        # Boundary Norms & Continuous Decay Vector for Gradient Stability
        self.layer_norm = nn.LayerNorm(d_model)
        self.manifold_decay = nn.Parameter(torch.ones(d_state) * 0.95)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        """
        x: [Batch_Size, Seq_Len, d_model]
        prev_state: [Batch_Size, d_state]
        """
        batch_size, seq_len, _ = x.shape
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.d_state, device=x.device)
            
        output_tokens = []
        current_state = prev_state
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_percept = self.proj_percept(x_t)
            
            s_candidate = torch.sigmoid(self.proj_subconscious(x_t))
            gate = torch.sigmoid(self.sub_density_gate(x_t))
            
            # State Update with Bounded Spectral Decay
            current_state = (self.manifold_decay * current_state) + (gate * s_candidate)
            s_sub = current_state
            
            phi = self.phi_recurrence(s_sub)
            s_energy = torch.mean(s_sub, dim=-1, keepdim=True)
