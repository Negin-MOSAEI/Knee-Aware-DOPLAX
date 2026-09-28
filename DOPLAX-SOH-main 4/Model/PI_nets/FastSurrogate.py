import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class FastSurrogateMLP(nn.Module):
    """
    Lightweight Amortized Predictor for Edge BMS Inference.
    Acts as the Student model distilled from the heavy LAX Teacher.
    """
    def __init__(self, window_size=40, num_features=3, hidden_dim=64):
        super().__init__()
        input_dim = window_size * num_features
        
        # A fast, lightweight MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Predicts u_2 (LAX SOH prediction)
        )

    def forward(self, features):
        """
        features: [B, window_size, num_features]
        """
        batch_size = features.shape[0]
        x_flat = features.reshape(batch_size, -1)
        return self.net(x_flat)

def train_surrogate_distillation(teacher_lax_model, dataloader, device, epochs=50):
    """
    Knowledge Distillation Logic:
    Distills the complex inner optimization of the Teacher LAX model 
    into the fast Student MLP for inference.
    """
    print("Starting Knowledge Distillation from LAX Teacher to Surrogate Student...")
    
    # 1. Initialize Student
    # Assume we peak at dataloader to get shapes
    sample_features, _, _, _ = next(iter(dataloader))
    window_size = sample_features.shape[1]
    num_features = sample_features.shape[2]
    
    student_model = FastSurrogateMLP(window_size=window_size, num_features=num_features).to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    teacher_lax_model.eval()
    student_model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for features, kpd, cycle_t, target_soh in tqdm(dataloader, desc=f"Distillation Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            cycle_t = cycle_t.to(device)
            
            # --- TEACHER PREDICTION (Slow but accurate) ---
            with torch.no_grad():
                last_x = features[:, -1, :3]
                try:
                    # Pass through heavy inner optimization
                    teacher_u2 = teacher_lax_model(x=last_x, t=cycle_t, epoch=1000, return_f=False)
                except Exception:
                    teacher_u2 = teacher_lax_model(x=last_x, t=cycle_t, epoch=1000)
                    
                if isinstance(teacher_u2, tuple): teacher_u2 = teacher_u2[0]

            # --- STUDENT PREDICTION (Fast) ---
            optimizer.zero_grad()
            student_u2 = student_model(features)
            
            # --- DISTILLATION LOSS ---
            # MSE between Student prediction and Teacher's "soft labels"
            loss = F.mse_loss(student_u2.view(-1), teacher_u2.view(-1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Distillation Loss: {epoch_loss/len(dataloader):.6f}")
        
    return student_model
