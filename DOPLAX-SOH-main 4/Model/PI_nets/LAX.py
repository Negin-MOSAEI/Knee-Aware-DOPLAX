
import torch 
from torch import nn
import random  
import math 
import os 
from pathlib import Path 
from copy import deepcopy 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm.auto import tqdm                                                            
import pandas as pd 
import argparse
from Model.Auxiliary_nets.MLP import MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import grad
from contextlib import nullcontext

x_dim, y_dim = 16, 16 


  
# =====================
# Early Stopper for OptimizationNet class
# =====================
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, ckpt_dir="checkpoints"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stop       = False
        self.best_state = None
        self.bestepoch = 0

        # --- set up the folder in a cross-platfor0 way -------------
        self.ckpt_dir   = Path(ckpt_dir)    
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)  # mkdir -p
        self.ckpt_path  = self.ckpt_dir / "best_weights.pth"

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch, opt_net_state_dict):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            # --------------------------------------------------------
            # 1) grab the state_dict
            self.best_state = deepcopy(model.state_dict())
            self.bestepoch = epoch
            # 2) save it to the path you chose
            torch.save({
            "epoch": epoch,
            "model_state":  self.best_state,
            "optimizer_state": opt_net_state_dict,
            "val_loss": val_loss,
        },  self.ckpt_path)
            # --------------------------------------------------------
            print(f" New best -> saved to {self.ckpt_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True



# =====================
# MAPE loss function 
# =====================
def mape_loss_fn(outputs, targets, eps=1e-6):
    # we use eps to avoid diving by zeor
    return torch.mean(torch.abs((targets-outputs)/(targets+eps)))



# =====================
# schedule warmup for our model's learning rate 
# =====================
def cosine_annealing(epoch, warmup_epochs, restart_period, min_lr, initial_lr):
    """Cosine annealing schedule with warmup and periodic restarts"""
    # Warmup phase
    if epoch < warmup_epochs:
        return initial_lr * (epoch / warmup_epochs)
    
    # Cosine annealing with restarts
    progress = (epoch - warmup_epochs) % restart_period
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress / restart_period))



# =====================
# this function is an equivalent of theta (1/t)
# =====================
class theta_inverser(nn.Module):
    def __init__(self):
        super(theta_inverser, self).__init__()
    def forward(self, x):
        return 1.0 / (x + 1e-8)


# =====================
# transformer blocks to make our network much complex to test it 
# =====================
class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_out, hidden_dim=16, ff_dim=64, num_layers=1, nhead=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU()
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)           # (B, 1, H)
        x = x + self.pos_embedding                    # (B, 1, H)
        x = self.transformer(x).squeeze(1)            # (B, H)
        return self.output_proj(x)                    # (B, d_out)
# =====================
#  Lp Distance  
# =====================

class LpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__()
        self.p = p
        
    def forward(self, x, y):
        diff = torch.abs(x - y)
        dist = torch.pow(torch.sum(diff ** self.p, dim=-1), 1.0 / self.p)
        return dist
# =====================
# Mahalanobis distance blolck  
# =====================

class LearnableDistance(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        # Learnable transformation for Mahalanobis distance
        self.L = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
    def forward(self, x, y):
        ex = self.encoder(x)
        ey = self.encoder(y)
        diff = ex - ey
        W = self.L.T @ self.L
        # d^2 = diff^T W diff
        dist2 = torch.sum(diff @ W * diff, dim=-1)
        return torch.sqrt(dist2 + 1e-6)  # add epsilon for stability
    

# =====================
# this class is specialized to model the distance block,
#  because as know, distance is a symmetic function. 
# =====================

class SumProductNetwork(nn.Module):
    def __init__(self, d_in, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 64)
        self.ln1 = nn.LayerNorm(64)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(64, h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.act2 = nn.ReLU()


    def forward(self, x, y):
        sum_xy = x + y
        prod_xy = x * y
        z = torch.cat([sum_xy, prod_xy], dim=-1)  # shape: (batch, 2 * d_in)
        z = self.fc1(z)
        z = self.ln1(z)
        z = self.act1(z)
        z = self.fc2(z)
        z = self.ln2(z)
        z = self.act2(z)
        return z


def beta_scheduler_linear(epoch, initial_beta, min_beta=0.0, growth_period=300, mode="triangular"):
    """
    Returns a beta value that starts at min_beta and increases linearly
    to initial_beta over growth_period epochs.
        epoch (int): The current epoch number.
        initial_beta (float): The maximum beta value to reach.
        min_beta (float): The minimum beta value to start from.
        growth_period (int): The number of epochs over which the beta value grows.
        mode (str): The scheduling mode.
            - "linear": Beta increases linearly to initial_beta and stays there.
            - "triangular": Beta oscillates between min_beta and initial_beta.
    Returns:    float: The calculated beta value for the current epoch.
    """
    if mode == "triangular":
        cycle_len = growth_period
        t_in_cycle = epoch % (2 * cycle_len)
        if t_in_cycle < cycle_len:
            frac = float(t_in_cycle) / max(1.0, cycle_len)
        else:
            frac = 1.0 - float(t_in_cycle - cycle_len) / max(1.0, cycle_len)
        return min_beta + (initial_beta - min_beta) * frac
    
    elif mode == "linear":  
        t = max(0, epoch)
        frac = min(1.0, float(t) / max(1.0, growth_period))
        return min_beta + (initial_beta - min_beta) * frac

class Integrator(nn.Module):
    def __init__(self, phi_network, n_steps=600):
        super().__init__()
        self.phi = phi_network  # your phi network
        self.n_steps = n_steps
    def forward(self, h_in, t_u):
        # Integration range
        t_min = torch.zeros_like(t_u)
        t_max = t_u
        # Time steps for numerical integration
        t_steps = torch.linspace(0, 1, self.n_steps).unsqueeze(0).to(t_u.device)
        t_integral_points = t_min.unsqueeze(-1) + (t_max.unsqueeze(-1) - t_min.unsqueeze(-1)) * t_steps
        # Expand h_in to match the number of integration points
        h_in_expanded = h_in.unsqueeze(1).expand(-1, self.n_steps, -1)
        # Prepare input for the phi network
        phi_input = torch.cat([h_in_expanded, t_integral_points.unsqueeze(-1)], dim=-1)
        # Evaluate phi network at all integration points
        phi_integral_out = self.phi(phi_input.view(-1, phi_input.shape[-1]))
        phi_integral_out = phi_integral_out.view(h_in.shape[0], self.n_steps, -1)
        # Trapezoidal rule
        integral_sum = torch.sum(phi_integral_out[:, 1:-1, :], dim=1) + \
                       0.5 * (phi_integral_out[:, 0, :] + phi_integral_out[:, -1, :])
        # Multiply by step size
        step_size = (t_max - t_min) / (self.n_steps - 1)
        tH = step_size.unsqueeze(-1) * integral_sum
        return tH


# =====================
# Main Model(Optmization network)
# =====================
class OptimizationNetwork(nn.Module): 
    def __init__(self, x_sts, args, x_dim, y_dim):
        super().__init__() 
        self.random_seed = random.randint(1, 10000) # this line of code is just for keep the y input identical when we aren't on y_opt step.
        self.X_mean = x_sts[0].to(device='cpu').to(device).to(torch.float32) 
        self.X_std = x_sts[1].to(device='cpu').to(device).to(torch.float32) 
        self.x_dim = x_dim 
        self.y_dim = y_dim 
        self.d_in = x_dim + y_dim
        self.all_epoch = args.epochs                # TODO what's this argument? is it for combination? 
        self.epoch_th = args.epoch_th_LAX
        self.current_lr_y = getattr(args, 'lr_y', 0.01)
        self.batch_size = args.batch_size
        self.theta = args.theta_LAX # it stores the weight of mape loss in the total loss
        self.zeta = args.zeta_LAX # it stores the weight of mse loss in the total loss
        self.kata= args.kata_LAX # it stores the weight of mae loss in the total loss
        self.alpha = args.betha_LAX # it stores the weight of monotonoc loss in compare with data_loss
        self.dual = args.dual_LAX # it stores the weight of dual loss in the total loss
        self.beta = args.beta_LAX # it stores the weight of pde loss in the lax network if we wanna use it. 
        self.center_block_LAX = args.center_block_LAX 
        self.distance_block_LAX = args.distance_block_LAX
        self.time_block_LAX = args.time_block_LAX
        self.h_dim = args.h_dim_LAX
        self.run_for_LAX = args.run_for_LAX
        self.opt_y_flag = False
        self.args = args 
        self.epoch_y = args.epoch_y_LAX
        self.s_LAX = args.s_LAX 
        
        self.y = nn.Parameter(torch.empty(args.batch_size, y_dim, device='cpu').to(device), requires_grad=False)
        
        self.dynamical_F = MLP(input_dim=2 * x_dim + 3,output_dim=1,
                               layers_num=3,
                               hidden_dim=25,
                               dropout=0.2).to(device)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 16)
        )

        if self.distance_block_LAX == 'SumProductNetwork':
            self.d = SumProductNetwork(d_in=self.d_in, h_dim=self.h_dim)

        elif self.distance_block_LAX == 'MLP':
            self.d = self.build_mlp(
                input_dim=self.d_in,
                layer_dims=self.args.inside_distance_block_MLP_layers + [self.h_dim],
                output_dim=None,                
                activations=['silu', 'relu'],
                use_layernorm=[True, True]
            )

        elif self.distance_block_LAX == 'Mahalanobis':
            self.d = LearnableDistance(dim=16, embed_dim=16)
        
        elif self.distance_block_LAX == 'Lp_norm':
            self.d = LpDistance(p= args.norm)


        if args.time_block_LAX == '1/t':
            self.theta = theta_inverser()
            
        elif args.time_block_LAX == 'theta':
            self.theta = self.build_mlp(
                            input_dim = 1, 
                            layer_dims =self.args.inside_theta_layers + [self.h_dim], 
                            output_dim=None, 
                            activations=['silu', 'none'], 
                            use_layernorm=[True, True]
                            )


        elif args.time_block_LAX == 'multi_var_theta':
            self.theta = self.build_mlp(
                            input_dim = x_dim + 1, 
                            layer_dims= self.args.inside_multivar_theta_layers + [self.h_dim], 
                            output_dim=None, 
                            activations=['silu', 'none'], 
                            use_layernorm=[True, True]
                            )


        self.H_star = self.build_mlp(
            input_dim = self.h_dim,
            layer_dims = args.inside_h_star_layers,
            # output_dim = self.args.H_out,
            output_dim = self.args.h_out_LAX,
            activations = ['silu', 'silu'],
            use_layernorm = [True, True]
        )
        
        self.phi = self.build_mlp(
            input_dim=self.h_dim + 1,
            layer_dims=self.args.inside_phi_layers,
            output_dim=self.args.phi_out_LAX,
            activations=['relu', 'relu'],
            use_layernorm=[True, True]
        )
        # if self.args.center_block_LAX == 'PhiIntegrator':
        self.integrator = Integrator(self.phi)


        # self.H_star = TransformerBlock(d_in=5, d_out=1)
        self.g = self.build_mlp(
            input_dim=self.args.g_dim,
            layer_dims=self.args.inside_g_layers,
            output_dim=self.args.g_out_LAX,
            activations=['silu'],
            use_layernorm=[True]
        )
        
        if self.args.center_block_LAX == 'H*':
            input_dim_s = self.args.g_out_LAX + self.args.h_out_LAX
        elif self.args.center_block_LAX == 'PhI':
            input_dim_s = self.args.g_out_LAX + self.args.phi_out_LAX
        else:
            raise ValueError(f"Unknown center_block!!!")

        self.s_MLP = self.build_mlp(
            input_dim=input_dim_s,
            layer_dims=self.args.inside_S_MLP_layers,
            output_dim=self.args.dim_output_LAX,
            activations=['silu', 'silu', 'silu'],
            use_layernorm=[True, False, False]
        )
        self.s_Transformer = TransformerBlock(2, 1)
        self.betan = self.build_mlp(
            input_dim=1,
            layer_dims=self.args.inside_betan_layers,
            output_dim=1,
            activations=['relu', 'relu'],
            use_layernorm=[True, False]
        )



    def dual_consistency_loss(self, p_batch, t_batch= None,num_inner_samples = 10000):
        """
        Efficient dual consistency loss without nested optimization
        using sampling and matrix operations.
        
        Args:
            p_batch: Input to H* (h_in tensor) (shape [B, d])
            t_batch: Corresponding time tensor for phi (shape [B]) 
            num_inner_samples: Number of samples for approximation
        """
        B, d = p_batch.shape
        device = p_batch.device
        
        # Return zero if not enough samples
        if B < 2 or num_inner_samples <= 0:
            return torch.tensor(0.0, device='cpu').to(device)
        
        # Step 1: Sample v and q from the batch
        indices_v = torch.randint(0, B, (num_inner_samples,), device='cpu').to(device)
        v_batch = p_batch[indices_v]  # [S, d]
        
        indices_q = torch.randint(0, B, (num_inner_samples,), device='cpu').to(device)
        q_batch = p_batch[indices_q]  # [S, d]


        if self.args.center_block_LAX in ('PhI', 'PhiIntegrator'):
            assert t_batch is not None, "t_batch must be provided when using PhI."
            t_v = t_batch[indices_v].unsqueeze(-1)  # [S, 1]
            t_q = t_batch[indices_q].unsqueeze(-1)  # [S, 1]
            t_p = t_batch.unsqueeze(-1)             # [B, 1]


        # Step 2: Compute H* for v batch
        if self.args.center_block_LAX == 'H*':
            H_v = self.H_star(v_batch).squeeze()  # [S]

        elif self.args.center_block_LAX in ('PhI', 'PhiIntegrator'):
            # print(v_batch.shape, t_v.shape)
            v_input = torch.cat([v_batch, t_v.view(-1, 1)], dim=-1)  # [S, d+1]
            H_v = self.phi(v_input).squeeze()            # [S]        
   
        # Step 3: Compute <q_j, v_i> for all pairs
        A = torch.mm(q_batch, v_batch.t())  # [S, S]
        
        # Step 4: Compute inner supremum T(q_j) = max_i [<q_j, v_i> - H*(v_i)]
        T = (A - H_v.unsqueeze(0)).max(dim=1).values  # [S]
        
        # Step 5: Compute <p_i, q_j> for all i,j
        M = torch.mm(p_batch, q_batch.t())  # [B, S]
        
        # Step 6: Compute biconjugate: (H*)**(p_i) = max_j [<p_i, q_j> - T(q_j)]
        biconjugate = (M - T.unsqueeze(0)).max(dim=1).values  # [B]
        
        # Step 7: Compute H*(p_batch) or phi(p_batch, t_batch)
        if self.args.center_block_LAX == 'H*':
            H_p = self.H_star(p_batch).squeeze()  # [B]
        
        elif self.args.center_block_LAX in ('PhI', 'PhiIntegrator'):
            p_input = torch.cat([p_batch, t_p.view(-1, 1)], dim=-1)  # [B, d+1]
            H_p = self.phi(p_input).squeeze()           # [B]
    
        # Step 8: Loss = E[(H*(p) - biconjugate)^2] or abs
        loss = torch.mean(abs(H_p - biconjugate) )
        return loss


    def optimize_y(self, x, t, steps, lr = None):
        if lr is None: 
            lr = self.current_lr_y


        with torch.enable_grad():
            
            gen = torch.Generator(device='cpu')
            gen.seed()   #TODO : each time will give a deferent number ?
            noise = torch.randn(x.shape[0], y_dim, #TODO syntax
                                generator=gen,
                                device='cpu').to(device)
            y = noise * self.X_std + self.X_mean

            self.y = nn.Parameter(y)
            self.set_mode('y')
            optimizer = torch.optim.Adam([self.y], lr=lr)
            for step in range(steps):
                optimizer.zero_grad()
                out_y = self.net(x, self.y, t) #TODO squeeze ?
                loss = out_y.mean()  # maximize f → minimize -f
                loss.backward()
                optimizer.step()

        self.set_mode('net')
        return loss.squeeze(-1)


    def set_mode(self, mode: str = 'net'):
        """Switch between optimizing y or network weights."""
        assert mode in ('y', 'net')
        self.mode = mode
        for name, param in self.named_parameters():     
            param.requires_grad = (name == 'y') if mode == 'y' else (name != 'y')
    

    def net(self, x_u, y_u, t_u, return_F = False, return_u_t = False):

        t_u = t_u.requires_grad_(True)
        x_u = x_u.requires_grad_(True)    
        # print("t_u.requires_grad:", t_u.requires_grad, "t_u.grad_fn:", t_u.grad_fn)
        
        encoded_x = self.encoder(x_u)

        if self.args.distance_block_LAX == 'MLP':
            d_xy = self.d(torch.cat([x_u, y_u], dim=-1)) #TODO Sepreate componets of x_u, y_u as two input for self.theta. ---> DeepONet may can help to this section. 

        elif self.args.distance_block_LAX == "Mahalanobis":
            d_xy = self.d(x_u, y_u).unsqueeze(dim =1)

        elif self.args.distance_block_LAX == "SumProductNetwork":
            d_xy = self.d(x_u, y_u)
        
        elif self.args.distance_block_LAX == "Lp_norm":
            d_xy = self.d(x_u, y_u).unsqueeze(dim = 1)
        
        # d_xy = self.dx(x_u) * self.dy(y_u) ## TODO
        
        if self.args.time_block_LAX == 'multi_var_theta':
            theta_t = self.theta(torch.cat([x_u, t_u.unsqueeze(dim = 1)], dim = 1))
        else:
            theta_t = self.theta(t_u.unsqueeze(1))

        h_in = d_xy * theta_t ## TODO
        # h_in = torch.bmm(d_xy.unsqueeze(1), theta_t.unsqueeze(2)).squeeze(-1)

        if self.args.center_block_LAX == 'PhI':
            phi_input = torch.cat([h_in, t_u.unsqueeze(-1)], dim=-1)
            tH = self.phi(phi_input)
        elif self.args.center_block_LAX == 'H*':
            H_out = self.H_star(h_in)
            tH = t_u.unsqueeze(-1) * H_out
        elif self.args.center_block_LAX == 'PhiIntegrator':
            tH = self.integrator(h_in, t_u)

        g_out = self.g(y_u)

        # Concatenate tH and g_out ## TODO for a new network I add a simple network s to learn maybe harder network than 'sum' 
        s_input = torch.cat([tH, g_out], dim=-1)
        if self.s_LAX == 'ordinary_sum':
            u = (g_out + tH).unsqueeze(-1)

        elif self.s_LAX == 'MLP':    
            u = self.s_MLP(s_input)

        elif self.s_LAX == 'Transformer':
            u= self.s_Transformer(s_input)

        # store h_in for dual loss calculation 
        self.last_h_in = h_in 

        # print(f"u.sum().requires_grad: {u.sum().requires_grad}") # Should be True
        # breakpoint()
        u_t = grad(u.sum(),t_u,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        if u_t is None:
            raise RuntimeError("Grad w.r.t. t_u is None.")

        u_x = grad(u.sum(),x_u,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        if u_x is None:
            raise RuntimeError("Grad w.r.t. x_u is None.")
                    
        # t_u = t_u.requires_grad_(False)
        # x_u = x_u.requires_grad_(False) # TODO check it to review the flow of turn on and torn of the gradient tracking of x_u, t_u 

        # print(f"x_u: {x_u.shape}, t_u: {t_u.shape}, u: {u.shape}, u_x: {u_x.shape}, u_t: {u_t.shape}")

        F = self.dynamical_F(torch.cat([x_u, t_u.unsqueeze(dim = 1) ,u ,u_x ,u_t.unsqueeze(dim=1)],dim=1))
        f = u_t - F
        
        if return_F and not return_u_t:
            return u, f
        if not return_F and return_u_t:
            return u, u_t
        if return_F and return_u_t:
            return u, f, u_t
        return u

    def forward(self, return_u_t = False, x = None, t = None, epoch = 0, return_f =False):
        if x is None or t is None:
            raise ValueError("Provide x and t in 'net' mode")
        x_u = x.to(device).float()
        t_u = t.to(device).view(-1).float()

        if self.run_for_LAX:
            if epoch >= self.epoch_th:
                out_y = self.optimize_y(x_u.detach(), t_u.detach(), steps=self.epoch_y, lr=None)
                self.opt_y_flag = True
            else:
                self.opt_y_flag = False
                fixed_ys = self.X_mean.unsqueeze(0).repeat(x.shape[0], 1)
                self.y = nn.Parameter(fixed_ys, requires_grad=False)
                # print(self.y.shape)

        else:
            if self.opt_y_flag:
                epoch = self.epoch_th
            else:
                fixed_ys = self.X_mean.unsqueeze(0).repeat(x.shape[0], 1)
                self.y = nn.Parameter(fixed_ys, requires_grad=False)
                
            if epoch >= self.epoch_th:
                out_y = self.optimize_y(x_u.detach(), t_u.detach(), steps=self.epoch_y, lr=None)
                for name, param in self.named_parameters():
                    param.requires_grad = False
                    
        with torch.enable_grad(): ## TODO this line addedx to solve the problem of no_grad in validation phase. 
            final_out, f, u_t= self.net(x_u, self.y, t_u, return_F= True, return_u_t= True)

        if return_f and not return_u_t: 
            return final_out, f
        if return_f and return_u_t:
            return final_out, f, u_t
        if not return_f and return_u_t:
            return final_out, u_t
        return final_out 

    @staticmethod
    def build_mlp(
        input_dim: int,
        layer_dims: list[int],
        output_dim: int = 1,
        activations=None,
        use_layernorm=None,
        ):
        activation_map = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'none': nn.Identity
        }
        num_layers = len(layer_dims)
        if activations is None:
            activations = ["silu"] * num_layers
        if use_layernorm is None or use_layernorm is False:
            use_layernorm = [False] * num_layers
        elif use_layernorm is True:
            use_layernorm = [True] * num_layers
        elif isinstance(use_layernorm, list):
            if len(use_layernorm) != num_layers:
                raise ValueError("Length of 'use_layernorm' must match number of layers.")
        else:
            raise TypeError("use_layernorm must be bool or list[bool].")
        
        layers = []
        in_dim = input_dim
        for next_dim, act_name, ln_flag in zip(layer_dims, activations, use_layernorm):
            layers.append(nn.Linear(in_dim, next_dim))
            if ln_flag:
                layers.append(nn.LayerNorm(next_dim))
            act_class = activation_map.get(act_name, nn.Identity)
            layers.append(act_class())
            in_dim = next_dim
        # output layer
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)





def run_epoch(model, opt_net, opt_F, args, batch_name, experiment_id, epoch=0, phase="train", dataloader=None):
    if dataloader is None: 
        raise ValueError("DataLoader must be provided to run_epoch")
    
    epoch_loss_mse = 0.0
    epoch_loss_mape = 0.0
    epoch_loss_mae = 0.0
    epoch_loss_dual = 0.0
    epoch_loss_pde = 0.0 

    # Define log file path 
    # For example, add args.log_file = 'training_log.txt' when you define args
    log_folder = f'{args.results_path}/{batch_name}-Experiments/'  # Assuming args.save_folder exists      
    log_filename = "training_metrics.txt" # Or use args.log_filename
    log_file_path = Path(os.path.join(log_folder, log_filename))
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    current_beta = beta_scheduler_linear(
            epoch=epoch,
            initial_beta=args.beta_LAX,
            min_beta=0.0,
            growth_period=100  
        )
        
    for iter, (x1, x2, y1, y2) in enumerate(dataloader):
        x1, y1 = x1.to(device), y1.to(device)
        x2, y2 = x2.to(device), y2.to(device)
        
        x1, x2 = x1[:, :-1], x2[:, :-1]
        t1, t2 = x1[:, -1],  x2[:, -1]
        
        out_1, f1, u_t_1 = model(x=x1, t=t1, epoch=epoch, return_f=True, return_u_t= True)
        out_1 = out_1.squeeze(-1)
        h_in1 = model.last_h_in 

        out_2, f2, u_t_2 = model(x=x2, t=t2, epoch=epoch, return_f=True, return_u_t= True)
        out_2 = out_2.squeeze(-1)
        h_in2 = model.last_h_in 

        u_x_1 = grad(out_1.sum(),x1,
                create_graph=True,
                only_inputs=True,
                allow_unused=True)[0]
        u_x_2 = grad(out_2.sum(),x2,
                create_graph=True,
                only_inputs=True,
                allow_unused=True)[0]
        
        t1 = t1.requires_grad_(True)
        t2 = t2.requires_grad_(True)
        x1 = x1.requires_grad_(True)
        x2 = x2.requires_grad_(True)
        
        if u_x_1 is None:
            raise RuntimeError("Grad w.r.t. x1 is None.")                    
        elif u_x_2 is None:
            raise RuntimeError("Grad w.r.t. x2 is None.")
        
        t_batch1 = None if args.time_block_LAX == 'H*' else t1
        t_batch2 = None if args.time_block_LAX == 'H*' else t2


        # data loss
        mse_loss_fn = nn.MSELoss() 
        mae_loss_fn = nn.L1Loss()

        ctx = torch.no_grad() if phase != 'train' else nullcontext()
        with ctx:
            loss_data_mse = 0.5 * mse_loss_fn(out_1,y1.squeeze()) + 0.5 * mse_loss_fn(out_2,y2.squeeze())
            loss_data_mape= 0.5 * mape_loss_fn(out_1,y1.squeeze()) + 0.5 * mape_loss_fn(out_2,y2.squeeze())
            loss_data_mae= 0.5 * mae_loss_fn(out_1,y1.squeeze()) + 0.5 * mae_loss_fn(out_2,y2.squeeze())
            loss_dual = 0 if args.dual_LAX == 0 else 0.5 * (model.dual_consistency_loss(p_batch=h_in1, t_batch= t_batch1) + 
                                                        model.dual_consistency_loss(p_batch=h_in2, t_batch= t_batch2))


        # I changed it to the folloing lines because I didn't want to track gradient during validation phase. # TODO 
        # loss_data_mse = 0.5 * mse_loss_fn(out_1,y1.squeeze()) + 0.5 * mse_loss_fn(out_2,y2.squeeze())            loss_data_mse = 0.5 * mse_loss_fn(out_1,y1.squeeze()) + 0.5 * mse_loss_fn(out_2,y2.squeeze())
        loss_data_mape= 0.5 * mape_loss_fn(out_1,y1.squeeze()) + 0.5 * mape_loss_fn(out_2,y2.squeeze())
        loss_data_mae= 0.5 * mae_loss_fn(out_1,y1.squeeze()) + 0.5 * mae_loss_fn(out_2,y2.squeeze())
        f_target = torch.zeros_like(f1)
        loss_data_pde = 0.5 * mse_loss_fn(f1, f_target) + 0.5 * mse_loss_fn(f2, f_target)
        loss_dual = 0 if args.dual_LAX == 0 else 0.5 * (model.dual_consistency_loss(p_batch=h_in1, t_batch= t_batch1) + 
                                                    model.dual_consistency_loss(p_batch=h_in2, t_batch= t_batch2))
   
        used_beta = model.beta if args.schedule_beta == 0 else current_beta
        # print(used_beta)
        if phase=='train':
            relu = nn.ReLU()
            # physics loss  u2-u1<0, considering capacity regeneration(monotonic loss)
            loss_mono = relu(torch.mul(out_2 - out_1,y1.squeeze() - y2.squeeze())).sum()
            # total loss
            loss = (args.zeta_LAX * ((loss_data_mse)) 
                    + args.betha_LAX * loss_mono + 
                    args.theta_LAX * loss_data_mape + 
                    args.kata_LAX * loss_data_mae +
                    args.dual_LAX * loss_dual +
                    used_beta  * loss_data_pde)
            
            opt_net.zero_grad()
            opt_F.zero_grad()
            loss.backward()
            opt_net.step()
            opt_F.step()
        epoch_loss_mse += loss_data_mse.item()
        epoch_loss_mape += loss_data_mape.item()
        epoch_loss_mae += loss_data_mae.item() 
        epoch_loss_pde += loss_data_pde.item() 

        if args.dual_LAX != 0: 
            epoch_loss_dual += loss_dual.item()
        else:
            epoch_loss_dual += 0
    # Calculate average losses
    avg_loss_mse = epoch_loss_mse / len(dataloader)
    avg_loss_rmse = math.sqrt(avg_loss_mse)
    avg_loss_mape = epoch_loss_mape / len(dataloader)
    avg_loss_mae = epoch_loss_mae / len(dataloader)
    avg_loss_dual = epoch_loss_dual / len(dataloader)
    avg_loss_pde = epoch_loss_pde / len(dataloader)

    log_string = (
        f"\n {phase} loss mse on epoch {epoch + 1} : {avg_loss_mse : .6f}\n"
        f"\n {phase} loss rmse on epoch {epoch + 1} : {avg_loss_rmse : .6f}\n"
        f"\n {phase} loss mape on epoch {epoch + 1} : {avg_loss_mape : .6f}\n"
        f"\n {phase} loss mae on epoch {epoch + 1} : {avg_loss_mae : .6f}\n"
        f"\n {phase} loss pde on epoch {epoch + 1} : {avg_loss_pde : .6f}\n"
        f"\n {phase} loss dual consistency on epoch {epoch + 1} : {avg_loss_dual : .8f}\n"
    )

    print(log_string)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, 'a') as f:
        f.write(log_string)
        
    return (avg_loss_mse, avg_loss_rmse, avg_loss_mape, avg_loss_mae, avg_loss_dual, avg_loss_pde)




def plot(args, filename, Y1, X1, Y2=None, X2=None, Y1_name="", Y2_name="", X_name="X"):
        x_axis = np.arange(len(Y1)) # Use length of data for axis      
            
        # Plot      
        plt.figure(figsize=(12, 6))  
        line_thickness = 0.8 

        if X1:  
            plt.plot(X1, Y1, label=Y1_name, linewidth=line_thickness)   
        else:
            plt.plot(x_axis, Y1, label=Y1_name, linewidth=line_thickness)
        if Y2:
            if X2:  
                plt.plot(X2, Y2, label=Y2_name, linewidth=line_thickness)   
            else:
                plt.plot(x_axis, Y2, label=Y2_name, linewidth=line_thickness)

            plt.title(f'{Y1_name} vs. {Y2_name}') 
        else:
            plt.title(f'{Y1_name}')   

        plt.xlabel(X_name)      
        plt.ylabel('Value')      
             
        plt.legend()      
        plt.grid(True)      
                 
        filename = f"{args.results_path}/{filename}.png"      
        plt.savefig(filename, dpi=300)   
        plt.close()   


def run_training_and_evaluation(args, dataloader, batch_name, experiment_id, return_val_loss=False):
    if args.data == 'XJTU':
        args.betha_LAX = args.betha_LAX_XJTU
        args.dual_LAX = args.dual_LAX_XJTU
        args.theta_LAX = args.theta_LAX_XJTU
        args.lr_net = args.lr_net_LAX_XJTU
        args.h_dim_LAX = args.h_dim_LAX_XJTU
        args.beta_LAX = args.beta_LAX_XJTU
        args.distance_block_LAX = args.distance_block_LAX_XJTU
        args.time_block_LAX = args.time_block_LAX_XJTU
        args.center_block_LAX = args.center_block_LAX_XJTU
        args.inside_distance_block_MLP_layers = args.inside_distance_block_MLP_layers_LAX_XJTU
        args.inside_theta_layers = args.inside_theta_layers_LAX_XJTU
        args.inside_multivar_theta_layers = args.inside_multivar_theta_layers_LAX_XJTU
        args.inside_h_star_layers = args.inside_h_star_layers_LAX_XJTU
        args.inside_phi_layers = args.inside_phi_layers_LAX_XJTU
        args.inside_g_layers = args.inside_g_layers_LAX_XJTU
        args.inside_S_MLP_layers = args.inside_S_MLP_layers_LAX_XJTU
        args.inside_betan_layers = args.inside_betan_layers_LAX_XJTU
        args.g_dim = args.g_dim_LAX_XJTU
        args.h_out_LAX = args.H_out_LAX_XJTU
        args.g_out_LAX = args.g_out_LAX_XJTU
        args.phi_out_LAX = args.phi_out_LAX_XJTU
        args.dim_output_LAX = args.dim_output_LAX_XJTU
        args.lr_F = args.lr_F_XJTU
 
    elif args.data == 'TJU':
        args.betha_LAX = args.betha_LAX_TJU
        args.dual_LAX = args.dual_LAX_TJU
        args.theta_LAX = args.theta_LAX_TJU
        args.lr_net = args.lr_net_LAX_TJU
        args.h_dim_LAX = args.h_dim_LAX_TJU
        args.beta_LAX = args.beta_LAX_TJU
        args.distance_block_LAX = args.distance_block_LAX_TJU
        args.time_block_LAX = args.time_block_LAX_TJU
        args.center_block_LAX = args.center_block_LAX_TJU
        args.inside_distance_block_MLP_layers = args.inside_distance_block_MLP_layers_LAX_TJU
        args.inside_theta_layers = args.inside_theta_layers_LAX_TJU
        args.inside_multivar_theta_layers = args.inside_multivar_theta_layers_LAX_TJU
        args.inside_h_star_layers = args.inside_h_star_layers_LAX_TJU
        args.inside_phi_layers = args.inside_phi_layers_LAX_TJU
        args.inside_g_layers = args.inside_g_layers_LAX_TJU
        args.inside_S_MLP_layers = args.inside_S_MLP_layers_LAX_TJU
        args.inside_betan_layers = args.inside_betan_layers_LAX_TJU
        args.g_dim = args.g_dim_LAX_TJU
        args.h_out_LAX = args.H_out_LAX_TJU
        args.g_out_LAX = args.g_out_LAX_TJU
        args.phi_out_LAX = args.phi_out_LAX_TJU
        args.dim_output_LAX = args.dim_output_LAX_TJU
        args.lr_F = args.lr_F_TJU

    elif args.data == 'MIT':
        args.betha_LAX = args.betha_LAX_MIT
        args.dual_LAX = args.dual_LAX_MIT
        args.theta_LAX = args.theta_LAX_MIT
        args.lr_net = args.lr_net_LAX_MIT
        args.h_dim_LAX = args.h_dim_LAX_MIT
        args.beta_LAX = args.beta_LAX_MIT
        args.distance_block_LAX = args.distance_block_LAX_MIT
        args.time_block_LAX = args.time_block_LAX_MIT
        args.center_block_LAX = args.center_block_LAX_MIT
        args.inside_distance_block_MLP_layers = args.inside_distance_block_MLP_layers_LAX_MIT
        args.inside_theta_layers = args.inside_theta_layers_LAX_MIT
        args.inside_multivar_theta_layers = args.inside_multivar_theta_layers_LAX_MIT
        args.inside_h_star_layers = args.inside_h_star_layers_LAX_MIT
        args.inside_phi_layers= args.inside_phi_layers_LAX_MIT
        args.inside_g_layers = args.inside_g_layers_LAX_MIT
        args.inside_S_MLP_layers = args.inside_S_MLP_layers_LAX_MIT
        args.inside_betan_layers = args.inside_betan_layers_LAX_MIT
        args.g_dim = args.g_dim_LAX_MIT
        args.h_out_LAX = args.H_out_LAX_MIT
        args.g_out_LAX = args.g_out_LAX_MIT
        args.phi_out_LAX = args.phi_out_LAX_MIT
        args.dim_output_LAX = args.dim_output_LAX_MIT
        args.lr_F = args.lr_F_MIT

    else:
        args.betha_LAX = args.betha_LAX_HUST
        args.dual_LAX = args.dual_LAX_HUST
        args.theta_LAX = args.theta_LAX_HUST
        args.lr_net = args.lr_net_LAX_HUST
        args.h_dim_LAX = args.h_dim_LAX_HUST
        args.beta_LAX = args.beta_LAX_HUST
        args.distance_block_LAX = args.distance_block_LAX_HUST
        args.time_block_LAX = args.time_block_LAX_HUST
        args.center_block_LAX = args.center_block_LAX_HUST
        args.inside_distance_block_MLP_layers = args.inside_distance_block_MLP_layers_LAX_HUST
        args.inside_theta_layers = args.inside_theta_layers_LAX_HUST
        args.inside_multivar_theta_layers = args.inside_multivar_theta_layers_LAX_HUST
        args.inside_h_star_layers = args.inside_h_star_layers_LAX_HUST
        args.inside_phi_layers = args.inside_phi_layers_LAX_HUST
        args.inside_g_layers = args.inside_g_layers_LAX_HUST
        args.inside_S_MLP_layers = args.inside_S_MLP_layers_LAX_HUST
        args.inside_betan_layers = args.inside_betan_layers_LAX_HUST
        args.g_dim = args.g_dim_LAX_HUST
        args.h_out_LAX = args.H_out_LAX_HUST
        args.g_out_LAX = args.g_out_LAX_HUST
        args.phi_out_LAX = args.phi_out_LAX_HUST
        args.dim_output_LAX = args.dim_output_LAX_HUST
        args.lr_F = args.lr_F_HUST

    # specifying the stats. 
    sum_ = 0.0
    sum_squared = 0.0
    n_samples = 0

    for x1, x2, _, _ in dataloader['train']:
        y_dim = x_dim = x1.shape[1] - 1
        x2 = x2[:, :x_dim]
        x1 = x1[:, :x_dim]
        sum_ += x2.sum(dim=0)  # sum across the batch (dim=0), result shape: [num_features]
        sum_squared += (x2 ** 2).sum(dim=0)
        n_samples += x2.size(0)  # batch_size
    X_mean = sum_ / n_samples
    X_std = torch.sqrt((sum_squared / n_samples) - (X_mean ** 2))

    model = OptimizationNetwork(
        x_sts=(X_mean, X_std),
        y_dim=y_dim,
        x_dim=x_dim,
        args= args
    ).to(device)


    # optimizers 
    params_F = list(model.dynamical_F.parameters())
    params_F_ids = {id(p) for p in params_F}
    params_net = [p for p in model.parameters() if id(p) not in params_F_ids]
    opt_net = torch.optim.Adam(params_net, lr=args.lr_net)


    opt_F = torch.optim.Adam(model.dynamical_F.parameters(),
            lr= args.lr_F)

    # =====================
    # Training Loop
    # =====================


    # set the random seed
    # torch.manual_seed(42)

    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, ckpt_dir=args.results_path)

    train_loss_mse_metric = []
    train_loss_rmse_metric = []
    train_loss_mape_metric = [] 
    train_loss_mae_metric = []
    train_loss_dual_metric = []
    train_loss_pde_metric = []


    val_loss_mse_metric = []
    val_loss_rmse_metric = []
    val_loss_mape_metric = [] 
    val_loss_mae_metric = []
    val_loss_dual_metric = []
    val_loss_pde_metric = []

    Epoch_list = []

    polt_th = min(args.plot_threshold, args.epoch_th_LAX * 2) #TODO

    # Store initial learning rates
    initial_lr_net = args.lr_net
    initial_lr_y = args.lr_y

    train_dataloader = dataloader['train']
    valid_dataloader = dataloader['valid']
    test_dataloader = dataloader['test']
    for epoch in tqdm(range(args.epoch_net)):
                
        # Update learning rates using cosine annealing
        current_lr_net = cosine_annealing(
            epoch, 
            args.warmup_epochs_net, 
            args.restart_period,
            args.min_lr_net, 
            initial_lr_net
        )
        current_lr_y = cosine_annealing(
            epoch, 
            args.warmup_epochs_y, 
            args.restart_period,
            args.min_lr_y, 
            initial_lr_y
        )
        
        # Update optimizers with new learning rates
        for param_group in opt_net.param_groups:
            param_group['lr'] = current_lr_net
        
        # Store the current lr_y for use in optimize_y method
        model.current_lr_y = current_lr_y

        Epoch_list.append(epoch)
        model.train()
        phase = 'train'
        train_loss_mse, train_loss_rmse, train_loss_mape, train_loss_mae, train_loss_dual, train_loss_pde = run_epoch(epoch=epoch, phase=phase, model= model, opt_F= opt_F,
                                                                                     opt_net= opt_net, args=args, dataloader=train_dataloader, batch_name= batch_name, experiment_id= experiment_id)
        train_loss_mse_metric.append(train_loss_mse)
        train_loss_rmse_metric.append(train_loss_rmse)
        train_loss_mape_metric.append(train_loss_mape) 
        train_loss_mae_metric.append(train_loss_mae)
        train_loss_dual_metric.append(train_loss_dual)
        train_loss_pde_metric.append(train_loss_pde)


        model.eval()
        phase = 'valid'

        val_loss_mse, val_loss_rmse, val_loss_mape, val_loss_mae, val_loss_dual, val_loss_pde = run_epoch(epoch=epoch, phase=phase, model= model, opt_F= opt_F,
                                                                                opt_net= opt_net, args= args, dataloader=valid_dataloader, batch_name= batch_name, experiment_id= experiment_id)
        val_loss_mse_metric.append(val_loss_mse)
        val_loss_rmse_metric.append(val_loss_rmse)
        val_loss_mape_metric.append(val_loss_mape) 
        val_loss_mae_metric.append(val_loss_mae)
        val_loss_dual_metric.append(val_loss_dual)
        val_loss_pde_metric.append(val_loss_pde)

        if (epoch % args.plot_update_period == 0) and (epoch > polt_th):
            plot(args, f"loss_mse", train_loss_mse_metric[polt_th:], Epoch_list[polt_th:], Y2 = val_loss_mse_metric[polt_th:] ,
            X2 = Epoch_list[polt_th:], Y1_name="train_mse", Y2_name="valid_mse" ,X_name="epoch")

            plot(args, f"loss_rmse", train_loss_rmse_metric[polt_th:], Epoch_list[polt_th:], Y2 = val_loss_rmse_metric[polt_th:] ,
            X2 = Epoch_list[polt_th:], Y1_name="train_rmse", Y2_name="valid_rmse" ,X_name="epoch")

            plot(args, f"loss_mape", train_loss_mape_metric[polt_th:], Epoch_list[polt_th:], Y2 = val_loss_mape_metric[polt_th:] ,
            X2 = Epoch_list[polt_th:], Y1_name="train_mape", Y2_name="valid_mape" ,X_name="epoch")
            
            plot(args, f"loss_mae", train_loss_mae_metric[polt_th:], Epoch_list[polt_th:], Y2 = val_loss_mape_metric[polt_th:] ,
            X2 = Epoch_list[polt_th:], Y1_name="train_mae", Y2_name="valid_mae" ,X_name="epoch")
            
            plot(args, f"loss_dual", train_loss_dual_metric[polt_th:], Epoch_list[polt_th:], Y2 = val_loss_dual_metric[polt_th:] ,
            X2 = Epoch_list[polt_th:], Y1_name="train_dual", Y2_name="valid_dual" ,X_name="epoch")
            

            plot(args, f"loss_pde", train_loss_pde_metric[polt_th:], Epoch_list[polt_th:], Y2 = val_loss_pde_metric[polt_th:] ,
            X2 = Epoch_list[polt_th:], Y1_name="train_pde", Y2_name="valid_pde" ,X_name="epoch")


        total_loss_valid = args.zeta_LAX * val_loss_mse + args.theta_LAX * val_loss_mape

        stopper(val_loss_mse, model, epoch, opt_net.state_dict()) # TODO check why doesn't exist symetry in this piece of code. thank you. 
        if stopper.stop:
            print(f"Early-stopped at epoch {epoch}")
            break
    
    # =====================
    # Test Evaluation After Training (using the best model state if early stopping occurred)
    # =====================    
    print(f"Loading best model from epoch {stopper.bestepoch} with validation loss: {stopper.best_loss:.6f}")
    model.load_state_dict(stopper.best_state)
    test_metrics = evaluate_on_set(model, test_dataloader, args, args.results_path, test= True, phase = 'test')
    val_metrics = evaluate_on_set(model, valid_dataloader, args, args.results_path, test= False, phase = 'validation')

    if return_val_loss: 
        return test_metrics, val_metrics
    return test_metrics


def evaluate_on_set(model, test_dataloader, args, experiment_folder_path,phase ,test= False, make_csv=False):
    """
    Evaluates the model on the test dataset and saves the metrics to a CSV.
    Also generates a capacity prediction plot for a subset of the test data.
    """
    print("\n--- Evaluating on Test Set ---")
    model.eval() # Set model to evaluation mode
    test_loss_mse = 0.0
    test_loss_mape = 0.0
    test_loss_mae = 0.0
    test_loss_dual = 0.0
    test_loss_pde = 0.0

    all_y_true = []
    all_predictions = []
    mse_loss_fn = nn.MSELoss(reduction='sum')
    mae_loss_fn = nn.L1Loss(reduction='sum')
    
    # with torch.no_grad():
    for x1, x2, y1_true, y2_true in test_dataloader:
        x1, y1_true = x1.to(device), y1_true.to(device)
        
        x1, x2 = x1[:, :-1], x2[:, :-1]
        t1, t2 = x1[:, -1], x2[:, -1]
        
        t_batch1 = None if args.time_block_LAX == 'H*' else t1
        # x2, t2, y2_true are not directly used for prediction, but kept for consistency if needed later
        # Predict y1 (current capacity)
        predictions, f = model(x=x1, t=t1.squeeze(), epoch=args.epoch_net, return_f = True) # Use epoch_net to trigger optimize_y if needed
        predictions = predictions.squeeze(-1)
        t1 = t1.requires_grad_(True)

        u_x_1 = grad(predictions.sum(),x1,
                create_graph=True,
                only_inputs=True,
                allow_unused=True)[0]
        
                    
        u_t_1 = grad(predictions.sum(),t1,
                create_graph=True,
                only_inputs=True,
                allow_unused=True)[0]
        
        ctx = torch.no_grad() if phase in ('validation', 'test') else nullcontext()
        with ctx:
            test_loss_mse += mse_loss_fn(predictions, y1_true.squeeze()).item()
            test_loss_mape += mape_loss_fn(predictions, y1_true.squeeze()).item() * (x1.size(0))
            test_loss_mae += mae_loss_fn(predictions, y1_true.squeeze()).item()
            test_loss_dual += model.dual_consistency_loss(p_batch = model.last_h_in, t_batch= t_batch1).item()



        f_target = torch.zeros_like(f)
        test_loss_pde += mse_loss_fn(f, f_target)


        all_y_true.append(y1_true.cpu())
        all_predictions.append(predictions.cpu())

    avg_test_mse = test_loss_mse / len(test_dataloader.dataset)
    avg_test_rmse = math.sqrt(avg_test_mse)
    avg_test_mape = test_loss_mape / len(test_dataloader.dataset)
    avg_test_mae = test_loss_mae / len(test_dataloader.dataset)
    avg_test_dual = test_loss_dual / len(test_dataloader.dataset)
    avg_test_pde = test_loss_pde / len(test_dataloader.dataset)


    if test: 
        print(f"Test MSE: {avg_test_mse:.6f}")
        print(f"Test RMSE: {avg_test_rmse:.6f}")
        print(f"Test MAPE: {avg_test_mape:.6f}")
        print(f"Test MAE: {avg_test_mae:.6f}")
        print(f"Test DUAL: {avg_test_dual: .8f}")
        print(f"Test PDE: {avg_test_pde: .6f}")

        def to_scalar(x):
            return x.item() if hasattr(x, "item") else x

        avg_test_mse = to_scalar(avg_test_mse)
        avg_test_rmse = to_scalar(avg_test_rmse)
        avg_test_mape = to_scalar(avg_test_mape)
        avg_test_mae = to_scalar(avg_test_mae)
        avg_test_dual = to_scalar(avg_test_dual)
        avg_test_pde = to_scalar(avg_test_pde)

        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAPE', 'MAE', 'DUAL', 'PDE'],
            'Value': [avg_test_mse, avg_test_rmse, avg_test_mape, avg_test_mae, avg_test_dual, avg_test_pde]
        })

        metrics_filename = os.path.join(experiment_folder_path, "test_metrics.csv")
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Test metrics saved to: {metrics_filename}")

        
        # Capacity Degradation Plot
        all_y_true_np = torch.cat(all_y_true).detach().cpu().numpy()
        all_predictions_np = torch.cat(all_predictions).detach().cpu().numpy()

        # --- save raw outputs to CSV ---
        if make_csv:
            outputs_df = pd.DataFrame({
                'y_true': all_y_true_np.flatten(),
                'y_pred': all_predictions_np.flatten()
            })

            outputs_df.to_csv(os.path.join(experiment_folder_path, "test_outputs.csv"), index=False)


        # Use a subset for plotting if the test set is very large
        plot_samples =len(all_y_true_np)
        plt.figure(figsize=(12, 6))
        plt.plot(all_y_true_np[:plot_samples], label='True Capacity')
        plt.plot(all_predictions_np[:plot_samples], label='Predicted Capacity')
        plt.xlabel('Cycle Index')
        plt.ylabel('Normalized Capacity')
        plt.legend()
        plt.title(f'Capacity Degradation Prediction for Batch: {args.batch}')
        plt.grid(True)
        plot_filename = os.path.join(experiment_folder_path, f"capacity_prediction_{args.batch}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Capacity prediction plot saved to: {plot_filename}")
        
    return {
        'mse': avg_test_mse,
        'rmse': avg_test_rmse,
        'mape': avg_test_mape,
        'mae': avg_test_mae,
        'dual': avg_test_dual, 
        'pde': avg_test_pde,
    }





def save_LAX_results(save_info):
    if save_info['type'] not in ['experiments', 'batches']:
        raise ValueError("type must be either experiments or batches")
    
    if save_info['type'] == 'experiments':
        if save_info['experiments_info'] is not None:
            # After all experiments for a batch are done, create a DataFrame of individual results
            batch_results_df = pd.DataFrame(save_info['experiments_info'])
            individual_results_csv_path = Path(f"{save_info['save_folder']}/{save_info['batch']}_Batch_summary/{save_info['batch']}_all_experiments_test_results.csv")
            individual_results_csv_path.parent.mkdir(parents=True, exist_ok=True)
            batch_results_df.to_csv(individual_results_csv_path, index=False)
            print(f"\nSaved individual test results for batch {save_info['batch']} to: {individual_results_csv_path}")

            # Calculate the mean of losses across these 10 experiments for the current batch
            mean_metrics = batch_results_df[['MSE', 'RMSE', 'MAPE', 'MAE', 'DUAL']].mean().to_dict()
            mean_metrics_df = pd.DataFrame([{'Metric': k, 'Mean_Value': v} for k, v in mean_metrics.items()])

            # Save the mean metrics to a new CSV file
            mean_results_csv_path = Path(f"{save_info['save_folder']}/results/{save_info['batch']}_Batch_summary/{save_info['batch']}_mean_test_results.csv")
            mean_results_csv_path.parent.mkdir(parents=True, exist_ok=True)
            # Corrected line: Call to_csv on the DataFrame, passing the path
            mean_metrics_df.to_csv(mean_results_csv_path, index=False) 
            print(f"Saved mean test results for batch {save_info['batch']} to: {mean_results_csv_path}")
        
            # Append the mean results of the current batch to the overall list
            save_info['batches_info'].append({
                'Batch': save_info['batch'],
                'MSE': mean_metrics['MSE'],
                'RMSE': mean_metrics['RMSE'],
                'MAPE': mean_metrics['MAPE'],
                'MAE': mean_metrics['MAE'],
                'DUAL': mean_metrics['DUAL']
            })
            return save_info['batches_info']
        else:
            raise ValueError("experiments_info must be not None when type is experiments")

    else:
        # After the loop, create the summary DataFrame and save it
        if save_info['batches_info']:
            summary_df = pd.DataFrame(save_info['batches_info'])
            # Define the path for the summary file
            summary_csv_path = Path(f"{save_info['save_folder']}/summary_of_all_batchs.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\n{'='*60}")
            print(f"Summary of all batchs saved to: {summary_csv_path}")
            print(f"{'='*60}")
        else:
            print("\nNo batch results to summarize.")


def load_model(model, model_path, freeze, batch_size):
    # Load checkpoint to GPU first
    checkpoint = torch.load(model_path, map_location=device)
    # set the y shape(which comes from loaded weights) args.batch_size of current DOPLAX model 
    checkpoint['model_state']['y'] = checkpoint['model_state']['y'][0].unsqueeze(0).repeat(batch_size, 1)

    model.load_state_dict(checkpoint["model_state"])
    for param in model.parameters():
        param.requires_grad = not freeze
