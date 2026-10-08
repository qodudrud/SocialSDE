import torch
import torch.nn as nn

import torch.nn.functional as F

def make_model(args, model_type):
    if model_type == 'drift':
        model = LBN_SWAG_Drift(args)
    elif model_type == 'diff':
        model = LBN_SWAG_Diff(args)
    else:
        raise ValueError("model_type should be 'drift' or 'diff'")

    return model


class ConditionalLayerNorm(nn.Module):
    """
    CLN(h, c) = (1 + gamma(c)) * LN(h) + beta(c), where gamma, beta are produced from condition c.
    """
    def __init__(self, hidden_dim: int, cond_dim: int, use_LN=True,  eps: float = 1e-5, affine_init_zero: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.use_LN = use_LN

        # We'll do LN without affine, then apply conditional affine afterwards
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps) if use_LN else nn.Identity()

        self.to_gamma_beta = nn.Linear(cond_dim, 2 * hidden_dim)

        if affine_init_zero:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h_hat = self.ln(h)  # normalized hidden
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        # (1+gamma) helps stability at init
        return (1.0 + gamma) * h_hat + beta


class CLNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, cond_dim, use_LN=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.cln = ConditionalLayerNorm(hidden_dim, cond_dim, use_LN=use_LN)
        self.act = nn.ELU(inplace=False)

    def forward(self, x, cond):
        h = self.fc(x)
        h = self.cln(h, cond)
        return self.act(h)



class LBN_SWAG_Drift(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device

        self.dim        = args.dim
        self.input_dim  = args.input_dim
        self.n_layer = args.n_layer
        self.n_hidden = args.n_hidden
        self.use_LN = getattr(args, "use_LN", False)

        # Option: args.n_cond_onehot (default: 0) - number of additional one-hot condition features to concatenate to input
        self.n_cond_onehot = getattr(args, "n_cond_onehot", 0)

        self.feature_dim = self.input_dim - self.n_cond_onehot

        self.blocks = nn.ModuleList()
        if self.n_cond_onehot > 0:
            for i in range(args.n_layer-1):
                in_dim = self.feature_dim if i == 0 else self.n_hidden
                self.blocks.append(
                    CLNBlock(in_dim,
                             self.n_hidden,
                             self.n_cond_onehot,
                             use_LN=self.use_LN)
                )
        else:
            for i in range(args.n_layer-1):
                in_dim = self.feature_dim if i == 0 else self.n_hidden
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(in_dim, self.n_hidden),
                        nn.LayerNorm(self.n_hidden) if self.use_LN else nn.Identity(),
                        nn.ELU(inplace=False)
                    )
                )
        
        self.output_layer = nn.Linear(self.n_hidden, self.dim)
        
        self._init_weights()
        

    def _init_weights(self):
        # Initialization: hidden - Kaiming, output - Xavier
        if self.n_cond_onehot > 0:
            for blk in self.blocks:
                nn.init.kaiming_normal_(blk.fc.weight, nonlinearity="relu")
                nn.init.zeros_(blk.fc.bias)
        else:
            for seq in self.blocks:
                lin = seq[0]
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)

        # output layer (Xavier)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim, f"x dim {x.shape[-1]} != input_dim {self.input_dim}"

        if self.n_cond_onehot > 0:
            cond_input = x[..., :self.n_cond_onehot]
            feature_input = x[..., self.n_cond_onehot:]

            out = feature_input
            for blk in self.blocks:
                out = blk(out, cond_input)
            return self.output_layer(out)
        
        else:
            out = x
            for blk in self.blocks:
                out = blk(out)
            return self.output_layer(out)


class LBN_SWAG_Diff(nn.Module):
    """
        Langevin Bayesian Network for Diffusion matrix (SWAG)
        Output: symmetric (or PSD) matrix of shape (..., dim, dim)
    """
    def __init__(self, args):
        super().__init__()
        self.device     = args.device

        self.dim        = args.dim
        self.input_dim  = args.input_dim
        self.n_layer    = args.n_layer
        self.n_hidden   = args.n_hidden
        self.use_LN     = getattr(args, "use_LN", False)

        # Option: args.n_cond_onehot (default: 0) - number of additional one-hot condition features to concatenate to input
        self.n_cond_onehot = getattr(args, "n_cond_onehot", 0)

        self.feature_dim = self.input_dim - self.n_cond_onehot

        # Option: args.ensure_psd (default: True), args.eps, args.use_softplus
        ensure_psd  = getattr(args, "ensure_psd", True)
        eps         = getattr(args, "eps", 1e-9)
        use_softplus = getattr(args, "use_softplus", True)

        self.ensure_psd = ensure_psd
        self.eps = eps
        self.use_softplus = use_softplus

        n_out = self.dim * (self.dim + 1) //2  # output dim for flat matrix

        # Hidden blocks
        self.blocks = nn.ModuleList()
        if self.n_cond_onehot > 0:
            for i in range(args.n_layer-1):
                in_dim = self.feature_dim if i == 0 else self.n_hidden
                self.blocks.append(
                    CLNBlock(in_dim,
                             self.n_hidden,
                             self.n_cond_onehot,
                             use_LN=self.use_LN)
                )
        else:
            for i in range(args.n_layer-1):
                in_dim = self.feature_dim if i == 0 else self.n_hidden
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(in_dim, self.n_hidden),
                        nn.LayerNorm(self.n_hidden) if self.use_LN else nn.Identity(),
                        nn.ELU(inplace=False)
                    )
                )
                        
        # Output d*(d+1)/2 entries; SymmetricMap constructs the full matrix.
        self.output_layer = nn.Linear(self.n_hidden, n_out)
        self.sym_map = SymmetricMap(self.dim, ensure_psd=ensure_psd, eps=eps, use_softplus=use_softplus)

        self._init_weights()

    def _init_weights(self):
        # Hidden: Kaiming
        if self.n_cond_onehot > 0:
            for blk in self.blocks:
                nn.init.kaiming_normal_(blk.fc.weight, nonlinearity="relu")
                if blk.fc.bias is not None:
                    nn.init.zeros_(blk.fc.bias)
        else:
            for seq in self.blocks:
                lin = seq[0]
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

        # Output: Xavier
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: (..., input_dim) where input_dim = feature_dim + n_cond_onehot (if n_cond_onehot > 0)
            returns: (..., dim, dim)
        """
        assert x.shape[-1] == self.input_dim, f"x dim {x.shape[-1]} != input_dim {self.input_dim}"

        if self.n_cond_onehot > 0:
            cond_input = x[..., :self.n_cond_onehot]
            feature_input = x[..., self.n_cond_onehot:]

            out = feature_input
            for blk in self.blocks:
                out = blk(out, cond_input)
        
        else:
            out = x
            for blk in self.blocks:
                out = blk(out)

        out = self.output_layer(out)
        return self.sym_map(out)
    

class SymmetricMap(nn.Module):
    """
        Map a d*(d+1)/2 vector to a symmetric matrix.
        If ensure_psd=True, transform eigenvalues with softplus + eps when
        use_softplus=True, or clamp them to at least eps otherwise.
    """
    def __init__(self, dim: int, ensure_psd: bool = True, eps: float = 1e-9, use_softplus: bool = True):
        super().__init__()
        self.dim = dim
        self.ensure_psd = ensure_psd
        self.eps = eps
        self.use_softplus = use_softplus

        self.tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        self.register_buffer('tril_row_idx', self.tril_indices[0])
        self.register_buffer('tril_col_idx', self.tril_indices[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        S = torch.zeros(*batch_shape, self.dim, self.dim, device=x.device, dtype=x.dtype)
        S[..., self.tril_row_idx, self.tril_col_idx] = x
        
        S = S + S.transpose(-1, -2) - torch.diag_embed(S.diagonal(dim1=-2, dim2=-1))
        
        if not self.ensure_psd:
            return S

        # PSD projection via eigenvalue nonlinearity (differentiable almost everywhere)
        evals, evecs = torch.linalg.eigh(S)  # symmetric -> real eigendecomp
        if self.use_softplus:
            evals_pos = F.softplus(evals) + self.eps
        else:
            evals_pos = torch.clamp(evals, min=self.eps)
        
        S_psd = evecs @ torch.diag_embed(evals_pos) @ evecs.transpose(-1, -2)

        if not self.training:
            diff = (S_psd - S_psd.transpose(-1, -2)).abs().max()
            min_diag = S_psd.diagonal(dim1=-2, dim2=-1).min()
            
            if diff > 1e-1 or min_diag < 0:
                print("===================================================")
                print("SymmetricMap BUG", diff.item(), min_diag.item())
                print("raw S example:", S_psd[0])
                print("====================================================")
        return S_psd
