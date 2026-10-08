import torch
import torch.nn as nn

def train_epoch(
        model, 
        train_loader, 
        optim, 
        model_type, 
        device=None,
        time_step = 0.01, 
        time_reg_lambda=0.0, 
        grad_clip=0.0,
        include_time=False, 
        label_noise = False
    ):
    """
        train the model for one epoch
        
        Args:
            model: the neural network model
            train_loader: data loader for training data
            optim: optimizer
            model_type: 'drift' or 'diff'
            device: device to use for computation (default: 'cuda' if available)
            time_step: Δt for time regularization (default: 0.01)
            time_reg_lambda: λ for time regularization (default: 0.0)
            grad_clip: max norm for gradient clipping (0 means no clipping, default: 0.0)
            include_time: whether to include time regularization (default: False)
            label_noise: whether to add noise to the labels (dxb) for data augmentation (default: False)

        Returns: dict with keys 'train_loss' and 'reg_loss', each is a list of losses for each batch
            train_losses: list of total losses (data loss + regularization) for each batch
            reg_losses: list of regularization losses for each batch
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert model_type in ['drift', 'diff'], "model_type must be 'drift' or 'diff'"

    train_losses = []
    reg_losses = []

    model.to(device)
    model.train()
    for batch in train_loader:
        xb, dxb, dtb = batch
        xb, dxb, dtb = xb.to(device), dxb.to(device), dtb.to(device)

        optim.zero_grad()

        if label_noise:
            dxb = dxb + torch.randn_like(dxb) * dxb.std() * 0.1

        f_pred = model(xb) if model_type == 'drift' else model(xb).flatten(start_dim=-2)

        reg_loss = torch.zeros((), device=device)
        # optional: regularization for time derivative ∂F/∂t, i.e., λ| Δt ∂_t F |^2
        if include_time and time_reg_lambda > 0:
            # time_col is the last column of xb, we add Δt to it to get the input for F(t+Δt)
            xb_next = xb.clone()
            xb_next[:, -1] += time_step

            f_pred_next = model(xb_next) if model_type == 'drift' else model(xb_next).flatten(start_dim=-2)

            # ΔF ≈ (F(t+Δt) - F(t))
            df_dt = (f_pred_next - f_pred)

            reg_loss += time_reg_lambda * (df_dt**2).mean()

        data_loss = (f_pred * dtb - dxb)**2
        data_loss = (data_loss).mean()

        # total
        loss = data_loss + reg_loss
        loss.backward()

        # optional: clip to stabilize training
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        train_losses.append(loss.item())
        reg_losses.append(reg_loss.item())

    return {'train_loss': train_losses, 'reg_loss': reg_losses}


def evaluate(
        model, 
        val_loader, 
        model_type, 
        device=None
    ):
    """
        evaluate the model on validation set

        Args:
            val_loader: data loader for validation data
            model: the neural network model
            model_type: 'drift' or 'diff'
            device: device to use for computation (default: 'cuda' if available)
        Returns:
            val_loss: squared error summed over output coordinates and divided by
                the number of dataset samples (without time regularization)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert model_type in ['drift', 'diff'], "model_type must be 'drift' or 'diff'"

    model.to(device)
    model.eval()
    val_loss = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in val_loader:
            xb, dxb, dtb = batch
            xb, dxb, dtb = xb.to(device), dxb.to(device), dtb.to(device)

            f_pred = model(xb) if model_type == 'drift' else model(xb).flatten(start_dim=-2)

            data_loss = (f_pred * dtb - dxb)**2

            val_loss += (data_loss).sum()

    return val_loss.item() / len(val_loader.dataset)
