import argparse

import torch
import torch.nn as nn


def get_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments

    The following arguments can be specified:

    --model_dim: int, dimensionality of the input model activations
    --dict_dim: int, dimensionality of the SAE dictionary
    --l1_coeff: float, L1 regularization coefficient for sparsity (default: 1e-4)
    --batch_size: int, training batch size (default: 128)
    --lr: float, learning rate (default: 1e-3)
    """
    parser = argparse.ArgumentParser(description="Train a Standard SAE")

    parser.add_argument("--model_dim", type=int, help="dimensionality of the input model activations")
    parser.add_argument("--dict_dim", type=int, help="dimensionality of the SAE dictionary")

    parser.add_argument("--l1_coeff", type=float, default=1e-4, help="L1 regularization coefficient for sparsity")
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    return parser.parse_args()


class StandardSAE(nn.Module):
    def __init__(self, model_dim, dict_dim):
        """
        Initialize a StandardSAE model.

        Args:
            model_dim (int): dimensionality of the input space
            dict_dim (int): dimensionality of the dictionary space

        """
        super().__init__()
        self.model_dim = model_dim
        self.dict_dim = dict_dim

        self.encoder = nn.Linear(model_dim, dict_dim, bias=True)
        self.decoder = nn.Linear(dict_dim, model_dim, bias=True)

    def encode(self, x):
        """
        Encode input x into dictionary space.

        Args:
            x (torch.Tensor): input activations

        Returns:
            torch.Tensor: encoded activations
        """
        return nn.ReLU()(self.encoder(x))

    def decode(self, z):
        """
        Decode a dictionary vector z into the input space.

        Args:
            z (torch.Tensor): dictionary vector

        Returns:
            torch.Tensor: decoded input activations
        """
        return nn.ReLU()(self.decoder(z))
    
    def forward(self, x, output_features=False):
        """
        Forward pass of the StandardSAE model.

        Args:
            x (torch.Tensor): input activations
            output_features (bool): if True, return the encoded features as well as the decoded x

        Returns:
            torch.Tensor: decoded input activations and/or encoded features
        """
        z = self.encode(x)
        x_hat = self.decode(z)

        if output_features:
            return x_hat, z
        else:
            return x_hat


def calculate_loss(ae, x, l1_coeff):
    """
    Calculate the loss of a StandardSAE model given the input activations and l1 regularization coefficient.

    Args:
        ae (StandardSAE): the StandardSAE model
        x (torch.Tensor): input activations
        l1_coeff (float): l1 regularization coefficient

    Returns:
        float: the loss of the StandardSAE model
    """
    x_hat, z = ae.forward(x, output_features=True)

    l2_loss = (x_hat - x).pow(2).sum(dim=1).mean()
    l1_loss = l1_coeff * z.sum()
    loss = l2_loss + l1_loss
    
    return loss


if __name__ == '__main__':
    
    args = get_args()

    model_dim = args.model_dim
    dict_dim = args.dict_dim
