"""An implementation of the DeepDDS model.

DeepDDS: deep graph neural network with attention mechanism to predict
synergistic drug combinations.

arXiv:2107.02467 [cs.LG]
https://arxiv.org/abs/2107.02467

SMILES strings transformed into a graph representation are used as input to
both the GAT and the GCN version of the model.

MLP is used to extract the feature embedding of gene expression profiles of
cancer cell line.

The embedding vector from both inputs are concatenated and fed into the
fully connected layers for binary classification of the drug combination as
synergistic or antagonistic.
"""
from typing import Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch import FloatTensor, nn
from torch.nn.functional import elu, normalize, relu, softmax
from torch_geometric.nn import GATConv, global_max_pool
from torchdrug.data import PackedGraph
from torchdrug.layers import MLP, MaxReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDDS",
]


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network.

    This implementation is based on the GATConv from torch_geometric and is
    needed because the TorchDrug implementation does not support true max
    readout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        dropout: float = 0.2,
    ):
        """Initialize the GraphAttentionNetwork.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            heads: Number of attention heads.
            concat: If True, concatenate the output of the attention heads.
            dropout: Dropout probability.
            negative_slope: LeakyReLU negative slope.
            residual: If True, add a residual connection.
        """
        super(GraphAttentionNetwork, self).__init__()

        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=out_channels * 10,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
        )
        self.fc = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: FloatTensor, edge_index: torch.LongTensor) -> FloatTensor:
        """Forward pass of the GraphAttentionNetwork.

        Args:
            x: Input tensor.
            edge_index: Edge indices.

        Returns:
            Output tensor.
        """
        x, arr = self.gat1(x, edge_index)
        x = elu(x)
        x = self.dropout(x)
        x, arr = self.gat2(x, edge_index)
        x = elu(x)
        x = self.dropout(x)
        x = global_max_pool(x, arr)
        x = self.fc(x)
        x = relu(x)
        return x


class DeepDDS(Model):
    r"""An implementation of the [DeepDDS model]_.

    .. [DeepDDS model] `DeepDDS: deep graph neural network with attention
    mechanism to predict synergistic drug combinations
    <https://arxiv.org/abs/2107.02467>`_
    """

    # todo: implement the GAT version of the model as well

    def __init__(
        self,
        *,
        context_feature_size: int,
        context_output_size: int,
        in_channels: int = TORCHDRUG_NODE_FEATURES,
        dropout: float = 0.2,  # Rate used in paper
        gat_heads: int = 10,
        drug_feature_model: Literal["gat", "gcn"] = "gcn",  # gcn or gat
    ):
        """Instantiate the DeepDDS model.

        :param context_feature_size:
        :param context_output_size:
        :param in_channels:
        :param dropout:
        :param drug_feature_model:
        """
        super(DeepDDS, self).__init__()
        self._drug_feature_model = drug_feature_model
        self.cell_mlp = MLP(input_dim=context_feature_size, hidden_dims=[2048, 512, context_output_size])
        print(f"Using {drug_feature_model} for drug features")
        if self._drug_feature_model == "gcn":
            self.conv_left = GraphConvolutionalNetwork(
                input_dim=in_channels, hidden_dims=[in_channels, in_channels * 2, in_channels * 4]
            )
            self.conv_right = self.conv_left
            self.mlp_left = MLP(
                input_dim=in_channels * 4,
                hidden_dims=[in_channels * 2, context_output_size],
                dropout=dropout,
            )
            self.mlp_right = self.mlp_left
        else:
            num_heads = gat_heads
            self.gat_left = GraphAttentionNetwork(
                in_channels=in_channels, out_channels=context_output_size, heads=num_heads, dropout=dropout
            )
            self.gat_right = self.gat_left

        self.mlp_final = MLP(input_dim=context_output_size * 3, hidden_dims=[1024, 512, 128, 1], dropout=dropout)
        self.max_readout = MaxReadout()

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features and right drug features."""
        return batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right

    def forward(
        self, context_features: FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> FloatTensor:
        """Run a forward pass of the DeeDDS model.

        Args:
            context_features (FloatTensor): A matrix of cell line features
            molecules_left (FloatTensor): A matrix of left drug features
            molecules_right (FloatTensor): A matrix of right drug features
        Returns:
            (FloatTensor): A column vector of predicted synergy scores
        """
        # Run the MLP forward for the cell line features
        mlp_out = self.cell_mlp(normalize(context_features, p=2, dim=1))

        if self._drug_feature_model == "gcn":
            features_left, features_right = self._gcn_forward(molecules_left, molecules_right)
        else:
            features_left, features_right = self._gat_forward(molecules_left, molecules_right)

        # Concatenate the output of the MLP and the GNN
        concat_in = torch.cat([mlp_out, features_left, features_right], dim=1)

        # Run the fully connected layers forward
        out_mlp = self.mlp_final(concat_in)

        # Apply the softmax function to the output
        return softmax(out_mlp, dim=1)

    def _gcn_forward(
        self, molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Run a forward pass of the DeeDDS model's GCN layers.

        Args:
            molecules_left (FloatTensor): A matrix of left drug features
            molecules_right (FloatTensor): A matrix of right drug features
        Returns:
            tuple(FloatTensor, FloatTensor): A column vector of predicted
            synergy scores
        """
        # Todo: source code does *not* run a final activation before
        #  concatenating; torch does this automatically
        features_left = self.conv_left(molecules_left, molecules_left.data_dict["node_feature"])["node_feature"]
        features_left = self.mlp_left(features_left)
        features_left = self.max_readout.forward(input=features_left, graph=molecules_left)
        features_right = self.conv_right(molecules_right, molecules_right.data_dict["node_feature"])["node_feature"]
        features_right = self.mlp_right(features_right)
        features_right = self.max_readout.forward(input=features_right, graph=molecules_right)
        return features_left, features_right

    def _gat_forward(
        self, molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Run a forward pass of the DeeDDS model's GAT layers.

        Args:
            molecules_left (PackedGraph): A matrix of left drug features
            molecules_right (PackedGraph): A matrix of right drug features
        Returns:
            tuple(FloatTensor, FloatTensor): A column vector of predicted
            synergy scores
        """
        features_left = self.gat_left(molecules_left, molecules_left.data_dict["node_feature"])["node_feature"]
        features_right = self.gat_right(molecules_right, molecules_right.data_dict["node_feature"])["node_feature"]
        return features_left, features_right
