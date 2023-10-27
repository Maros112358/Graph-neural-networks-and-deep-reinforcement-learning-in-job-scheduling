#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#


from generic.mlp import MLP
import torch
import dgl
from dgl.nn import EGATConv

from dgl import LaplacianPE
from .agent_observation import AgentObservation
from .edge_embedder import PspEdgeEmbedder
from .rewiring import rewire


class GnnDGL(torch.nn.Module):
    def __init__(
        self,
        input_dim_features_extractor,
        graph_pooling,
        max_n_nodes,
        max_n_resources,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        activation_features_extractor,
        n_attention_heads,
        residual=True,
        normalize=False,
        conflicts="att",
        edge_embedding_flavor="sum",
        layer_pooling="all",
        factored_rp=False,
        add_rp_edges="all",
        add_self_loops=False,
        vnode=False,
        update_edge_features=False,
        update_edge_features_pe=False,
        rwpe_k=16,
        rwpe_h=16,
        cache_rwpe=False,
    ):
        super().__init__()
        self.conflicts = conflicts
        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        self.input_dim_features_extractor = input_dim_features_extractor
        self.layer_pooling = layer_pooling
        self.rwpe_k = rwpe_k
        if self.rwpe_k == 0:
            self.rwpe_h = 0
        else:
            self.rwpe_h = rwpe_h
        if layer_pooling == "all":
            self.features_dim = (
                input_dim_features_extractor
                + (hidden_dim_features_extractor + self.rwpe_h)
                * (n_layers_features_extractor + 1)
                + self.rwpe_h
            )
        else:
            self.features_dim = hidden_dim_features_extractor + self.rwpe_h

        self.factored_rp = factored_rp
        self.add_rp_edges = add_rp_edges
        self.features_dim *= 2
        self.max_n_resources = max_n_resources
        self.add_self_loops = add_self_loops
        self.update_edge_features = update_edge_features
        self.update_edge_features_pe = update_edge_features_pe

        self.hidden_dim = hidden_dim_features_extractor
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor
        self.features_extractors = torch.nn.ModuleList()
        self.vnode = vnode

        if cache_rwpe:
            self.rwpe_cache = {}
        else:
            self.rwpe_cache = None

        if self.rwpe_k != 0:
            self.rwpe_embedder = torch.nn.Linear(self.rwpe_k * 4, self.rwpe_h)

        self.pool_node_embedder = torch.nn.Embedding(1, input_dim_features_extractor)
        self.vnode_embedder = torch.nn.Embedding(1, input_dim_features_extractor)
        self.resource_node_embedder = torch.nn.Embedding(
            max_n_resources, input_dim_features_extractor
        )

        self.edge_embedder = PspEdgeEmbedder(
            edge_embedding_flavor,
            self.max_n_resources,
            self.hidden_dim,
            self.add_rp_edges,
            self.factored_rp,
        )
        if self.rwpe_k != 0:
            self.edge_embedder_pe = PspEdgeEmbedder(
                edge_embedding_flavor,
                self.max_n_resources,
                self.hidden_dim,
                self.add_rp_edges,
                self.factored_rp,
            )

        self.features_embedder = MLP(
            n_layers=n_mlp_layers_features_extractor,
            input_dim=input_dim_features_extractor,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            batch_norm=self.normalize,
            activation=activation_features_extractor,
        )

        if self.normalize:
            self.norms = torch.nn.ModuleList()
            self.normsbis = torch.nn.ModuleList()
            self.norm0 = torch.nn.BatchNorm1d(input_dim_features_extractor)
            self.norm1 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.mlps = torch.nn.ModuleList()
        if self.update_edge_features:
            self.mlps_edges = torch.nn.ModuleList()
        if self.update_edge_features_pe and self.rwpe_k != 0:
            self.mlps_edges_pe = torch.nn.ModuleList()

        if self.rwpe_k != 0:
            self.pe_conv = torch.nn.ModuleList()
            self.pe_mlp = torch.nn.ModuleList()

        for layer in range(self.n_layers_features_extractor):
            if self.normalize:
                self.norms.append(torch.nn.BatchNorm1d(self.hidden_dim))
                if self.residual:
                    self.normsbis.append(torch.nn.BatchNorm1d(self.hidden_dim))

            if self.rwpe_k != 0:
                self.pe_conv.append(
                    EGATConv(
                        self.rwpe_h,
                        self.hidden_dim,
                        self.rwpe_h,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                    )
                )
                self.pe_mlp.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=self.rwpe_h * n_attention_heads,
                        hidden_dim=self.rwpe_h * n_attention_heads,
                        output_dim=self.rwpe_h,
                        batch_norm=self.normalize,
                        activation="tanh",
                    )
                )
            self.features_extractors.append(
                EGATConv(
                    self.hidden_dim + self.rwpe_h,
                    self.hidden_dim,
                    self.hidden_dim + self.rwpe_h,
                    self.hidden_dim,
                    num_heads=n_attention_heads,
                )
            )
            self.mlps.append(
                MLP(
                    n_layers=n_mlp_layers_features_extractor,
                    input_dim=(self.hidden_dim + self.rwpe_h) * n_attention_heads,
                    hidden_dim=(self.hidden_dim + self.rwpe_h) * n_attention_heads,
                    output_dim=self.hidden_dim,
                    batch_norm=self.normalize,
                    activation=activation_features_extractor,
                )
            )
            if self.update_edge_features:
                self.mlps_edges.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=self.hidden_dim * n_attention_heads,
                        hidden_dim=self.hidden_dim * n_attention_heads,
                        output_dim=self.hidden_dim,
                        batch_norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )
            if self.update_edge_features_pe and rwpe_k != 0:
                self.mlps_edges_pe.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=self.hidden_dim * n_attention_heads,
                        hidden_dim=self.hidden_dim * n_attention_heads,
                        output_dim=self.hidden_dim,
                        batch_norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )

    def reset_egat(self):
        for egat in self.features_extractors:
            egat.reset_parameters()

    def forward(self, obs):
        observation = AgentObservation(
            obs,
            conflicts=self.conflicts,
            add_self_loops=self.add_self_loops,
            factored_rp=self.factored_rp,
            add_rp_edges=self.add_rp_edges,
            max_n_resources=self.max_n_resources,
            rwpe_k=self.rwpe_k,
            rwpe_cache=self.rwpe_cache,
        )
        batch_size = observation.get_batch_size()
        n_nodes = observation.get_n_nodes()

        g = observation.to_graph()

        num_nodes = g.num_nodes()

        g, poolnodes, resource_nodes, vnodes = rewire(
            g,
            self.graph_pooling == "learn",
            self.conflicts == "node",
            self.vnode,
            batch_size,
            self.input_dim_features_extractor,
            self.max_n_resources,
            6,
            7,
            10,
            10,
            8,
            9,
        )

        g = g.to(next(self.parameters()).device)
        features = g.ndata["feat"]

        if self.rwpe_k != 0:
            g.ndata["pe"] = self.rwpe_embedder(
                torch.cat(
                    [
                        g.ndata["rwpe_global"],
                        g.ndata["rwpe_pr"],
                        g.ndata["rwpe_rp"],
                        g.ndata["rwpe_rc"],
                    ],
                    1,
                ),
            )

        if self.graph_pooling == "learn":
            features[poolnodes] = self.pool_node_embedder(
                torch.LongTensor([0] * len(poolnodes)).to(features.device)
            )

        if self.vnode:
            features[vnodes] = self.vnode_embedder(
                torch.LongTensor([0] * len(vnodes)).to(features.device)
            )
        if self.conflicts == "node":
            features[resource_nodes] = self.resource_node_embedder(
                torch.LongTensor(list(range(num_resources)) * batch_size).to(
                    features.device
                )
            )

        if self.rwpe_k != 0:
            pe = g.ndata["pe"]

        if self.normalize:
            features = self.norm0(features)

        if self.layer_pooling == "all":
            features_list = []
            if self.rwpe_k != 0:
                features_list.append(torch.cat([features, pe], dim=-1))
            else:
                features_list.append(features)

        features = self.features_embedder(features)

        if self.normalize:
            features = self.norm1(features)

        if self.layer_pooling == "all":
            if self.rwpe_k != 0:
                features_list.append(torch.cat([features, pe], dim=-1))
            else:
                features_list.append(features)

        edge_features = self.edge_embedder(g)
        if self.rwpe_k != 0:
            edge_features_pe = self.edge_embedder_pe(g)

        if self.layer_pooling == "last":
            previous_feat = features
            if self.rwpe_k != 0:
                previous_pe = pe

        for layer in range(self.n_layers_features_extractor):
            if self.rwpe_k != 0:
                features = torch.cat([features, pe], dim=-1)

            if self.update_edge_features:
                features, new_e_attr = self.features_extractors[layer](
                    g, features, edge_features.clone()
                )
                edge_features += self.mlps_edges[layer](
                    new_e_attr.flatten(start_dim=-2, end_dim=-1)
                )
            else:
                features, _ = self.features_extractors[layer](
                    g, features, edge_features
                )

            features = self.mlps[layer](features.flatten(start_dim=-2, end_dim=-1))

            if self.rwpe_k != 0:
                if self.update_edge_features_pe:
                    pe, new_e_pe_attr = self.pe_conv[layer](
                        g, pe, edge_features_pe.clone()
                    )
                    edge_features_pe += self.mlps_edges_pe[layer](
                        new_e_pe_attr.flatten(start_dim=-2, end_dim=-1)
                    )
                else:
                    pe, _ = self.pe_conv[layer](g, pe, edge_features_pe)
                pe = self.pe_mlp[layer](pe.flatten(start_dim=-2, end_dim=-1))

            # if self.layer_pooling == "all":
            #     features_list.append(features)
            if self.normalize:
                features = self.norms[layer](features)

            if self.residual:
                if self.layer_pooling == "all":
                    features += features_list[-1][:, : self.hidden_dim]
                    if self.rwpe_k != 0:
                        pe += features_list[-1][:, self.hidden_dim :]
                else:
                    features += previous_feat[:, : self.hidden_dim]
                    if self.rwpe_k != 0:
                        pe += previous_pe
                if self.normalize:
                    features = self.normsbis[layer](features)
                if self.layer_pooling == "last":
                    previous_feat = features
                    if self.rwpe_k != 0:
                        previous_pe = pe

            if self.layer_pooling == "all":
                if self.rwpe_k != 0:
                    features_list.append(torch.cat([features, pe], dim=-1))
                else:
                    features_list.append(features)

        if self.layer_pooling == "all":
            features = torch.cat(
                features_list, axis=1
            )  # The final embedding is concatenation of all layers embeddings
        node_features = features[:num_nodes, :]
        node_features = node_features.reshape(batch_size, n_nodes, -1)

        if self.graph_pooling == "max":
            max_elts, max_ind = torch.max(node_features, dim=1)
            graph_embedding = max_elts
        elif self.graph_pooling == "avg":
            graph_pooling = torch.ones(n_nodes, device=self.device) / n_nodes
            graph_embedding = torch.matmul(graph_pooling, node_features)
        elif self.graph_pooling == "learn":
            graph_embedding = features[num_nodes : num_nodes + batch_size, :]
        else:
            raise Exception(
                f"Graph pooling {self.graph_pooling} not recognized. Only accepted pooling are max and avg"
            )

        node_features = torch.nn.functional.pad(
            node_features,
            (0, 0, 0, self.max_n_nodes - node_features.shape[1]),
            mode="constant",
            value=0.0,
        )

        graph_embedding = graph_embedding.reshape(batch_size, 1, -1)
        # repeat the graph embedding to match the nodes embedding size
        repeated = graph_embedding.expand(node_features.shape)
        ret = torch.cat((node_features, repeated), dim=2)
        return ret
