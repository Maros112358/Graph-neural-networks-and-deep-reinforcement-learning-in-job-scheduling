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


from functools import partial

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from generic.agent import Agent
from .gnn_dgl import GnnDGL
from .gnn_tokengt import GnnTokenGT
from generic.mlp import MLP
from jssp.utils.utils import obs_as_tensor, obs_as_tensor_add_batch_dim, rebatch_obs


class Agent(Agent):
    def __init__(
        self,
        env_specification,
        gnn=None,
        value_net=None,
        action_net=None,
        agent_specification=None,
    ):
        """
        There are 2 ways to init an Agent:
         - Either provide a valid env_specification and agent_specification
         - Or use the load method, to load an already saved Agent
        """
        super().__init__(
            env_specification, gnn, value_net, action_net, agent_specification
        )

        self.obs_as_tensor_add_batch_dim = obs_as_tensor_add_batch_dim
        self.obs_as_tensor = obs_as_tensor

        self.rebatch_obs = rebatch_obs
        self.graphobs = False

        # If a model is provided, we simply load the existing model.
        if gnn is not None and value_net is not None and action_net is not None:
            self.gnn = gnn
            self.value_net = value_net
            self.action_net = action_net
            return

        if self.agent_specification.fe_type == "dgl":
            self.gnn = GnnDGL(
                input_dim_features_extractor=env_specification.n_features,
                gconv_type=agent_specification.gconv_type,
                graph_pooling=agent_specification.graph_pooling,
                graph_has_relu=agent_specification.graph_has_relu,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
                activation_features_extractor=agent_specification.activation_fn_graph,
                n_layers_features_extractor=agent_specification.n_layers_features_extractor,
                hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
                n_attention_heads=agent_specification.n_attention_heads,
                residual=agent_specification.residual_gnn,
                normalize=agent_specification.normalize_gnn,
                conflicts=agent_specification.conflicts,
                mid_in_edges=agent_specification.mid_in_edges,
            )
        elif self.agent_specification.fe_type == "tokengt":
            self.gnn = GnnTokenGT(
                input_dim_features_extractor=env_specification.n_features,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                conflicts=agent_specification.conflicts,
                encoder_layers=agent_specification.n_layers_features_extractor,
                encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_attention_heads=agent_specification.n_attention_heads,
                activation_fn=agent_specification.activation_fn_graph,
                lap_node_id=True,
                lap_node_id_k=agent_specification.lap_node_id_k,
                lap_node_id_sign_flip=True,
                type_id=True,
                transformer_flavor=agent_specification.transformer_flavor,
                layer_pooling=agent_specification.layer_pooling,
                dropout=agent_specification.dropout,
                attention_dropout=agent_specification.dropout,
                act_dropout=agent_specification.dropout,
                cache_lap_node_id=agent_specification.cache_lap_node_id,
            )
        else:
            print("unknown fe_type: ", agent_specification.fe_type)

        self.init_heads()

    @classmethod
    def load(cls, path):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        if not path.endswith(".pkl"):
            path = path + "agent.pkl"
        save_data = torch.load(path)
        agent_specification = save_data["agent_specification"]
        env_specification = save_data["env_specification"]
        if agent_specification.fe_type == "dgl":
            gnn = GnnDGL(
                input_dim_features_extractor=env_specification.n_features,
                gconv_type=agent_specification.gconv_type,
                graph_pooling=agent_specification.graph_pooling,
                graph_has_relu=agent_specification.graph_has_relu,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
                activation_features_extractor=agent_specification.activation_fn_graph,
                n_layers_features_extractor=agent_specification.n_layers_features_extractor,
                hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
                n_attention_heads=agent_specification.n_attention_heads,
                residual=agent_specification.residual_gnn,
                normalize=agent_specification.normalize_gnn,
                conflicts=agent_specification.conflicts,
                mid_in_edges=agent_specification.mid_in_edges,
            )
        elif agent_specification.fe_type == "tokengt":
            gnn = GnnTokenGT(
                input_dim_features_extractor=env_specification.n_features,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                conflicts=agent_specification.conflicts,
                encoder_layers=agent_specification.n_layers_features_extractor,
                encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_attention_heads=agent_specification.n_attention_heads,
                activation_fn=agent_specification.activation_fn_graph,
                lap_node_id=True,
                lap_node_id_k=agent_specification.lap_node_id_k,
                lap_node_id_sign_flip=True,
                type_id=True,
                transformer_flavor=agent_specification.transformer_flavor,
                layer_pooling=agent_specification.layer_pooling,
                dropout=agent_specification.dropout,
                attention_dropout=agent_specification.dropout,
                act_dropout=agent_specification.dropout,
                cache_lap_node_id=agent_specification.cache_lap_node_id,
            )
        value_net = MLP(
            len(agent_specification.net_arch["vf"]),
            gnn.features_dim // 2,
            agent_specification.net_arch["vf"][0],
            1,
            False,
            agent_specification.activation_fn,
        )

        # # action
        action_net = MLP(
            len(agent_specification.net_arch["pi"]),
            gnn.features_dim,
            agent_specification.net_arch["pi"][0],
            1,
            False,
            agent_specification.activation_fn,
        )

        agent = cls(env_specification, gnn, value_net, action_net, agent_specification)
        # constructors init weight!!!
        agent.gnn.load_state_dict(save_data["gnn"])
        agent.action_net.load_state_dict(save_data["action_net"])
        agent.value_net.load_state_dict(save_data["value_net"])
        return agent

    def get_obs(self, b_obs, mb_ind):
        minibatched_obs = {}
        for key in b_obs:
            minibatched_obs[key] = b_obs[key][mb_ind]
        return minibatched_obs

    def init_heads(self):
        """Initialize new heads, removing old heads if existing."""
        if hasattr(self, "value_net") and self.value_net is not None:
            device = next(self.value_net.parameters())
        else:
            device = "cpu"

        self.value_net = MLP(
            len(self.agent_specification.net_arch["vf"]),
            self.gnn.features_dim // 2,
            self.agent_specification.net_arch["vf"][0],
            1,
            False,
            self.agent_specification.activation_fn,
        )
        # # action
        self.action_net = MLP(
            len(self.agent_specification.net_arch["pi"]),
            self.gnn.features_dim,
            self.agent_specification.net_arch["pi"][0],
            1,
            False,
            self.agent_specification.activation_fn,
        )
        # usually ppo use gain = np.sqrt(2) here
        # best so far below
        self.gnn.apply(partial(self.init_weights, gain=1.0, zero_bias=False))
        # usually ppo use gain = 0.01 here
        self.action_net.apply(partial(self.init_weights, gain=1.0, zero_bias=True))
        # usually ppo use gain = 1 here
        self.value_net.apply(partial(self.init_weights, gain=1.0, zero_bias=True))

        self.value_net.to(device)
        self.action_net.to(device)
