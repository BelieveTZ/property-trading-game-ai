from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoding import (
    ACTION_NUMERIC_FEATURES,
    DEED_TILES,
    GLOBAL_FEATURES,
    MAX_PLAYERS,
    PLAYER_FEATURES,
    PROPERTY_FEATURES,
    collate_actions,
    collate_observations,
)
from .env import ActionKind


@dataclass(slots=True)
class ModelConfig:
    d_model: int = 192
    nhead: int = 6
    layers: int = 4
    feedforward: int = 512
    dropout: float = 0.0


class MonopolyPolicy(nn.Module):
    """Entity-aware recurrent actor-critic with candidate-action scoring."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.d_model
        self.global_projection = nn.Sequential(nn.Linear(GLOBAL_FEATURES, d), nn.LayerNorm(d), nn.GELU())
        self.player_projection = nn.Sequential(nn.Linear(PLAYER_FEATURES, d), nn.LayerNorm(d), nn.GELU())
        self.property_projection = nn.Sequential(nn.Linear(PROPERTY_FEATURES, d), nn.LayerNorm(d), nn.GELU())
        self.token_type = nn.Embedding(3, d)
        self.player_slot = nn.Embedding(MAX_PLAYERS, d)
        self.property_slot = nn.Embedding(len(DEED_TILES), d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.nhead,
            dim_feedforward=self.config.feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.config.layers, norm=nn.LayerNorm(d))
        self.state_projection = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.LayerNorm(d))
        self.memory = nn.GRUCell(d, d)

        self.action_kind = nn.Embedding(len(ActionKind), d)
        self.action_tile = nn.Embedding(len(DEED_TILES) + 1, d)
        self.action_give_tile = nn.Embedding(len(DEED_TILES) + 1, d)
        self.action_take_tile = nn.Embedding(len(DEED_TILES) + 1, d)
        self.action_target = nn.Embedding(MAX_PLAYERS + 1, d)
        self.action_numeric = nn.Sequential(nn.Linear(ACTION_NUMERIC_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.action_norm = nn.LayerNorm(d)
        self.policy_query = nn.Linear(d, d, bias=False)
        self.policy_bias = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        self.apply(self._init_weights)

    @property
    def hidden_size(self) -> int:
        return self.config.d_model

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def encode_state(self, observation: dict[str, Tensor], hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        global_token = self.global_projection(observation["global"]).unsqueeze(1)
        player_tokens = self.player_projection(observation["players"])
        property_tokens = self.property_projection(observation["properties"])

        global_token = global_token + self.token_type.weight[0].view(1, 1, -1)
        player_tokens = player_tokens + self.token_type.weight[1].view(1, 1, -1) + self.player_slot.weight.unsqueeze(0)
        property_tokens = property_tokens + self.token_type.weight[2].view(1, 1, -1) + self.property_slot.weight.unsqueeze(0)
        encoded = self.encoder(torch.cat((global_token, player_tokens, property_tokens), dim=1))
        state = self.state_projection(torch.cat((encoded[:, 0], encoded[:, 1]), dim=-1))
        if hidden is None:
            hidden = self.initial_hidden(state.shape[0], state.device)
        next_hidden = self.memory(state, hidden)
        return next_hidden, next_hidden

    def encode_actions(self, actions: dict[str, Tensor]) -> Tensor:
        action = (
            self.action_kind(actions["kind"])
            + self.action_tile(actions["tile"])
            + self.action_give_tile(actions["give_tile"])
            + self.action_take_tile(actions["take_tile"])
            + self.action_target(actions["target"])
            + self.action_numeric(actions["numeric"])
        )
        return self.action_norm(action)

    def forward(
        self,
        observation: dict[str, Tensor],
        actions: dict[str, Tensor],
        action_mask: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state, next_hidden = self.encode_state(observation, hidden)
        action_embeddings = self.encode_actions(actions)
        query = self.policy_query(state).unsqueeze(1)
        logits = (query * action_embeddings).sum(dim=-1) / math.sqrt(self.hidden_size)
        logits = logits + self.policy_bias(action_embeddings).squeeze(-1)
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        value = self.value_head(state).squeeze(-1)
        return logits, value, next_hidden
