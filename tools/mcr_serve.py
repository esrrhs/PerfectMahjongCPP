#!/usr/bin/env python3
"""Serves the pre-trained Botzone MCR TokenTransformer baseline over TCP.

Protocol (tab-separated single line in, action index line out):
  0: player_id         (int 0..3)
  1: phase             (str: draw|discard|claim|terminal)
  2: prevalent_wind    (int 0..3)
  3: wall_remaining    (int)
  4: last_discard      (str: "player,tile" or "-")
  5: flags             (str: "wall_last,about_kong,claim_hu_only")
  6: hand              (str: comma-separated tile IDs or "-")
  7: discards          (str: 4 players separated by ';', comma-separated tile IDs or "-")
  8: melds             (str: 4 players separated by ';', each '|' list of "kind:from_player:t0,t1,...")
  9: events            (str: ';' list of "KIND:player:tile:discard" or "-")
 10: legal_actions     (str: ';' list of "kind,tile,s0,s1,s2,discard")

Response:
  "<best_action_index>\\n"
"""
import argparse
from collections import Counter, namedtuple
from enum import IntEnum
import os
from pathlib import Path
import socket
import torch
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = ""

TOKEN_SIZE = 12
MAX_STATE_TOKENS = 256
MAX_ACTION_TOKENS = 4

TOKEN_GLOBAL = 1
TOKEN_HAND = 2
TOKEN_DISCARD = 3
TOKEN_MELD = 4
TOKEN_EVENT = 5
TOKEN_UNSEEN = 6
TOKEN_ACTION = 7

EVENT_IDS = {
    "DRAW": 1, "PLAY": 2, "CHI": 3, "PENG": 4, "GANG": 5,
    "ANGANG": 6, "BUGANG": 7, "HU": 8, "BUHUA": 9,
}


class ActionType(IntEnum):
    PASS = 0
    PLAY = 1
    CHI = 2
    PENG = 3
    GANG = 4
    BUGANG = 5
    HU = 6


Meld = namedtuple("Meld", "kind tiles from_player")
Action = namedtuple("Action", "kind tile sequence discard")


def _relative(player, observer):
    return (int(player) - int(observer)) % 4


def _token(kind, tile=-1, player=-1, action=-1, order=0, flags=0, value=0.0, extra=0.0):
    return [
        float(kind), float(tile + 1), float(player + 1), float(action + 1),
        float(order) / 128.0, float(flags), float(value), float(extra),
        0.0, 0.0, 0.0, 0.0,
    ]


def _pad(tokens, maximum):
    tokens = tokens[:maximum]
    mask = [1] * len(tokens)
    while len(tokens) < maximum:
        tokens.append([0.0] * TOKEN_SIZE)
        mask.append(0)
    return tokens, mask


def encode_observation_v2(observation):
    observer = int(observation["player_id"])
    last = observation.get("last_discard")
    phase = {"ack": 0, "draw": 1, "discard": 2, "claim": 3, "terminal": 4}.get(
        observation.get("phase"), 0)
    flags = (
        int(bool(observation.get("wall_last"))) |
        (int(bool(observation.get("about_kong"))) << 1) |
        (int(bool(observation.get("claim_hu_only"))) << 2)
    )
    tokens = [_token(
        TOKEN_GLOBAL, tile=last[1] if last else -1,
        player=_relative(observation.get("current_player", observer), observer),
        action=phase, flags=flags,
        value=float(observation.get("wall_remaining", 0)) / 84.0,
        extra=float(_relative(observation.get("prevalent_wind", 0), observer)) / 3.0,
    )]
    for order, tile in enumerate(observation["hand"]):
        tokens.append(_token(TOKEN_HAND, tile=tile, player=0, order=order))

    claimed = Counter()
    for owner, melds in enumerate(observation["melds"]):
        for meld in melds:
            if meld.from_player != -1 and meld.from_player != owner:
                claimed[(meld.from_player, meld.tiles[-1])] += 1
    for relative in range(4):
        player = (observer + relative) % 4
        river = observation["discards"][player]
        consumed = Counter(claimed)
        for order, tile in enumerate(river):
            used = consumed[(player, tile)] > 0
            if used:
                consumed[(player, tile)] -= 1
            tokens.append(_token(
                TOKEN_DISCARD, tile=tile, player=_relative(player, observer),
                order=order, flags=int(used)))

    for relative in range(4):
        player = (observer + relative) % 4
        melds = observation["melds"][player]
        for order, meld in enumerate(melds):
            for tile_order, tile in enumerate(meld.tiles):
                tokens.append(_token(
                    TOKEN_MELD, tile=tile, player=_relative(player, observer),
                    action=int(meld.kind), order=order,
                    flags=_relative(meld.from_player, observer),
                    extra=float(tile_order) / 4.0))

    for order, event in enumerate(observation.get("events", [])[-64:]):
        if not event:
            continue
        kind = str(event[0]).upper()
        player = event[1] if len(event) > 1 and isinstance(event[1], int) else observer
        tile = next((item for item in event[2:] if isinstance(item, int) and 0 <= item < 34), -1)
        tokens.append(_token(
            TOKEN_EVENT, tile=tile, player=_relative(player, observer),
            action=EVENT_IDS.get(kind, 0), order=order))

    visible = Counter(observation["hand"])
    for river in observation["discards"]:
        visible.update(river)
    for melds in observation["melds"]:
        for meld in melds:
            visible.update(meld.tiles)
    for tile in range(34):
        tokens.append(_token(
            TOKEN_UNSEEN, tile=tile, value=max(0, 4 - visible[tile]) / 4.0))
    for relative in range(4):
        player = (observer + relative) % 4
        meld_tiles = sum(len(meld.tiles) for meld in observation["melds"][player])
        concealed = 13 - (meld_tiles // 3) * 3
        if player == observation.get("current_player") and observation.get("phase") == "discard":
            concealed += 1
        tokens.append(_token(
            TOKEN_GLOBAL, player=_relative(player, observer),
            value=max(0, concealed) / 14.0,
            extra=float(observation.get("wall_remaining_by_player", [0] * 4)[player]) / 21.0))
    return _pad(tokens, MAX_STATE_TOKENS)


def encode_action_v2(action):
    tiles = list(action.sequence) if action.sequence else ([action.tile] if action.tile >= 0 else [])
    tokens = [_token(
        TOKEN_ACTION, tile=action.tile, action=int(action.kind),
        flags=int(action.kind == ActionType.HU), extra=(action.discard + 1) / 34.0)]
    for order, tile in enumerate(tiles[:3]):
        tokens.append(_token(TOKEN_ACTION, tile=tile, action=int(action.kind), order=order + 1))
    return _pad(tokens, MAX_ACTION_TOKENS)


class TokenTransformer(nn.Module):
    feature_version = 2
    architecture_version = 3

    def __init__(self, token_size=TOKEN_SIZE, d_model=192, layers=4, heads=6,
                 dropout=0.1, belief_mode="aux", auxiliary_logit_fusion=True):
        super().__init__()
        self.token_size = token_size
        self.d_model = d_model
        self.belief_mode = belief_mode
        self.auxiliary_logit_fusion = auxiliary_logit_fusion
        self.state_encoder = nn.Linear(token_size, d_model)
        self.action_encoder = nn.Linear(token_size, d_model)
        self.kind_embedding = nn.Embedding(16, d_model)
        self.tile_embedding = nn.Embedding(36, d_model)
        self.player_embedding = nn.Embedding(8, d_model)
        self.action_embedding = nn.Embedding(16, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, layers)
        interaction_size = d_model * 3
        self.family_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 7))
        self.tactical_head = nn.Sequential(
            nn.Linear(interaction_size, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.action_outcome_head = nn.Linear(interaction_size, 4)
        self.action_fan_head = nn.Linear(interaction_size, 5)
        self.auxiliary_logit_gate = nn.Parameter(torch.zeros(5))
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.outcome_head = nn.Linear(d_model, 4)
        self.fan_head = nn.Linear(d_model, 5)
        self.belief_head = nn.Linear(d_model, 3 * 34 * 5)
        self.belief_adapter = nn.Linear(3 * 34 * 5, d_model)

    def _embed(self, tokens, encoder):
        continuous = encoder(tokens)
        return (continuous
                + self.kind_embedding(tokens[..., 0].long().clamp(0, 15))
                + self.tile_embedding(tokens[..., 1].long().clamp(0, 35))
                + self.player_embedding(tokens[..., 2].long().clamp(0, 7))
                + self.action_embedding(tokens[..., 3].long().clamp(0, 15)))

    def forward(self, features, actions, action_mask=None, feature_mask=None,
                action_token_mask=None):
        state_tokens = self._embed(features, self.state_encoder)
        encoded = self.transformer(
            state_tokens.transpose(0, 1),
            src_key_padding_mask=(feature_mask == 0) if feature_mask is not None else None,
        ).transpose(0, 1)
        if feature_mask is None:
            state = encoded.mean(1)
        else:
            weight = feature_mask.float().unsqueeze(-1)
            state = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        initial_belief = self.belief_head(state).view(-1, 3, 34, 5)
        if self.belief_mode == "actor":
            probabilities = initial_belief.softmax(-1).detach()
            state = state + self.belief_adapter(probabilities.reshape(state.size(0), -1))
        action_encoded = self._embed(actions, self.action_encoder)
        if action_token_mask is None:
            action_state = action_encoded.mean(2)
        else:
            weight = action_token_mask.float().unsqueeze(-1)
            action_state = (action_encoded * weight).sum(2) / weight.sum(2).clamp_min(1.0)
        expanded = state.unsqueeze(1).expand_as(action_state)
        interaction = torch.cat((expanded, action_state, expanded * action_state), -1)
        action_kinds = actions[:, :, 0, 3].long().sub(1).clamp(0, 6)
        family_scores = self.family_head(state)
        family_logits = family_scores.gather(1, action_kinds)
        tactical_logits = self.tactical_head(interaction).squeeze(-1)
        action_outcome = self.action_outcome_head(interaction)
        action_fan_logits = self.action_fan_head(interaction)
        logits = family_logits + tactical_logits
        if self.auxiliary_logit_fusion:
            fan_values = torch.arange(5, device=actions.device, dtype=action_fan_logits.dtype)
            expected_fan = (action_fan_logits.softmax(-1) * fan_values).sum(-1)
            auxiliary = torch.stack((
                action_outcome[..., 0], -action_outcome[..., 1],
                action_outcome[..., 2], action_outcome[..., 3], expected_fan), -1)
            logits = logits + (auxiliary * self.auxiliary_logit_gate).sum(-1)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e4)
        return {"logits": logits}


def _parse_int_list(tok):
    if not tok or tok == "-":
        return []
    return [int(x) for x in tok.split(",") if x != ""]


def parse_request(line):
    fields = line.split("\t")
    if len(fields) != 11:
        raise ValueError(f"expected 11 fields, got {len(fields)}")

    player_id = int(fields[0])
    phase = fields[1]
    prevalent_wind = int(fields[2])
    wall_rem = int(fields[3])
    last_discard = None
    if fields[4] != "-":
        lp, lt = fields[4].split(",")
        last_discard = (int(lp), int(lt))
    fl_parts = [int(x) for x in fields[5].split(",")]
    wall_last = bool(fl_parts[0])
    about_kong = bool(fl_parts[1])
    claim_hu_only = bool(fl_parts[2])

    hand = _parse_int_list(fields[6])

    discards = []
    for p_tok in fields[7].split(";"):
        discards.append(_parse_int_list(p_tok))

    melds = []
    for p_tok in fields[8].split(";"):
        p_melds = []
        if p_tok and p_tok != "-":
            for m_tok in p_tok.split("|"):
                k_str, fp_str, t_str = m_tok.split(":")
                p_melds.append(Meld(
                    ActionType(int(k_str)),
                    tuple(_parse_int_list(t_str)),
                    int(fp_str)
                ))
        melds.append(p_melds)

    events = []
    if fields[9] and fields[9] != "-":
        for ev_tok in fields[9].split(";"):
            parts = ev_tok.split(":")
            if len(parts) >= 4:
                k_name = parts[0]
                ev_p = int(parts[1])
                ev_t = int(parts[2])
                ev_d = int(parts[3])
                if ev_t >= 0 and ev_d >= 0:
                    events.append((k_name, ev_p, ev_t, ev_d))
                elif ev_t >= 0:
                    events.append((k_name, ev_p, ev_t))
                else:
                    events.append((k_name, ev_p))

    legal_actions = []
    for a_tok in fields[10].split(";"):
        if not a_tok:
            continue
        k, t, s0, s1, s2, d = [int(x) for x in a_tok.split(",")]
        seq = (s0, s1, s2) if s0 >= 0 else ()
        legal_actions.append(Action(ActionType(k), t, seq, d))

    obs = {
        "player_id": player_id,
        "current_player": player_id,
        "phase": phase,
        "hand": hand,
        "melds": melds,
        "discards": discards,
        "wall_remaining": wall_rem,
        "wall_remaining_by_player": [wall_rem // 4] * 4,
        "prevalent_wind": prevalent_wind,
        "events": events,
        "last_discard": last_discard,
        "wall_last": wall_last,
        "about_kong": about_kong,
        "claim_hu_only": claim_hu_only,
    }
    return obs, legal_actions


def load_botzone_model(ckpt_path: str):
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("metadata", {})
    cfg = payload.get("model_config") or meta.get("model_config") or {
        "token_size": 12, "d_model": 256, "layers": 6, "heads": 8,
        "dropout": 0.1, "belief_mode": "aux", "auxiliary_logit_fusion": True,
    }
    model = TokenTransformer(**cfg)
    state = payload.get("model") or payload.get("model_state") or payload
    model.load_state_dict(state)
    model.eval()
    return model


def choose_action_index(model, obs, legal_actions) -> int:
    if len(legal_actions) <= 1:
        return 0
    state_tokens, state_mask = encode_observation_v2(obs)
    encoded_actions = [encode_action_v2(a) for a in legal_actions]
    action_tokens = [item[0] for item in encoded_actions]
    action_token_mask = [item[1] for item in encoded_actions]
    action_mask = [1] * len(legal_actions)

    with torch.inference_mode():
        features_t = torch.tensor([state_tokens], dtype=torch.float32)
        feature_mask_t = torch.tensor([state_mask], dtype=torch.long)
        actions_t = torch.tensor([action_tokens], dtype=torch.float32)
        action_token_mask_t = torch.tensor([action_token_mask], dtype=torch.long)
        action_mask_t = torch.tensor([action_mask], dtype=torch.long)

        out = model(
            features_t, actions_t,
            action_mask=action_mask_t,
            feature_mask=feature_mask_t,
            action_token_mask=action_token_mask_t,
        )
        logits = out["logits"][0]
        return int(torch.argmax(logits).item())


def serve(port: int, ckpt_path: str):
    model = load_botzone_model(ckpt_path)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    print(f"[McrBaselineServer] Loaded {ckpt_path}", flush=True)
    print(f"[McrBaselineServer] Listening on 127.0.0.1:{port}... Ready!", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[McrBaselineServer] Accepted eval connection from {addr}...", flush=True)
            buf = b""
            moves = 0
            try:
                while True:
                    while b"\n" not in buf:
                        chunk = conn.recv(1 << 20)
                        if not chunk:
                            break
                        buf += chunk
                    if not buf:
                        break
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode().strip()
                    if not text or text == "QUIT":
                        break
                    obs, legal = parse_request(text)
                    idx = choose_action_index(model, obs, legal)
                    conn.sendall(f"{idx}\n".encode())
                    moves += 1
            except Exception as exc:
                print(f"[McrBaselineServer] Connection error: {exc}", flush=True)
            finally:
                conn.close()
                print(f"[McrBaselineServer] Finished eval session ({moves} moves handled)", flush=True)
    finally:
        srv.close()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    default_ckpt = str(ROOT / "baselines" / "botzone_mcr" / "botzone_model_bc.pt")
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--ckpt", default=default_ckpt)
    a = p.parse_args()
    serve(a.port, a.ckpt)
