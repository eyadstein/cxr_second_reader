"""Compact experimental image-to-report models for notebooks 02 and 03."""

from __future__ import annotations

import re
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class ReportVocabulary:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, texts, max_size=20000, min_frequency=2):
        counts = Counter()
        for text in texts:
            counts.update(re.findall(r"[a-z0-9]+|[^\w\s]", str(text).lower()))
        tokens = [token for token, count in counts.most_common() if count >= min_frequency]
        self.tokens = [self.PAD, self.BOS, self.EOS, self.UNK] + tokens[: max_size - 4]
        self.to_id = {token: index for index, token in enumerate(self.tokens)}
        self.to_token = self.tokens

    def encode(self, text, max_length=128):
        words = re.findall(r"[a-z0-9]+|[^\w\s]", str(text).lower())
        ids = [self.to_id[self.BOS]]
        ids.extend(self.to_id.get(word, self.to_id[self.UNK]) for word in words[: max_length - 2])
        ids.append(self.to_id[self.EOS])
        return ids

    def decode(self, ids):
        words = []
        for token_id in ids:
            token = self.to_token[int(token_id)]
            if token == self.EOS:
                break
            if token not in {self.PAD, self.BOS}:
                words.append(token)
        text = " ".join(words)
        return re.sub(r"\s+([,.!?;:])", r"\1", text)

    @property
    def pad_id(self):
        return self.to_id[self.PAD]

    @property
    def bos_id(self):
        return self.to_id[self.BOS]

    @property
    def eos_id(self):
        return self.to_id[self.EOS]


def report_collate(batch, vocab, max_length=128):
    images = torch.stack([item["image"] for item in batch])
    encoded = [vocab.encode(item["text"], max_length=max_length) for item in batch]
    width = max(map(len, encoded))
    targets = torch.full((len(batch), width), vocab.pad_id, dtype=torch.long)
    for row, ids in enumerate(encoded):
        targets[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return images, targets


class AdditiveAttentionLSTM(nn.Module):
    """Frozen DenseNet-121 spatial features with an additive-attention LSTM."""

    def __init__(self, vocab_size, pad_id, embedding_dim=256, hidden_dim=512,
                 pretrained=True, freeze_encoder=True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        self.encoder = backbone.features
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.key = nn.Linear(1024, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.initial_h = nn.Linear(1024, hidden_dim)
        self.initial_c = nn.Linear(1024, hidden_dim)
        self.cell = nn.LSTMCell(embedding_dim + 1024, hidden_dim)
        self.output = nn.Linear(hidden_dim + 1024, vocab_size)

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen pretrained encoder's BatchNorm statistics fixed.
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def image_features(self, images):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)
        return F.relu(features, inplace=False).flatten(2).transpose(1, 2)

    def _step(self, features, hidden, cell, previous_token):
        scores = self.attention_score(
            torch.tanh(self.key(features) + self.query(hidden).unsqueeze(1))
        ).squeeze(-1)
        weights = scores.softmax(dim=1)
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        hidden, cell = self.cell(
            torch.cat([self.embedding(previous_token), context], dim=-1),
            (hidden, cell),
        )
        logits = self.output(torch.cat([hidden, context], dim=-1))
        return logits, hidden, cell

    def forward(self, images, token_ids):
        features = self.image_features(images)
        pooled = features.mean(dim=1)
        hidden, cell = torch.tanh(self.initial_h(pooled)), torch.tanh(self.initial_c(pooled))
        logits = []
        for step in range(token_ids.shape[1] - 1):
            step_logits, hidden, cell = self._step(features, hidden, cell, token_ids[:, step])
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def generate(self, images, bos_id, eos_id, max_new_tokens=128):
        features = self.image_features(images)
        pooled = features.mean(dim=1)
        hidden, cell = torch.tanh(self.initial_h(pooled)), torch.tanh(self.initial_c(pooled))
        token = torch.full((images.shape[0],), bos_id, device=images.device, dtype=torch.long)
        outputs, finished = [], torch.zeros(images.shape[0], device=images.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits, hidden, cell = self._step(features, hidden, cell, token)
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, eos_id), token)
            outputs.append(token)
            finished |= token.eq(eos_id)
            if finished.all():
                break
        return torch.stack(outputs, dim=1)
