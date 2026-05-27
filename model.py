import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        x = inputs.transpose(-1, -2)
        x = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x)))))
        return x.transpose(-1, -2)


class COAST(torch.nn.Module):
    def __init__(self, item_num, content_emb, args):
        super().__init__()
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.hybrid = getattr(args, "hybrid", True)
        content_dim = content_emb.shape[1]
        hidden = args.hidden_units

        self.register_buffer(
            "content_emb",
            torch.tensor(content_emb, dtype=torch.float32),
        )
        self.item_proj = torch.nn.Linear(content_dim, hidden)
        if self.hybrid:
            self.id_emb = torch.nn.Embedding(item_num + 1, hidden, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, hidden, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(hidden, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(hidden, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(hidden, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden, args.dropout_rate))

    def item_vec(self, ids, mask_unseen_id=False, seen_train=None):
        ids_t = torch.as_tensor(ids, device=self.dev, dtype=torch.long)
        content = self.item_proj(self.content_emb[ids_t])
        if not self.hybrid:
            return content

        id_part = self.id_emb(ids_t)
        if mask_unseen_id and seen_train is not None:
            flat = np.asarray(ids).reshape(-1)
            mask = torch.tensor(
                [1.0 if int(i) in seen_train else 0.0 for i in flat],
                device=self.dev,
                dtype=id_part.dtype,
            ).view(id_part.shape[:-1] + (1,))
            id_part = id_part * mask
        return id_part + content

    def log2feats(self, log_seqs, mask_unseen_id=False, seen_train=None):
        seqs = self.item_vec(log_seqs, mask_unseen_id=mask_unseen_id, seen_train=seen_train)
        seqs *= self.item_proj.out_features ** 0.5

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= log_seqs != 0
        seqs += self.pos_emb(torch.as_tensor(poss, device=self.dev, dtype=torch.long))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha, _ = self.attention_layers[i](x, x, x, attn_mask=mask)
                seqs = seqs + mha
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=mask)
                seqs = self.attention_layernorms[i](seqs + mha)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_vec(pos_seqs)
        neg_embs = self.item_vec(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, seen_train=None):
        mask_unseen = self.hybrid and seen_train is not None
        log_feats = self.log2feats(
            log_seqs, mask_unseen_id=mask_unseen, seen_train=seen_train
        )
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_vec(
            item_indices, mask_unseen_id=mask_unseen, seen_train=seen_train
        )
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
