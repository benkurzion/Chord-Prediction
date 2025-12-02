import torch
from torch.nn.functional import cross_entropy
from model import ChordTransformer
from data import ChordDataModule
import math

def evaluate_model(model, dataloader, pad_idx, device="cuda", k=5):
    model.eval()
    model.to(device)

    total_tokens = 0
    correct_top1 = 0
    correct_topk = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)          # (B, T)
            tgt = batch["tgt"].to(device)          # (B, T)
            mask = batch["src_mask"].to(device)    # (B, T), 1 for real tokens
            
            logits = model(src, src_padding_mask=mask)     # (B, T, vocab)
            
            # Flatten for CE loss
            B, T, V = logits.shape
            logits_flat = logits.reshape(B*T, V)
            tgt_flat = tgt.reshape(B*T)

            loss = cross_entropy(
                logits_flat,
                tgt_flat,
                ignore_index=pad_idx,
                reduction="sum"
            )
            total_loss += loss.item()

            nonpad_mask = (tgt_flat != pad_idx)
            total_tokens += nonpad_mask.sum().item()

            top1_pred = logits.argmax(dim=-1).reshape(B*T)
            correct_top1 += (top1_pred[nonpad_mask] == tgt_flat[nonpad_mask]).sum().item()

            topk_pred = logits.topk(k, dim=-1).indices.reshape(B*T, k)
            tgt_expanded = tgt_flat.unsqueeze(-1)
            correct_topk += (topk_pred == tgt_expanded).any(dim=-1)[nonpad_mask].sum().item()

    avg_loss = total_loss / total_tokens
    top1_acc = correct_top1 / total_tokens
    topk_acc = correct_topk / total_tokens
    ppl = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "top1_accuracy": top1_acc,
        f"top{k}_accuracy": topk_acc
    }

def run_evaluations():
    ckpt_path = "/mnt/pccfs2/backed_up/justinolcott/superviren/vmod-john/chord-pred/Chord-Prediction/logs/chord_transformer_pop/version_0/checkpoints/chord-transformer-pop-epoch=27-val_loss=1.80.ckpt"

    dm = ChordDataModule(
        batch_size=64,
        max_seq_len=256,
        min_freq=5,
        num_workers=4
    )
    dm.setup()
    print("Loaded Datamodule")

    model = ChordTransformer.load_from_checkpoint(ckpt_path, vocab_size=len(dm.vocab))

    dm.setup("test")
    test_loader = dm.test_dataloader()
    pad_idx = dm.vocab.token2idx["[PAD]"]
    results = evaluate_model(model, test_loader, pad_idx, device="cuda", k=5)
    print(results)

if __name__ == "__main__":
    run_evaluations()