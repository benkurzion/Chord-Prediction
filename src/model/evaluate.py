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

def evaluate_genre(model, dataloader, vocab, pad_idx, device="cuda", k=5):
    """
    Evaluate model per genre and return metrics for each genre.
    
    Args:
        model: ChordTransformer
        dataloader: DataLoader yielding batches with 'src', 'tgt', 'src_mask', 'genre_ids'
        vocab: ChordVocabulary
        pad_idx: ID of [PAD] token
        device: cuda/cpu
        k: top-k accuracy
    Returns:
        Dictionary mapping genre -> metrics dict
    """
    model.eval()
    model.to(device)

    # Initialize per-genre accumulators
    genre_metrics = {genre: {"tokens": 0, "correct_top1": 0, "correct_topk": 0, "loss_sum": 0.0} 
                     for genre in vocab.genre2idx.keys()}

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)          # (B, T)
            tgt = batch["tgt"].to(device)          # (B, T)
            mask = batch["src_mask"].to(device)
            genre_ids = batch["genre_ids"].to(device)  # (B,)

            logits = model(src, src_padding_mask=mask)  # (B, T, vocab)
            B, T, V = logits.shape

            logits_flat = logits.reshape(B*T, V)
            tgt_flat = tgt.reshape(B*T)

            loss = cross_entropy(logits_flat, tgt_flat, ignore_index=pad_idx, reduction="none")
            loss = loss.reshape(B, T)

            top1_pred = logits.argmax(dim=-1)
            topk_pred = logits.topk(k, dim=-1).indices

            for i in range(B):
                genre_id = genre_ids[i].item()
                genre_name = vocab.idx2genre.get(genre_id, "unknown")

                nonpad_mask = (tgt[i] != pad_idx)
                num_tokens = nonpad_mask.sum().item()
                genre_metrics[genre_name]["tokens"] += num_tokens

                genre_metrics[genre_name]["loss_sum"] += loss[i][nonpad_mask].sum().item()
                genre_metrics[genre_name]["correct_top1"] += (top1_pred[i][nonpad_mask] == tgt[i][nonpad_mask]).sum().item()
                genre_metrics[genre_name]["correct_topk"] += (topk_pred[i][nonpad_mask].eq(tgt[i][nonpad_mask].unsqueeze(-1))).any(dim=-1).sum().item()

    # Compute final metrics
    results = {}
    for genre, m in genre_metrics.items():
        if m["tokens"] > 0:
            avg_loss = m["loss_sum"] / m["tokens"]
            top1_acc = m["correct_top1"] / m["tokens"]
            topk_acc = m["correct_topk"] / m["tokens"]
            ppl = math.exp(avg_loss)
            results[genre] = {
                "loss": avg_loss,
                "perplexity": ppl,
                "top1_accuracy": top1_acc,
                f"top{k}_accuracy": topk_acc
            }
        else:
            results[genre] = {
                "loss": None,
                "perplexity": None,
                "top1_accuracy": None,
                f"top{k}_accuracy": None
            }

    return results

def run_evaluations():
    ckpt_path = "/mnt/pccfs2/backed_up/justinolcott/superviren/vmod-john/chord-pred/Chord-Prediction/logs/chord_transformer_full_w_genre/version_0/checkpoints/chord-transformer-full_w_genre-epoch=22-val_loss=1.00.ckpt"

    dm = ChordDataModule(
        batch_size=1,
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
    results = evaluate_genre(model, test_loader, dm.vocab, pad_idx, device="cuda", k=5)
    print(results)

if __name__ == "__main__":
    run_evaluations()