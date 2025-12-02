from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import ChordTransformer
from data import ChordVocabulary

app = FastAPI()

# Load everything ONCE
ckpt = "model/chord_transformer.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChordTransformer.load_from_checkpoint(ckpt).to(device).eval()
vocab = model.vocab  # assuming you stored vocab in LightningModule

class GenerateRequest(BaseModel):
    genre: str = "pop"
    max_length: int = 64
    temperature: float = 1.0
    top_k: int = 8

@app.post("/generate")
def generate(req: GenerateRequest):

    prefix_token = vocab.token2idx.get(f"[GENRE_{req.genre.upper()}]", 
                                       vocab.token2idx["[GENRE_UNKNOWN]"])

    input_ids = torch.tensor([[prefix_token, vocab.token2idx["[BOS]"]]], 
                             dtype=torch.long).to(device)

    # autoregressive loop
    for _ in range(req.max_length):
        logits = model(input_ids)[:, -1, :] / req.temperature
        topk = torch.topk(logits, req.top_k)
        probs = torch.softmax(topk.values, dim=-1)
        next_id = topk.indices[torch.multinomial(probs, 1)]
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

        if next_id.item() == vocab.token2idx["[EOS]"]:
            break

    decoded = vocab.decode(input_ids[0].tolist())

    return {"generated_chords": decoded}
