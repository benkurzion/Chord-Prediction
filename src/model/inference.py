from model import ChordTransformer
from data import ChordDataModule


def inference():
    ckpt_path = "/mnt/pccfs2/backed_up/justinolcott/superviren/vmod-john/chord-pred/Chord-Prediction/shared_weights/chord-transformer-epoch=29-val_loss=1.00.ckpt"

    dm = ChordDataModule(
        batch_size=64,
        max_seq_len=256,
        min_freq=5,
        num_workers=4
    )
    dm.setup()
    print("Loaded Datamodule")

    model = ChordTransformer.load_from_checkpoint(ckpt_path, vocab_size=len(dm.vocab))
    model.eval()
    model.freeze()
    print("Loaded and froze model")

    for i in range(5):
        print(f"Generated {i}")
        print(model.generate(dm.vocab, max_len=20, temperature=0.9))
    
    print("inference completed.")

if __name__ == "__main__":
    inference()

