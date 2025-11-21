import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from collections import Counter
import re
from typing import Dict, List, Tuple, Optional
import math
import pickle
from pathlib import Path

class ChordVocabulary:
    """Builds and manages chord vocabulary."""
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[SEP]': 4 
        }
        self.genre2idx = {}
        self.idx2genre = {}
        
    def build_vocab(self, chord_sequences: List[str], genres: List[str] = None):
        """Build vocabulary from chord sequences and genres."""
        # Initialize with special tokens
        self.token2idx = self.special_tokens.copy()
        self.idx2token = {v: k for k, v in self.special_tokens.items()}
        
        # Count all chords
        chord_counter = Counter()
        for seq in chord_sequences:
            chords = self._parse_chords(seq)
            chord_counter.update(chords)
        
        # Add chords that meet minimum frequency
        idx = len(self.special_tokens)
        for chord, freq in chord_counter.most_common():
            if freq >= self.min_freq:
                self.token2idx[chord] = idx
                self.idx2token[idx] = chord
                idx += 1
        
        # Build genre vocabulary
        if genres:
            unique_genres = sorted(set(g for g in genres if g and g != 'unknown'))
            self.genre2idx = {genre: i for i, genre in enumerate(unique_genres)}
            self.genre2idx['unknown'] = len(self.genre2idx)
            self.idx2genre = {i: genre for genre, i in self.genre2idx.items()}
            print(f"Number of genres: {len(self.genre2idx)}")
            print(f"Genres: {list(self.genre2idx.keys())}")
                
        print(f"Vocabulary size: {len(self.token2idx)}")
        print(f"Most common chords: {chord_counter.most_common(20)}")
        
    def _parse_chords(self, chord_string: str) -> List[str]:
        """Parse chord string, removing section markers."""
        # Split by spaces
        tokens = chord_string.split()
        # Keep only chords, not section markers like <verse_1>
        chords = [t for t in tokens if not (t.startswith('<') and t.endswith('>'))]
        return chords
    
    def encode(self, chord_string: str, add_special_tokens: bool = True) -> List[int]:
        """Convert chord string to token IDs."""
        chords = self._parse_chords(chord_string)
        
        # Convert to IDs
        ids = [self.token2idx.get(c, self.token2idx['[UNK]']) for c in chords]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.token2idx['[BOS]']] + ids + [self.token2idx['[EOS]']]
            
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to chord string."""
        tokens = [self.idx2token.get(idx, '[UNK]') for idx in token_ids]
        
        if skip_special_tokens:
            special = set(self.special_tokens.keys())
            tokens = [t for t in tokens if t not in special]
            
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.token2idx)


class ChordDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for chord progressions with caching."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_seq_len: int = 256,
        min_freq: int = 2,
        num_workers: int = 0,
        train_split: float = 0.8,
        val_split: float = 0.1,
        cache_dir: str = './data'
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = None
        
    def prepare_data(self):
        """Download dataset (called on 1 GPU/process)."""
        # Cache HuggingFace dataset
        load_dataset("ailsntua/Chordonomicon", cache_dir=str(self.cache_dir / 'Chordonomicon'))
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets (called on every GPU)."""
        # Check for cached processed data
        cache_file = self.cache_dir / f'processed_seq{self.max_seq_len}_freq{self.min_freq}.pkl'
        
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            self.vocab = cached['vocab']
            self.train_dataset = cached['train_dataset']
            self.val_dataset = cached['val_dataset']
            self.test_dataset = cached['test_dataset']
            print(f"✓ Cache loaded! Vocab size: {len(self.vocab)}, "
                  f"Train: {len(self.train_dataset)}, "
                  f"Val: {len(self.val_dataset)}, "
                  f"Test: {len(self.test_dataset)}")
            return
        
        print("Processing data for first time (will be cached)...")
        
        # Load dataset with cache
        dataset = load_dataset("ailsntua/Chordonomicon", cache_dir=str(self.cache_dir / 'hf_cache'))
        
        # Get all chord sequences and genres for vocab building
        all_chords = dataset['train']['chords']
        all_genres = dataset['train']['main_genre']
        
        # Build vocabulary
        print("Building vocabulary...")
        if self.vocab is None:
            self.vocab = ChordVocabulary(min_freq=self.min_freq)
            self.vocab.build_vocab(all_chords, all_genres)
        
        # Split dataset
        print("Splitting dataset...")
        full_dataset = dataset['train']
        total = len(full_dataset)
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        
        splits = full_dataset.train_test_split(
            test_size=(val_size + (total - train_size - val_size)),
            seed=42
        )
        train_data = splits['train']
        remaining = splits['test']
        
        val_test_splits = remaining.train_test_split(
            test_size=0.5,
            seed=42
        )
        val_data = val_test_splits['train']
        test_data = val_test_splits['test']
        
        # Process datasets
        print("Encoding train data...")
        self.train_dataset = self._process_dataset(train_data)
        print("Encoding val data...")
        self.val_dataset = self._process_dataset(val_data)
        print("Encoding test data...")
        self.test_dataset = self._process_dataset(test_data)
        
        # Save to cache
        print(f"Saving cache to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'train_dataset': self.train_dataset,
                'val_dataset': self.val_dataset,
                'test_dataset': self.test_dataset
            }, f)
        print("✓ Cache saved! Next run will be much faster.")
        
    def _process_dataset(self, dataset):
        """Process dataset by encoding chord sequences."""
        encoded_data = []
        
        for example in dataset:
            # Encode chords
            token_ids = self.vocab.encode(example['chords'])
            
            # Truncate if too long
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            
            # Get genre ID
            genre = example.get('main_genre', 'unknown')
            if not genre or genre == '':
                genre = 'unknown'
            genre_id = self.vocab.genre2idx.get(genre, self.vocab.genre2idx.get('unknown', 0))
                
            encoded_data.append({
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'genre': genre,
                'genre_id': genre_id,
                'decade': example.get('decade', -1.0),
            })
            
        return encoded_data
    
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        # Pad sequences
        input_ids = [item['input_ids'] for item in batch]
        input_ids_padded = nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.vocab.token2idx['[PAD]']
        )
        
        # Create attention mask
        attention_mask = (input_ids_padded != self.vocab.token2idx['[PAD]']).long()
        
        # Get genre IDs
        genre_ids = torch.tensor([item['genre_id'] for item in batch], dtype=torch.long)
        
        # For autoregressive training: input is all tokens except last, target is all except first
        src = input_ids_padded[:, :-1]
        tgt = input_ids_padded[:, 1:]
        src_mask = attention_mask[:, :-1]
        
        return {
            'src': src,
            'tgt': tgt,
            'src_mask': src_mask,
            'genre_ids': genre_ids,
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False  # Reuse workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def clear_cache(self):
        """Clear cached processed data."""
        cache_file = self.cache_dir / f'processed_seq{self.max_seq_len}_freq{self.min_freq}.pkl'
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cache cleared: {cache_file}")