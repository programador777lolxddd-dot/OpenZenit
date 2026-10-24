import argparse, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(4, os.cpu_count() or 1))
ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / 'data' / 'corpus.txt'
CHECKPOINT = ROOT / 'mini_codegpt.pt'

class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text)) or [' ']
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]
    def decode(self, ids):
        return ''.join(self.itos.get(int(i), '') for i in ids)

class MiniCodeGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, heads=4, layers=2, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4*d_model, dropout=0.0, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))
    def forward(self, x, targets=None):
        _, t = x.shape
        x = self.token_emb(x) + self.pos_emb(torch.arange(t, device=x.device))[None, :, :]
        logits = self.head(self.norm(self.transformer(x, mask=self.mask[:t, :t])))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

def main():
    parser = argparse.ArgumentParser(description='Mini-CodeGPT educativo')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--no-chat', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    text = CORPUS.read_text(encoding='utf-8')
    tokenizer = Tokenizer(text)
    config = {'d_model': 64, 'heads': 4, 'layers': 2, 'block_size': 128}
    if CHECKPOINT.exists() and not args.retrain:
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        tokenizer.chars = saved['chars']
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
        config = saved['config']
        model = MiniCodeGPT(len(tokenizer.chars), **config)
        model.load_state_dict(saved['model'])
        print('Checkpoint cargado.')
    else:
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(data) < 130:
            data = data.repeat(130 // max(1, len(data)) + 1)
        model = MiniCodeGPT(len(tokenizer.chars), **config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        model.train(); start = time.time()
        for step in range(args.steps):
            starts = torch.randint(0, len(data)-129, (16,))
            x = torch.stack([data[i:i+128] for i in starts])
            y = torch.stack([data[i+1:i+129] for i in starts])
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step == 0 or (step+1) % 100 == 0: print(f'Paso {step+1}/{args.steps} | pérdida {loss.item():.4f}')
        torch.save({'model': model.state_dict(), 'chars': tokenizer.chars, 'config': config}, CHECKPOINT)
        print(f'Entrenamiento terminado en {time.time()-start:.1f}s')
    if args.no_chat: return
    @torch.inference_mode()
    def generate(prompt, max_new_tokens=200):
        model.eval(); ids = tokenizer.encode(prompt) or [0]; idx = torch.tensor([ids])
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -config['block_size']:])
            probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1).view(1, 1)], dim=1)
        return tokenizer.decode(idx[0].tolist())
    print('OpenZenit — escribe /quit para salir')
    while True:
        try: prompt = input('\nTú> ')
        except (EOFError, KeyboardInterrupt): break
        if prompt.strip() == '/quit': break
        print('\nOpenZenit> ' + generate(prompt))

if __name__ == '__main__': main()