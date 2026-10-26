# openzenit.py
# OpenZenit — mini GPT educativo para generación de código Python.
# Versión ultra ligera: < 10,000 parámetros.

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Corpus de entrenamiento integrado
# ---------------------------------------------------------------
CORPUS_TEXT = """# Funciones básicas
def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b

def multiplicar(a, b):
    return a * b

def dividir(a, b):
    if b == 0:
        return None
    return a / b

# Condicionales
def es_par(n):
    if n % 2 == 0:
        return True
    return False

def maximo(a, b):
    if a > b:
        return a
    return b

# Bucles
def suma_hasta(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

def contar_vocales(texto):
    contador = 0
    for letra in texto.lower():
        if letra in "aeiou":
            contador += 1
    return contador

# Fibonacci
def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    secuencia = [0, 1]
    while len(secuencia) < n:
        secuencia.append(secuencia[-1] + secuencia[-2])
    return secuencia

# Listas y comprensión
def cuadrados(n):
    return [i * i for i in range(n)]

def filtrar_pares(lista):
    return [x for x in lista if x % 2 == 0]

# Diccionarios
def contar_frecuencias(lista):
    frecuencias = {}
    for elemento in lista:
        frecuencias[elemento] = frecuencias.get(elemento, 0) + 1
    return frecuencias

# Clases
class Rectangulo:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto

    def area(self):
        return self.ancho * self.alto

# Recursión
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

# ---------------------------------------------------------------
# Configuración ultra ligera de OpenZenit
# ---------------------------------------------------------------
CONFIG = {
    'block_size': 64,    # Reducido a la mitad
    'd_model':    16,    # Reducido drásticamente (antes 64)
    'heads':      2,     # Reducido (antes 4)
    'layers':     1,     # Solo 1 capa (antes 2)
    'dropout':    0.1,
    'lr':         3e-3,  # Subimos el LR porque el modelo es más pequeño
    'steps':      1000,
    'batch_size': 16,
}

HERE      = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, 'openzenit.pt')


# ---------------------------------------------------------------
# Tokenizer a nivel de carácter
# ---------------------------------------------------------------
class Tokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '') for i in ids)


# ---------------------------------------------------------------
# Bloque Transformer de OpenZenit
# ---------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------
# Modelo OpenZenit
# ---------------------------------------------------------------
class OpenZenit(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, heads, layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks  = nn.ModuleList([Block(d_model, heads, dropout) for _ in range(layers)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


# ---------------------------------------------------------------
# Datos y Batches
# ---------------------------------------------------------------
def build_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size]         for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------
def train(model, tokenizer, text, args):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    model.train()
    for step in range(args.steps):
        x, y = build_batch(data, CONFIG['block_size'], CONFIG['batch_size'])
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0:
            print(f'[OpenZenit] paso {step+1:4d}/{args.steps} | pérdida {loss.item():.4f}')

    torch.save({
        'model_state':    model.state_dict(),
        'tokenizer_stoi': tokenizer.stoi,
        'tokenizer_itos': tokenizer.itos,
        'config':         CONFIG,
    }, CKPT_PATH)
    print(f'[OpenZenit] checkpoint guardado en {CKPT_PATH}')


# ---------------------------------------------------------------
# Generación
# ---------------------------------------------------------------
@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=200):
    model.eval()
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -CONFIG['block_size']:])
        next_id = torch.argmax(logits[0, -1], dim=-1, keepdim=True).view(1, 1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


# ---------------------------------------------------------------
# Carga de checkpoint o entrenamiento
# ---------------------------------------------------------------
def load_or_train(args):
    text = CORPUS_TEXT

    if os.path.exists(CKPT_PATH) and not args.retrain:
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        tokenizer = Tokenizer.__new__(Tokenizer)
        tokenizer.stoi       = ckpt['tokenizer_stoi']
        tokenizer.itos       = ckpt['tokenizer_itos']
        tokenizer.vocab_size = len(tokenizer.stoi)
        CONFIG.update(ckpt.get('config', CONFIG))

        model = OpenZenit(
            tokenizer.vocab_size,
            CONFIG['block_size'],
            CONFIG['d_model'],
            CONFIG['heads'],
            CONFIG['layers'],
            CONFIG['dropout'],
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f'[OpenZenit] checkpoint cargado desde {CKPT_PATH}')
        return model, tokenizer

    tokenizer = Tokenizer(text)
    model = OpenZenit(
        tokenizer.vocab_size,
        CONFIG['block_size'],
        CONFIG['d_model'],
        CONFIG['heads'],
        CONFIG['layers'],
        CONFIG['dropout'],
    )
    
    # Imprimir el conteo de parámetros
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[OpenZenit] Parámetros del modelo: {num_params:,}')
    
    train(model, tokenizer, text, args)
    return model, tokenizer


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='OpenZenit — mini GPT educativo')
    parser.add_argument('--retrain', action='store_true', help='Reentrenar desde cero')
    parser.add_argument('--steps',   type=int, default=CONFIG['steps'])
    parser.add_argument('--prompt',  type=str, default=None, help='Generar texto y salir')
    parser.add_argument('--tokens',  type=int, default=100, help='Tokens máximos a generar')
    args = parser.parse_args()

    model, tokenizer = load_or_train(args)

    if args.prompt is not None:
        print(generate(model, tokenizer, args.prompt, args.tokens))
        return

    print('\nOpenZenit listo. Escribe "salir" para terminar.\n')
    while True:
        try:
            prompt = input('Tú> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {'salir', 'exit', 'quit'}:
            break
        print('OpenZenit>', generate(model, tokenizer, prompt, args.tokens), '\n')


if __name__ == '__main__':
    main()
