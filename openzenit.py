import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════
# CONFIGURACIÓN "LITE / MENOS INTELIGENTE"
# ══════════════════════════════════════════════════════════
torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')

CKPT         = "mini_dummy.pt"
STEPS        = 800       # Entrena mucho más rápido
BATCH        = 8
LR           = 3e-3
D, HEADS, LAYERS, BLOCK = 16, 2, 1, 32  # ¡Modelo enano!

device = "cpu"
torch.manual_seed(1337)

# Corpus reducido para que no se vuelva loco con tanto texto
BUILTIN_CORPUS = r'''
def suma(a, b):
    return a + b
def resta(a, b):
    return a - b
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
def es_par(n):
    return n % 2 == 0
for i in range(10):
    print(i)
'''

# ══════════════════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════════════════
class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi  = {c: i for i, c in enumerate(self.chars)}
        self.itos  = {i: c for c, i in self.stoi.items()}
        self.V     = len(self.chars)
    @classmethod
    def from_chars(cls, chars):
        t = cls.__new__(cls)
        t.chars = list(chars)
        t.stoi  = {c: i for i, c in enumerate(t.chars)}
        t.itos  = {i: c for c, i in t.stoi.items()}
        t.V     = len(t.chars)
        return t
    def encode(self, s): return [self.stoi.get(c, 0) for c in s]
    def decode(self, ids): return "".join(self.itos.get(i, '?') for i in ids)

# ══════════════════════════════════════════════════════════
# MODELO ENANO
# ══════════════════════════════════════════════════════════
class MiniDummy(nn.Module):
    def __init__(self, V, d=D, heads=HEADS, block=BLOCK, layers=LAYERS):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(block, d)
        
        # Solo 1 capa, dimensión 16 (muy poca capacidad)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            batch_first=True, dropout=0.0, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, V)
        
        mask = torch.triu(torch.ones(block, block), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        return self.head(self.tr(x, mask=self.mask[:T, :T]))

# ══════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
def train():
    tok = Tokenizer(BUILTIN_CORPUS)
    data = torch.tensor(tok.encode(BUILTIN_CORPUS), dtype=torch.long)
    model = MiniDummy(tok.V)

    print(f"🧠 Modelo Enano | vocab: {tok.V} | params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    B = model.block

    def batch(bs=BATCH):
        ix = torch.randint(0, len(data) - B - 1, (bs,))
        x = torch.stack([data[i:i+B] for i in ix])
        y = torch.stack([data[i+1:i+B+1] for i in ix])
        return x, y

    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = batch()
        loss = F.cross_entropy(model(x).view(-1, tok.V), y.view(-1))
        opt.zero_grad(); loss.backward()
        opt.step()
        
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}/{STEPS}  loss={loss.item():.3f}  ({time.time()-t0:.1f}s)")

    model.eval()
    return tok, model

def save_model(model, tok):
    torch.save({"state_dict": model.state_dict(), "chars": tok.chars}, CKPT)

def load_model():
    data = torch.load(CKPT, map_location="cpu", weights_only=True)
    tok = Tokenizer.from_chars(data["chars"])
    model = MiniDummy(tok.V)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return tok, model

# ══════════════════════════════════════════════════════════
# GENERACIÓN RÁPIDA
# ══════════════════════════════════════════════════════════
@torch.inference_mode()
def generate(model, tok, prompt, length=150, temp=0.8):
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    
    for _ in range(length):
        ctx = idx[:, -model.block:]
        logits = model(ctx)[:, -1, :] / max(temp, 0.01)
        probs = F.softmax(logits, -1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], 1)
        
        # Cortar si termina una función
        if len(idx[0]) > 10 and tok.decode([nxt.item()]) == '\n':
            if tok.decode(idx[0, -3:].tolist()).endswith('\n\n'):
                break
                
    return prompt + tok.decode(idx[0, len(ids):].tolist())

# ══════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════
C_CYAN, C_GREEN, C_RST = "\033[1;36m", "\033[1;32m", "\033[0m"

def chat_loop(model, tok):
    print(f"\n{C_CYAN}════════════════════════════════════════════{C_RST}")
    print(" 🤖 Mini-Dummy (Menos inteligente, ultra rápido)")
    print(f"{C_CYAN}════════════════════════════════════════════{C_RST}")
    print(" Escribe algo como 'def suma' o 'def factorial'.\n")

    while True:
        try:
            raw = input(f"{C_CYAN}🧑 >{C_RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw: continue
        if raw == "/quit": break

        print(f"\n{C_GREEN}🤖{C_RST}")
        print("─" * 40)
        print(generate(model, tok, raw))
        print("─" * 40)

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    retrain = "--retrain" in sys.argv
    if os.path.exists(CKPT) and not retrain:
        print(f"📂 Cargando modelo enano...")
        tok, model = load_model()
    else:
        print("🚀 Entrenando modelo enano...")
        tok, model = train()
        save_model(model, tok)
        print(f"💾 Guardado en {CKPT}")

    chat_loop(model, tok)