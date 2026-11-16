import re
import torch
import torch.nn as nn

# чтение и очистка текста
with open("saltan.txt", "rb") as f:
    raw = f.read()

for enc in ("utf-8", "cp1251", "koi8-r", "cp866"):
    try:
        text = raw.decode(enc)
        print("Подошла кодировка:", enc)
        break
    except UnicodeDecodeError:
        continue

text = text.lower()
# оставляем только русские буквы, пробелы и переводы строк
text = re.findall(r"[а-яё\s]", text)
text = "".join(text)
text = re.sub(r"\s+", " ", text).strip()

# токенизация по словам
words = text.split()
vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"Всего слов: {len(words)}, словарь: {vocab_size}")

# кодируем весь текст в индексы
data = torch.tensor([word2idx[w] for w in words], dtype=torch.long)

# модель
class WordRNN(nn.Module):
    def __init__(self, vocab_size, emb_size=64, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # вход ячейки: [embedding; prev_hidden]
        self.cell = nn.Linear(emb_size + hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x, prev):
        # x: (batch,) индексы слов; prev: (batch, hidden)
        emb = self.embedding(x)                    # (batch, emb)
        h = self.activation(self.cell(torch.cat([emb, prev], dim=1)))
        logits = self.output(h)                    # (batch, vocab)
        return logits, h

    def fit(self, data, seq_len=25, epochs=3, batch_size=64):
        self.train()
        n = len(data) - seq_len - 1
        # индексы стартов окон
        starts = torch.arange(0, n)

        for epoch in range(epochs):
            perm = starts[torch.randperm(len(starts))]
            epoch_loss = 0.0
            n_batches = 0

            for b in range(0, len(perm), batch_size):
                batch_starts = perm[b:b + batch_size]
                batch_size_cur = len(batch_starts)

                # собираем окна: (batch, seq_len)
                windows = torch.stack([
                    data[s:s + seq_len] for s in batch_starts
                ])
                targets = torch.stack([
                    data[s + seq_len] for s in batch_starts
                ])

                prev = torch.zeros(batch_size_cur, self.hidden_size)

                # проходим по всем словам окна
                for t in range(seq_len):
                    logits, prev = self.forward(windows[:, t], prev)

                loss = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                # обрезка градиентов — полезно для RNN
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if b % (batch_size * 50) == 0:
                    print(f"epoch {epoch} | {b}/{len(perm)} | loss {loss.item():.4f}")

            print(f"=== epoch {epoch} done, avg loss {epoch_loss / n_batches:.4f} ===")

    @torch.no_grad()
    def predict_next(self, context_words, temperature=1.0):
        self.eval()
        prev = torch.zeros(1, self.hidden_size)
        for w in context_words:
            idx = torch.tensor([word2idx.get(w, 0)])
            _, prev = self.forward(idx, prev)
        emb = self.embedding(torch.tensor([word2idx.get(context_words[-1], 0)]))
        h = self.activation(self.cell(torch.cat([emb, prev], dim=1)))
        logits = self.output(h).squeeze(0)

        if temperature == 0:
            return idx2word[torch.argmax(logits).item()]

        probs = torch.softmax(logits / temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return idx2word[idx]

    @torch.no_grad()
    def generate(self, seed_words, n_words=30, temperature=0.8):
        context = list(seed_words)
        for _ in range(n_words):
            nxt = self.predict_next(context, temperature=temperature)
            context.append(nxt)
        return " ".join(context)


# обучение и проверка
model = WordRNN(vocab_size, emb_size=64, hidden_size=128)
model.fit(data, seq_len=25, epochs=3, batch_size=64)

seed = ["три", "девицы", "под", "окном"]
print("\nСгенерированный текст:")
print(model.generate(seed, n_words=40, temperature=0.8))