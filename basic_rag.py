"""
RAG (from scratch, no APIs)

Run:
  python basic_rag.py --docs path/to/folder --query "the question"
"""

import os, re, math, argparse, glob, textwrap
from collections import Counter, defaultdict
import numpy as np

# ----------------------------- utils -----------------------------
def read_folder(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
    texts = []
    for f in files:
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                if txt.strip():
                    texts.append((os.path.basename(f), txt))
            except Exception:
                pass
    return texts

def sent_split(s):
    s = re.sub(r"\s+", " ", s.strip())
    parts = re.split(r"(?<=[.!?])\s+", s) if s else []
    return [p.strip() for p in parts if p.strip()]

def tokenize(s):
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks

# ----------------------------- chunker -----------------------------
def chunk_doc(doc_text, L=160, S=120):
    toks = tokenize(doc_text)
    chunks = []
    for i in range(0, max(1, len(toks) - L + 1), S):
        chunk_tokens = toks[i:i+L]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
    if not chunks and toks:
        chunks = [" ".join(toks)]
    return chunks

# ----------------------------- tf-idf -----------------------------
class Tfidf:
    def __init__(self):
        self.idf = {}
        self.vocab = {}

    def fit(self, texts):
        df = Counter()
        for t in texts:
            terms = set(tokenize(t))
            df.update(terms)
        N = len(texts)
        vocab = sorted(df.keys())
        self.vocab = {t:i for i,t in enumerate(vocab)}
        self.idf = {}
        for t, dft in df.items():
            self.idf[t] = math.log((1+N)/(1+dft)) + 1.0

    def transform(self, texts):
        V = len(self.vocab)
        mats = np.zeros((len(texts), V), dtype=np.float64)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if not toks:
                continue
            counts = Counter(toks)
            L = float(len(toks))
            for w, c in counts.items():
                j = self.vocab.get(w)
                if j is None: continue
                tf = c / L
                mats[i, j] = tf * self.idf.get(w, 0.0)
        norms = np.linalg.norm(mats, axis=1, keepdims=True) + 1e-12
        mats /= norms
        return mats

# ----------------------------- retriever -----------------------------
class Retriever:
    def __init__(self, chunks, meta):
        self.chunks = chunks
        self.meta = meta
        self.tfidf = Tfidf()
        self.tfidf.fit(chunks)
        self.X = self.tfidf.transform(chunks)

    def query(self, q, k=5):
        qv = self.tfidf.transform([q])
        sims = (self.X @ qv[0].T)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i]), self.chunks[i], self.meta[i]) for i in idx]

# ----------------------------- textrank -----------------------------
class TextRankSummarizer:
    def __init__(self, d=0.85, iters=40):
        self.d = d
        self.iters = iters

    def summarize(self, text, k=3):
        sents = sent_split(text)
        if not sents:
            return []
        tf = Tfidf()
        tf.fit(sents)
        M = tf.transform(sents)
        W = M @ M.T
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True) + 1e-12
        P = W / row_sums
        n = len(sents)
        R = np.ones((n,1), dtype=np.float64) / n
        v = np.ones((n,1), dtype=np.float64) / n
        for _ in range(self.iters):
            R = (1-self.d)*v + self.d*(P.T @ R)
        order = np.argsort(-R.flatten())[:k]
        picked = sorted(order.tolist())
        return [sents[i] for i in picked]

# ----------------------------- RAG pipeline -----------------------------
class RAG:
    def __init__(self, docs, L=160, S=120, topk=5):
        chunks, meta = [], []
        for name, txt in docs:
            for ci, c in enumerate(chunk_doc(txt, L=L, S=S)):
                chunks.append(c)
                meta.append((name, ci))
        self.retriever = Retriever(chunks, meta)
        self.summarizer = TextRankSummarizer()
        self.topk = topk

    def ask(self, q, summary_sentences=3):
        hits = self.retriever.query(q, k=self.topk)
        ctx = "\n".join([h[2] for h in hits])
        summ = self.summarizer.summarize(ctx, k=summary_sentences)
        return hits, summ

# ----------------------------- demo -----------------------------
def build_docs_from_folder(folder):
    if folder and os.path.isdir(folder):
        return read_folder(folder)
    toy = [
        ("doc1.txt", """
        The Transformer architecture introduced self-attention, allowing each token to attend to all others. 
        Attention computes compatibility between queries and keys to mix values. Scaling by 1/sqrt(d_k) stabilizes gradients. 
        Multi-head attention projects inputs into multiple subspaces to capture diverse relations.
        """),
        ("doc2.txt", """
        In vector search, TF-IDF weighs terms by frequency and rarity. Cosine similarity compares angle between vectors. 
        Dense retrieval uses learned embeddings; sparse retrieval uses exact term features. Hybrid methods combine both.
        """),
        ("doc3.txt", """
        TextRank builds a graph of sentences with edges weighted by similarity and runs PageRank to score importance. 
        Extractive summarization selects the top sentences while keeping original wording.
        """),
        ("doc4.txt", """
        Retrieval-Augmented Generation first retrieves relevant passages from a corpus and then conditions generation on them. 
        Even a simple extractive generator can provide concise answers when combined with strong retrieval.
        """),
    ]
    return [(n, textwrap.dedent(t)) for n,t in toy]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', type=str, default='', help='folder of .txt/.md etc')
    ap.add_argument('--query', type=str, default='what is attention scaling and why?', help='question')
    ap.add_argument('--topk', type=int, default=4)
    ap.add_argument('--L', type=int, default=160)
    ap.add_argument('--S', type=int, default=120)
    ap.add_argument('--K', type=int, default=3, help='summary sentences')
    args = ap.parse_args()

    docs = build_docs_from_folder(args.docs)
    rag = RAG(docs, L=args.L, S=args.S, topk=args.topk)
    hits, summ = rag.ask(args.query, summary_sentences=args.K)

    print("\n== Query ==\n" + args.query)
    print("\n== Top Passages ==")
    for rank,(i,score,chunk,meta) in enumerate(hits,1):
        name, ci = meta
        preview = " ".join(chunk.split()[:60])
        print(f"[{rank}] {name}#chunk{ci}  sim={score:.4f}\n    {preview}...")
    print("\n== Answer (extractive) ==")
    for s in summ:
        print("- "+s)

if __name__ == '__main__':
    main()
