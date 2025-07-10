import os
import json
import random
import numpy as np
from collections import defaultdict, Counter
from gensim.models import Word2Vec
import signal
import sys

class SemanticSyntaxGraph:
    def __init__(self, max_context_len=5, vector_size=50):
        self.max_context_len = max_context_len
        self.graphs = {length: defaultdict(Counter) for length in range(1, max_context_len + 1)}
        self.vocab = set()
        self.sentences = []
        self.word_model = None
        self.vector_size = vector_size

    def tokenize(self, text):
        return [w.lower() for w in text.split() if w.isalpha()]

    def update_graphs(self, tokens):
        tokens = ['<s>'] * self.max_context_len + tokens + ['</s>']
        self.sentences.append(tokens)
        self.vocab.update(tokens)
        for length in range(1, self.max_context_len + 1):
            for i in range(len(tokens) - length):
                context = tuple(tokens[i:i + length])
                next_token = tokens[i + length]
                self.graphs[length][context][next_token] += 1

    def train_on_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = self.tokenize(line.strip())
                if tokens:
                    self.update_graphs(tokens)

    def train_on_directory(self, dirpath):
        files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        for filename in files:
            path = os.path.join(dirpath, filename)
            print(f"Training on file: {filename}")
            self.train_on_file(path)

    def train_word_vectors(self, epochs=50):
        if not self.sentences:
            print("Нет данных для обучения word2vec.")
            return
        self.word_model = Word2Vec(self.sentences, vector_size=self.vector_size, window=5, min_count=1, workers=2, epochs=epochs)

    def semantic_score(self, context_vec, word):
        if self.word_model is None or word not in self.word_model.wv:
            return 0.0
        word_vec = self.word_model.wv[word]
        score = np.dot(context_vec, word_vec) / (np.linalg.norm(context_vec) * np.linalg.norm(word_vec) + 1e-10)
        return float(score)

    def get_candidates(self, context):
        length = len(context)
        if length == 0 or length > self.max_context_len:
            return {}
        return self.graphs[length].get(tuple(context), {})

    def generate_next(self, context, prev_words=None):
        for length in range(min(len(context), self.max_context_len), 0, -1):
            sub_context = context[-length:]
            candidates = self.get_candidates(sub_context)
            if candidates:
                total = sum(candidates.values())
                if total == 0:
                    continue
                candidates_prob = {w: freq / total for w, freq in candidates.items()}
                if prev_words:
                    for w in list(candidates_prob.keys()):
                        if len(prev_words) >= 2 and prev_words[-1] == w and prev_words[-2] == w:
                            del candidates_prob[w]
                if not candidates_prob:
                    continue
                valid_context_words = [w for w in sub_context if self.word_model and w in self.word_model.wv]
                if valid_context_words:
                    context_vec = np.mean([self.word_model.wv[w] for w in valid_context_words], axis=0)
                else:
                    context_vec = None
                max_freq = max(candidates.values())
                scored = []
                for word, freq in candidates.items():
                    freq_weight = freq / max_freq
                    sem_score = self.semantic_score(context_vec, word) if context_vec is not None else 0.0
                    freq_coef = 0.7 + 0.1 * (length - 1)
                    freq_coef = min(freq_coef, 0.95)
                    sem_coef = 1 - freq_coef
                    combined_score = freq_coef * freq_weight + sem_coef * sem_score
                    scored.append((combined_score, word))
                scored.sort(reverse=True)
                return scored[0][1]
        fallback_vocab = self.vocab - {'<s>', '</s>'}
        return random.choice(list(fallback_vocab)) if fallback_vocab else None

    def generate_sentence(self, seed=None, max_len=50):
        seed_tokens = self.tokenize(seed) if seed else []
        context = ['<s>'] * max(0, self.max_context_len - len(seed_tokens)) + seed_tokens
        context = context[-self.max_context_len:]
        result = []
        for _ in range(max_len):
            next_token = self.generate_next(context, prev_words=result)
            if next_token == '</s>' or next_token is None:
                break
            result.append(next_token)
            context = context[1:] + [next_token]
        return ' '.join(result).strip()

    def online_train(self, text):
        tokens = self.tokenize(text)
        if tokens:
            self.update_graphs(tokens)
            self.sentences.append(['<s>'] * self.max_context_len + tokens + ['</s>'])
            if self.word_model:
                self.word_model.build_vocab([tokens], update=True)
                self.word_model.train([tokens], total_examples=1, epochs=5)
            else:
                self.train_word_vectors(epochs=20)

    def save(self, folder="model_data"):
        os.makedirs(folder, exist_ok=True)
        graph_to_save = {}
        for length, g in self.graphs.items():
            graph_to_save[str(length)] = {','.join(k): dict(v) for k, v in g.items()}
        with open(os.path.join(folder, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph_to_save, f, ensure_ascii=False, indent=2)
        with open(os.path.join(folder, "sentences.json"), "w", encoding="utf-8") as f:
            json.dump(self.sentences, f, ensure_ascii=False, indent=2)
        if self.word_model:
            self.word_model.save(os.path.join(folder, "word2vec.model"))
        print(f"Модель сохранена в папке '{folder}'.")

    def load(self, folder="model_data"):
        graph_path = os.path.join(folder, "graph.json")
        sentences_path = os.path.join(folder, "sentences.json")
        word2vec_path = os.path.join(folder, "word2vec.model")

        if os.path.exists(graph_path) and os.path.exists(sentences_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_loaded = json.load(f)
            self.graphs = {}
            for length_str, g in graph_loaded.items():
                length = int(length_str)
                new_g = defaultdict(Counter)
                for k_str, v in g.items():
                    key_tuple = tuple(k_str.split(','))
                    new_g[key_tuple] = Counter(v)
                self.graphs[length] = new_g
            with open(sentences_path, "r", encoding="utf-8") as f:
                self.sentences = json.load(f)
            self.vocab = set(w for sent in self.sentences for w in sent)
            print("Граф и предложения загружены.")

            if os.path.exists(word2vec_path):
                self.word_model = Word2Vec.load(word2vec_path)
                print("Word2Vec модель загружена.")
            else:
                print("Word2Vec модель не найдена, обучаем заново.")
                self.train_word_vectors(epochs=50)
        else:
            print("Данные модели не найдены, начнем с чистого листа.")

def interactive_mode(model):
    def signal_handler(sig, frame):
        print("\nCtrl+C пойман, сохраняем модель...")
        model.save()
        print("Выход.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Интерактивный режим модели. Введите текст для генерации или команду:")
    print("  /exit — выход (с автосохранением)")
    print("  /train <текст> — обучить модель на тексте онлайн")
    print("  /seed <начало предложения> — сгенерировать текст, начиная с этого")
    print("  /save — сохранить состояние модели")
    print("  /load — загрузить состояние модели из папки")
    print("  /help — показать эту подсказку")
    while True:
        inp = input(">> ").strip()
        if inp == "/exit":
            print("Сохраняем модель перед выходом...")
            model.save()
            print("Выход.")
            break
        elif inp == "/load":
            print("Загружаем модель из диска...")
            model.load()
        elif inp.startswith("/train "):
            text_to_train = inp[len("/train "):].strip()
            if text_to_train:
                model.online_train(text_to_train)
                print("Модель обучена на введённом тексте.")
            else:
                print("Текст для обучения пуст.")
        elif inp.startswith("/seed "):
            seed = inp[len("/seed "):].strip()
            generated = model.generate_sentence(seed=seed, max_len=30)
            print("Сгенерировано:", generated)
        elif inp == "/save":
            model.save()
        elif inp == "/help":
            print("Команды:")
            print("  /exit — выход (с автосохранением)")
            print("  /train <текст> — обучить модель на тексте онлайн")
            print("  /seed <начало предложения> — сгенерировать текст, начиная с этого")
            print("  /save — сохранить состояние модели")
            print("  /load — загрузить состояние модели из папки")
            print("  /help — показать эту подсказку")
        else:
            generated = model.generate_sentence(seed=inp, max_len=30)
            print("Сгенерировано:", generated)


if __name__ == "__main__":
    model = SemanticSyntaxGraph(max_context_len=10, vector_size=100)
    model.load()

    interactive_mode(model)
