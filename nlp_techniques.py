"""
Natural Language Processing Techniques Cheatsheet
From basic text processing to advanced transformers and LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    BertModel, GPT2Model, T5ForConditionalGeneration,
    Trainer, TrainingArguments
)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import nltk
from collections import Counter

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def basic_text_preprocessing(text):
    """Basic text cleaning pipeline"""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_text_preprocessing(text, remove_stopwords=True, lemmatize=True):
    """Advanced preprocessing with NLTK"""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    text = basic_text_preprocessing(text)
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

# ============================================================================
# TOKENIZATION
# ============================================================================

# Byte-Pair Encoding (BPE) tokenizer usage
def tokenize_with_transformer(texts, model_name='bert-base-uncased', max_length=512):
    """Tokenize texts using transformer tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encodings

# Custom vocabulary builder
class VocabularyBuilder:
    def __init__(self, min_freq=1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.min_freq = min_freq
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = Counter()
        for text in texts:
            word_freq.update(text.split())
        
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.word2idx.get(word, 1) for word in text.split()]
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

# ============================================================================
# WORD EMBEDDINGS
# ============================================================================

# Word2Vec-style embeddings
class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super(Word2VecSkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, center_word):
        embedded = self.embeddings(center_word)
        output = self.output(embedded)
        return output

# GloVe-style embeddings
def load_glove_embeddings(glove_file, vocab, embedding_dim=300):
    """Load pre-trained GloVe embeddings"""
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
        else:
            embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))
    
    return embedding_matrix

# FastText-style character n-grams
class FastTextEmbedding(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, embed_dim=300):
        super(FastTextEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, embed_dim)
        
    def forward(self, word_ids, char_ids):
        word_embed = self.word_embeddings(word_ids)
        char_embed = self.char_embeddings(char_ids).mean(dim=1)
        return word_embed + char_embed

# ============================================================================
# RNN/LSTM/GRU MODELS
# ============================================================================

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 num_classes, dropout=0.5, bidirectional=True):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed)
        
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# Attention mechanism for RNN
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(AttentionRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output

# ============================================================================
# TRANSFORMER MODELS
# ============================================================================

# Basic Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_seq_length=512):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention

# ============================================================================
# USING PRETRAINED TRANSFORMERS
# ============================================================================

# BERT for classification
class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Multi-task BERT
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes_task1=2, num_classes_task2=3):
        super(MultiTaskBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.classifier1 = nn.Linear(hidden_size, num_classes_task1)
        self.classifier2 = nn.Linear(hidden_size, num_classes_task2)
        
    def forward(self, input_ids, attention_mask, task='task1'):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        
        if task == 'task1':
            return self.classifier1(pooled)
        else:
            return self.classifier2(pooled)

# ============================================================================
# NAMED ENTITY RECOGNITION (NER)
# ============================================================================

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=300, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF transitions
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix['<START>'], :] = -10000
        self.transitions.data[:, tag_to_ix['<STOP>']] = -10000
        
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions

# Transformer-based NER
def ner_with_transformers(text, model_name='dbmdz/bert-large-cased-finetuned-conll03-english'):
    """Perform NER using pretrained transformer"""
    from transformers import pipeline
    
    ner_pipeline = pipeline("ner", model=model_name, grouped_entities=True)
    entities = ner_pipeline(text)
    return entities

# ============================================================================
# TEXT CLASSIFICATION
# ============================================================================

# CNN for text classification
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, 
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        output = self.fc(cat)
        return output

# Hierarchical Attention Network
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.word_encoder = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(hidden_dim * 2, 1)
        self.sentence_encoder = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.sentence_attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, text, sent_lengths, word_lengths):
        # Word-level encoding
        embedded = self.embedding(text)
        word_out, _ = self.word_encoder(embedded)
        word_att = F.softmax(self.word_attention(word_out), dim=1)
        sent_rep = torch.sum(word_att * word_out, dim=1)
        
        # Sentence-level encoding
        sent_out, _ = self.sentence_encoder(sent_rep.unsqueeze(1))
        sent_att = F.softmax(self.sentence_attention(sent_out), dim=1)
        doc_rep = torch.sum(sent_att * sent_out, dim=1)
        
        output = self.fc(doc_rep.squeeze(1))
        return output

# ============================================================================
# SEQUENCE-TO-SEQUENCE MODELS
# ============================================================================

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Seq2SeqDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

# Attention-based Seq2Seq
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys):
        # query: (batch, hidden_dim)
        # keys: (batch, seq_len, hidden_dim)
        query = query.unsqueeze(1)  # (batch, 1, hidden_dim)
        score = self.V(torch.tanh(self.W1(query) + self.W2(keys)))
        attention_weights = F.softmax(score, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        return context, attention_weights

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

def sentiment_analysis_pipeline(text, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
    """Quick sentiment analysis using transformers"""
    from transformers import pipeline
    
    classifier = pipeline('sentiment-analysis', model=model_name)
    result = classifier(text)
    return result

# Aspect-Based Sentiment Analysis
class AspectBasedSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_aspects, num_sentiments):
        super(AspectBasedSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.aspect_fc = nn.Linear(hidden_dim * 2, num_aspects)
        self.sentiment_fc = nn.Linear(hidden_dim * 2, num_sentiments)
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        
        aspect_logits = self.aspect_fc(pooled)
        sentiment_logits = self.sentiment_fc(pooled)
        
        return aspect_logits, sentiment_logits

# ============================================================================
# QUESTION ANSWERING
# ============================================================================

def qa_with_transformers(question, context, model_name='distilbert-base-cased-distilled-squad'):
    """Question answering using transformers"""
    from transformers import pipeline
    
    qa_pipeline = pipeline('question-answering', model=model_name)
    result = qa_pipeline(question=question, context=context)
    return result

# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(prompt, model_name='gpt2', max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """Generate text using pretrained language model"""
    from transformers import pipeline
    
    generator = pipeline('text-generation', model=model_name)
    generated = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1
    )
    return generated[0]['generated_text']

# Beam search decoding
def beam_search_decode(model, input_ids, beam_width=5, max_length=50):
    """Beam search for sequence generation"""
    sequences = [(input_ids, 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in sequences:
            if seq[0, -1].item() == 2:  # EOS token
                all_candidates.append((seq, score))
                continue
            
            with torch.no_grad():
                output = model(seq)
                logits = output[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
            
            top_log_probs, top_indices = log_probs.topk(beam_width)
            
            for i in range(beam_width):
                candidate_seq = torch.cat([seq, top_indices[:, i].unsqueeze(0).unsqueeze(0)], dim=1)
                candidate_score = score + top_log_probs[0, i].item()
                all_candidates.append((candidate_seq, candidate_score))
        
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return sequences[0][0]

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_text_model(model, train_loader, val_loader, criterion, optimizer, 
                     num_epochs=10, device='cuda', clip_grad=1.0):
    """Training loop for NLP models"""
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_nlp_model.pth')
    
    return model

# Fine-tuning with Hugging Face Trainer
def finetune_transformer(model_name, train_dataset, eval_dataset, num_labels, output_dir='./results'):
    """Fine-tune transformer model with Trainer API"""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    return model

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_bleu_score(reference, hypothesis):
    """Calculate BLEU score for machine translation"""
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([reference.split()], hypothesis.split())

def calculate_rouge_scores(reference, hypothesis):
    """Calculate ROUGE scores for summarization"""
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    return scores

def calculate_perplexity(model, dataloader, device='cuda'):
    """Calculate perplexity for language models"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# ============================================================================
# TEXT AUGMENTATION
# ============================================================================

def synonym_replacement(text, n=3):
    """Replace n words with synonyms"""
    from nltk.corpus import wordnet
    import random
    
    words = text.split()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(set(synonyms)))
            words = [synonym if word == random_word else word for word in words]
            num_replaced += 1
        
        if num_replaced >= n:
            break
    
    return ' '.join(words)

def back_translation(text, src_lang='en', tgt_lang='fr'):
    """Back translation for data augmentation"""
    from transformers import MarianMTModel, MarianTokenizer
    
    # Translate to target language
    model_name_forward = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer_forward = MarianTokenizer.from_pretrained(model_name_forward)
    model_forward = MarianMTModel.from_pretrained(model_name_forward)
    
    translated = model_forward.generate(**tokenizer_forward(text, return_tensors="pt", padding=True))
    tgt_text = tokenizer_forward.decode(translated[0], skip_special_tokens=True)
    
    # Translate back
    model_name_back = f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}'
    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
    model_back = MarianMTModel.from_pretrained(model_name_back)
    
    back_translated = model_back.generate(**tokenizer_back(tgt_text, return_tensors="pt", padding=True))
    final_text = tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
    
    return final_text
