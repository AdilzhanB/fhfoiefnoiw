"""
LLM Generation Techniques Cheatsheet
Prompt engineering, fine-tuning, generation strategies, RAG, and LLM applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration,
    LlamaForCausalLM, pipeline, Trainer, TrainingArguments,
    BitsAndBytesConfig, GenerationConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

# ============================================================================
# TEXT GENERATION STRATEGIES
# ============================================================================

def greedy_decoding(model, tokenizer, prompt, max_length=100, device='cuda'):
    """Greedy decoding - always select highest probability token"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def beam_search_decoding(model, tokenizer, prompt, num_beams=5, max_length=100, device='cuda'):
    """Beam search decoding for better quality"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def top_k_sampling(model, tokenizer, prompt, top_k=50, temperature=0.7, max_length=100, device='cuda'):
    """Top-K sampling - sample from top K probable tokens"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def top_p_nucleus_sampling(model, tokenizer, prompt, top_p=0.9, temperature=0.7, max_length=100, device='cuda'):
    """Top-P (nucleus) sampling - sample from smallest set with cumulative prob >= p"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def contrastive_search(model, tokenizer, prompt, penalty_alpha=0.6, top_k=4, max_length=100, device='cuda'):
    """Contrastive search for coherent and diverse generation"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_with_custom_config(model, tokenizer, prompt, device='cuda'):
    """Generate with custom generation config"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generation_config = GenerationConfig(
        max_length=200,
        min_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        num_return_sequences=3,
        early_stopping=True
    )
    
    with torch.no_grad():
        outputs = model.generate(input_ids, generation_config=generation_config)
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ============================================================================
# PROMPT ENGINEERING TEMPLATES
# ============================================================================

class PromptTemplate:
    """Template-based prompt engineering"""
    
    @staticmethod
    def zero_shot(task, input_text):
        """Zero-shot prompting"""
        return f"Task: {task}\n\nInput: {input_text}\n\nOutput:"
    
    @staticmethod
    def few_shot(task, examples, input_text):
        """Few-shot prompting with examples"""
        prompt = f"Task: {task}\n\n"
        
        for i, (ex_input, ex_output) in enumerate(examples, 1):
            prompt += f"Example {i}:\nInput: {ex_input}\nOutput: {ex_output}\n\n"
        
        prompt += f"Now solve this:\nInput: {input_text}\nOutput:"
        return prompt
    
    @staticmethod
    def chain_of_thought(task, input_text):
        """Chain-of-thought prompting"""
        return f"{task}\n\n{input_text}\n\nLet's solve this step by step:"
    
    @staticmethod
    def few_shot_cot(task, examples_with_reasoning, input_text):
        """Few-shot chain-of-thought"""
        prompt = f"Task: {task}\n\n"
        
        for i, (ex_input, reasoning, ex_output) in enumerate(examples_with_reasoning, 1):
            prompt += f"Example {i}:\nInput: {ex_input}\n"
            prompt += f"Reasoning: {reasoning}\nOutput: {ex_output}\n\n"
        
        prompt += f"Now solve this:\nInput: {input_text}\nReasoning:"
        return prompt
    
    @staticmethod
    def instruction_following(instruction, input_text):
        """Instruction-following format"""
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    
    @staticmethod
    def role_prompting(role, task, input_text):
        """Role-based prompting"""
        return f"You are {role}.\n\nTask: {task}\n\nInput: {input_text}\n\nResponse:"
    
    @staticmethod
    def self_consistency(task, input_text, num_samples=5):
        """Self-consistency prompting (generate multiple times and vote)"""
        return [f"{task}\n\n{input_text}\n\nLet's approach this step by step (attempt {i+1}):" 
                for i in range(num_samples)]

# Example usage of prompt templates
def classify_sentiment_with_prompts(text, model, tokenizer):
    """Sentiment classification using different prompting strategies"""
    
    # Zero-shot
    zero_shot_prompt = PromptTemplate.zero_shot(
        "Classify the sentiment as positive, negative, or neutral",
        text
    )
    
    # Few-shot
    examples = [
        ("I love this product!", "positive"),
        ("This is terrible.", "negative"),
        ("It's okay, nothing special.", "neutral")
    ]
    few_shot_prompt = PromptTemplate.few_shot(
        "Classify the sentiment",
        examples,
        text
    )
    
    # Generate with both approaches
    zero_shot_result = greedy_decoding(model, tokenizer, zero_shot_prompt)
    few_shot_result = greedy_decoding(model, tokenizer, few_shot_prompt)
    
    return {
        'zero_shot': zero_shot_result,
        'few_shot': few_shot_result
    }

# ============================================================================
# FINE-TUNING TECHNIQUES
# ============================================================================

def full_fine_tuning(model_name, train_dataset, eval_dataset, output_dir='./finetuned_model'):
    """Full fine-tuning of language model"""
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        learning_rate=2e-5,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# LoRA (Low-Rank Adaptation) Fine-tuning
def lora_fine_tuning(model_name, train_dataset, eval_dataset, output_dir='./lora_model'):
    """Fine-tune using LoRA for parameter efficiency"""
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Which layers to apply LoRA
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy='steps',
        eval_steps=100,
        learning_rate=2e-4,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    
    return model, tokenizer

# QLoRA (Quantized LoRA) for even more efficiency
def qlora_fine_tuning(model_name, train_dataset, eval_dataset, output_dir='./qlora_model'):
    """Fine-tune using QLoRA with 4-bit quantization"""
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    
    return model, tokenizer

# Prefix Tuning
class PrefixTuning(nn.Module):
    """Prefix tuning for parameter-efficient fine-tuning"""
    def __init__(self, num_prefix_tokens, hidden_size, num_layers):
        super(PrefixTuning, self).__init__()
        self.num_prefix_tokens = num_prefix_tokens
        
        # Learnable prefix parameters
        self.prefix_params = nn.Parameter(
            torch.randn(num_layers, 2, num_prefix_tokens, hidden_size)
        )
    
    def forward(self):
        return self.prefix_params

# ============================================================================
# RETRIEVAL-AUGMENTED GENERATION (RAG) - COMPACT VERSION
# ============================================================================

class SimpleRAG:
    """Compact RAG with FAISS + SentenceTransformers"""
    
    def __init__(self, model_name='gpt2', embedding_model='all-MiniLM-L6-v2'):
        import faiss
        from sentence_transformers import SentenceTransformer
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # Cosine similarity
        self.documents = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
    
    def add_documents(self, docs):
        """Add documents to knowledge base"""
        embeddings = self.embedding_model.encode(docs, normalize_embeddings=True)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(docs)
    
    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents"""
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_emb.astype('float32'), top_k)
        return [self.documents[i] for i in indices[0]]
    
    def generate(self, question, max_length=200):
        """Generate answer with context"""
        context = "\n".join(self.retrieve(question))
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, temperature=0.7)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================================
# vLLM INFERENCE - INDEPENDENT USAGE
# ============================================================================

def vllm_basic_inference(model_name='gpt2', prompts=None, max_tokens=100):
    """
    Basic vLLM inference (10-20x faster than HuggingFace)
    
    Example:
        prompts = ["Write a story about", "Explain quantum computing"]
        outputs = vllm_basic_inference('gpt2', prompts, max_tokens=150)
    """
    from vllm import LLM, SamplingParams
    
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
    
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def vllm_batch_inference(model_name='gpt2', prompts=None, temperature=0.7, top_p=0.9):
    """
    Batch inference with vLLM (efficient for multiple prompts)
    
    Example:
        results = vllm_batch_inference('gpt2-medium', 
                                       prompts=['Prompt 1', 'Prompt 2'],
                                       temperature=0.8)
    """
    from vllm import LLM, SamplingParams
    
    llm = LLM(model=model_name, tensor_parallel_size=1)
    params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=256)
    
    outputs = llm.generate(prompts, params)
    results = []
    for output in outputs:
        results.append({
            'prompt': output.prompt,
            'generated_text': output.outputs[0].text,
            'tokens': len(output.outputs[0].token_ids)
        })
    
    return results

# ============================================================================
# INSTRUCTION TUNING
# ============================================================================

def format_instruction_data(instruction, input_text, output_text):
    """Format data for instruction tuning (Alpaca-style)"""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output_text}"""

def create_instruction_dataset(examples):
    """Create instruction tuning dataset"""
    formatted_examples = []
    
    for example in examples:
        formatted = format_instruction_data(
            example['instruction'],
            example.get('input', ''),
            example['output']
        )
        formatted_examples.append(formatted)
    
    return formatted_examples

# ============================================================================
# REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF)
# ============================================================================

class RewardModel(nn.Module):
    """Reward model for RLHF"""
    def __init__(self, model_name='gpt2'):
        super(RewardModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        
        # Use last token's hidden state
        last_token_hidden = last_hidden[:, -1, :]
        reward = self.reward_head(last_token_hidden)
        
        return reward

def train_reward_model(model, preference_dataset, optimizer, device='cuda'):
    """Train reward model on preference data"""
    model.train()
    
    for batch in preference_dataset:
        chosen_ids = batch['chosen_input_ids'].to(device)
        rejected_ids = batch['rejected_input_ids'].to(device)
        chosen_mask = batch['chosen_attention_mask'].to(device)
        rejected_mask = batch['rejected_attention_mask'].to(device)
        
        # Get rewards
        chosen_reward = model(chosen_ids, chosen_mask)
        rejected_reward = model(rejected_ids, rejected_mask)
        
        # Loss: maximize difference
        loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()

# ============================================================================
# CONSTRAINED GENERATION
# ============================================================================

def constrained_beam_search(model, tokenizer, prompt, constraints, beam_width=5):
    """Beam search with hard constraints"""
    # This is a simplified version
    # For production, use Hugging Face's constrained generation
    
    from transformers import DisjunctiveConstraint
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Define constraints (e.g., must include certain words)
    force_words_ids = [
        tokenizer.encode(word, add_special_tokens=False) 
        for word in constraints
    ]
    
    output = model.generate(
        input_ids,
        max_length=100,
        num_beams=beam_width,
        force_words_ids=force_words_ids,
        num_return_sequences=1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ============================================================================
# PROMPT OPTIMIZATION
# ============================================================================

class AutomaticPromptEngineer:
    """Automatic prompt engineering/optimization"""
    
    def __init__(self, model, tokenizer, task_description):
        self.model = model
        self.tokenizer = tokenizer
        self.task_description = task_description
    
    def generate_prompt_candidates(self, num_candidates=10):
        """Generate prompt candidates"""
        meta_prompt = f"""Generate {num_candidates} different prompts for the following task:
        {self.task_description}
        
        Provide diverse prompts that could work well for this task."""
        
        input_ids = self.tokenizer.encode(meta_prompt, return_tensors='pt')
        
        outputs = self.model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=num_candidates,
            temperature=1.0,
            top_p=0.95
        )
        
        prompts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        return prompts
    
    def evaluate_prompt(self, prompt, test_cases):
        """Evaluate prompt on test cases"""
        correct = 0
        
        for input_text, expected_output in test_cases:
            full_prompt = f"{prompt}\n\nInput: {input_text}\nOutput:"
            
            input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt')
            output = self.model.generate(input_ids, max_length=100)
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            if expected_output.lower() in generated.lower():
                correct += 1
        
        return correct / len(test_cases)
    
    def optimize_prompt(self, test_cases, iterations=3):
        """Optimize prompt through iterative refinement"""
        best_prompt = None
        best_score = 0
        
        for _ in range(iterations):
            candidates = self.generate_prompt_candidates()
            
            for prompt in candidates:
                score = self.evaluate_prompt(prompt, test_cases)
                
                if score > best_score:
                    best_score = score
                    best_prompt = prompt
        
        return best_prompt, best_score

# ============================================================================
# TEXT SUMMARIZATION
# ============================================================================

def extractive_summarization(text, num_sentences=3):
    """Simple extractive summarization"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= num_sentences:
        return text
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Compute sentence importance (similarity to document)
    doc_vector = tfidf_matrix.sum(axis=0)
    similarities = cosine_similarity(tfidf_matrix, doc_vector)
    
    # Select top sentences
    top_indices = similarities.flatten().argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)
    
    summary = '. '.join([sentences[i] for i in top_indices]) + '.'
    return summary

def abstractive_summarization(text, model_name='facebook/bart-large-cnn'):
    """Abstractive summarization with transformers"""
    summarizer = pipeline('summarization', model=model_name)
    
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# ============================================================================
# DIALOGUE SYSTEMS
# ============================================================================

class DialogueAgent:
    """Simple dialogue agent with context"""
    
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.conversation_history = []
    
    def respond(self, user_input, max_length=1000):
        """Generate response maintaining conversation context"""
        # Add user input to history
        self.conversation_history.append(user_input)
        
        # Encode conversation history
        bot_input = self.tokenizer.encode(
            ' '.join(self.conversation_history) + self.tokenizer.eos_token,
            return_tensors='pt'
        )
        
        # Generate response
        chat_output = self.model.generate(
            bot_input,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(
            chat_output[:, bot_input.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        # Add response to history
        self.conversation_history.append(response)
        
        return response
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []

# ============================================================================
# EVALUATION METRICS FOR GENERATION
# ============================================================================

def calculate_perplexity(model, tokenizer, text, device='cuda'):
    """Calculate perplexity of generated text"""
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score"""
    from nltk.translate.bleu_score import sentence_bleu
    
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    
    score = sentence_bleu(reference_tokens, hypothesis_tokens)
    return score

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores"""
    from rouge import Rouge
    
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

def calculate_bertscore(references, hypotheses, model_type='bert-base-uncased'):
    """Calculate BERTScore for semantic similarity"""
    from bert_score import score
    
    P, R, F1 = score(hypotheses, references, model_type=model_type, verbose=False)
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

# ============================================================================
# INFERENCE OPTIMIZATION
# ============================================================================

def optimize_inference_with_kv_cache(model, tokenizer, prompt, max_new_tokens=50):
    """Efficient generation using KV-cache"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate with past_key_values caching
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # Enable KV-cache
            do_sample=False
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def batch_inference(model, tokenizer, prompts, batch_size=8):
    """Efficient batch inference"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        batch_results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        results.extend(batch_results)
    
    return results
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Google Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Reranker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI


DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "text-embedding-004"

# -------------------------
# 1) Load corpus
# -------------------------
loader = TextLoader(str(DATA_DIR / "corpus.txt"), encoding="utf-8")
docs = loader.load()

# -------------------------
# 2) Chunking
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)

# -------------------------
# 3) Google Embeddings
# -------------------------
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# -------------------------
# 4) Create VectorStore
# -------------------------
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
vectordb.persist()


# ============================================================================
#                 ★★★★★  RERANKER (bge-reranker-large)  ★★★★★
# ============================================================================

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query, docs, top_k=4):
        pairs = [[query, doc.page_content] for doc in docs]
        tokens = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**tokens).logits.squeeze()
        scores = scores.numpy()

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, s in ranked[:top_k]]


reranker = CrossEncoderReranker()


# ============================================================================
#                           RAG PIPELINE
# ============================================================================

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def rag_answer(query):
    # 1) векторный поиск
    retrieved = vectordb.similarity_search(query, k=15)

    # 2) реранкинг
    best_docs = reranker.rerank(query, retrieved, top_k=4)

    # 3) сбор контекста
    context = "\n\n".join([d.page_content for d in best_docs])

    # 4) финальный ответ LLM
    prompt = f"""
Ты — эксперт. Ответь на основе контекста ниже.

Контекст:
{context}

Вопрос:
{query}

Ответ:
"""
    return llm.predict(prompt)


# ----------------------------------------
# Пример запроса
# ----------------------------------------
print(rag_answer("Объясни основные причины кризиса Римской Империи"))