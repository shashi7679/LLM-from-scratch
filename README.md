# Training LLM - From Scratch to GPT-2

A comprehensive educational repository implementing Large Language Models (LLMs) from scratch, covering tokenization, transformer architecture, and training of a GPT-2 style model.

## 📋 Overview

This project is a deep dive into building language models from first principles. It covers:
- **Byte Pair Encoding (BPE)**: Tokenization strategy used by modern LLMs
- **Multi-head Attention**: Core mechanism of transformer models
- **Transformer Architecture**: Building blocks of modern LLMs
- **GPT-2 Implementation**: Complete implementation of a GPT-2 style model
- **Model Training**: Training pipeline with custom datasets and optimization

The project uses **PyTorch** for deep learning and **tiktoken** for tokenization, providing hands-on experience with the technologies that power production language models.

---

## 🗂️ Project Structure

```
Training_LLM/
├── 1_Byte_Pair_Encoding.ipynb       # Tokenization exploration and BPE implementation
├── 2_Multi-head_Attention.ipynb      # Understanding multi-head attention mechanisms
├── 3_LLM_Scratch.ipynb                # Building LLM blocks and architecture visualization
├── 4_GPT_2.ipynb                      # Detailed GPT-2 model architecture
├── 5_train.ipynb                      # Model training and evaluation
├── model.py                           # Core model implementations
├── archive/
│   └── the-verdict.txt                # Training dataset (text from a literary work)
├── LLM/                               # Python virtual environment
└── README.md                          # This file
```

---

## 📓 Notebooks Description

### 1. **1_Byte_Pair_Encoding.ipynb**
Explores tokenization through the Byte Pair Encoding (BPE) algorithm:
- Raw byte-level encoding
- Iterative merging of frequent byte pairs
- Building a complete BPE tokenizer
- Encoding and decoding text
- **Output**: Understanding of how modern LLMs tokenize text

### 2. **2_Multi-head_Attention.ipynb**
Deep dive into the attention mechanism:
- Single-head attention computation
- Query, Key, Value matrices
- Multi-head attention implementation
- Scaled dot-product attention
- Visualization of attention patterns
- **Output**: Mastery of the core transformer component

### 3. **3_LLM_Scratch.ipynb**
Building complete transformer blocks:
- Embedding layers (token + positional)
- Feedforward networks
- Layer normalization
- Transformer blocks assembly
- Architecture visualization
- **Output**: Complete understanding of transformer structure

### 4. **4_GPT_2.ipynb**
GPT-2 specific implementation details:
- Model configuration (124M parameters)
- Architecture components
- Forward pass implementation
- Model initialization strategies
- **Output**: Ready-to-train GPT-2 model

### 5. **5_train.ipynb** ⭐ Main Training Script
Comprehensive training pipeline:
- Data loading from "the-verdict.txt"
- GPT Dataset creation with windowing
- Train/validation split (90/10)
- Training loop with loss tracking
- Model evaluation
- **Output**: Trained language model checkpoint

---

## 🏗️ Model Architecture

### Model Configuration (GPT_CONFIG_124M)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 50,257 | GPT-2 vocabulary size |
| `context_length` | 256 | Maximum sequence length |
| `emb_dim` | 768 | Embedding dimension |
| `n_heads` | 12 | Number of attention heads |
| `n_layers` | 12 | Number of transformer layers |
| `drop_rate` | 0.1 | Dropout probability |
| `qkv_bias` | False | Bias in Q, K, V projections |

### Core Components (model.py)

#### **LayerNorm**
- Implements layer normalization with learnable scale and shift
- Stabilizes training by normalizing activations

#### **GELU**
- Gaussian Error Linear Unit activation function
- Smooth approximation used in transformers
- Formula: $0.5 \cdot x \cdot (1 + \tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)))$

#### **FeedForward**
- Two-layer feedforward network with GELU activation
- Expands to 4x embedding dimension, then contracts back
- Applied after attention in each transformer block

#### **MultiHeadAttention**
- Scaled dot-product attention across multiple heads
- Allows model to attend to different representation subspaces
- Causal masking prevents attending to future tokens
- Components:
  - Query (Q), Key (K), Value (V) projections
  - Attention score computation: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
  - Dropout regularization

#### **TransformerBlock**
- Combines attention and feedforward with residual connections
- Layer normalization applied before each sub-layer (pre-norm)
- Residual connections enable deep networks
- Architecture:
  ```
  x → LayerNorm → MultiHeadAttention → Dropout → Add → x'
  x' → LayerNorm → FeedForward → Dropout → Add → output
  ```

#### **GPTModel**
- Complete language model combining all components
- Token embedding: converts token IDs to vectors
- Positional embedding: adds position information
- Stack of transformer blocks: 12 layers of processing
- Final layer norm: normalizes before output projection
- Output head: projects to vocabulary logits

---

## 📦 Dependencies

The project requires:
- **PyTorch**: Deep learning framework
- **tiktoken**: OpenAI's tokenizer encoding library
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization (for attention patterns)
- **Jupyter**: Interactive notebooks

### Installation

```bash
# Create virtual environment
python -m venv LLM
source LLM/Scripts/activate  # On Windows: LLM\Scripts\activate

# Install dependencies
pip install torch tiktoken numpy matplotlib jupyter ipykernel
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
# Activate virtual environment
source LLM/Scripts/activate  # Windows: LLM\Scripts\activate

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
```

### 2. Navigate Through Notebooks in Order
```
1. 1_Byte_Pair_Encoding.ipynb      → Understand tokenization
2. 2_Multi-head_Attention.ipynb    → Learn attention mechanism
3. 3_LLM_Scratch.ipynb             → Build transformer blocks
4. 4_GPT_2.ipynb                   → Apply to GPT-2 architecture
5. 5_train.ipynb                   → Train the model
```

### 3. Execute Notebooks
```bash
# Start Jupyter Lab or Notebook
jupyter lab
# or
jupyter notebook

# Open each notebook sequentially and run all cells
```

### 4. Monitor Training
- Watch loss convergence
- Compare train/validation metrics
- Analyze attention patterns (if included)

---

## 📊 Training Details

### Dataset
- **Source**: `archive/the-verdict.txt` (Classic literature)
- **Total Characters**: ~170K characters
- **Tokenization**: GPT-2 tokenizer (50K vocab)
- **Split**: 90% training, 10% validation

### Data Processing
The `GPTDataset` class:
1. Tokenizes text with GPT-2 tokenizer
2. Creates sliding windows of length 256 tokens
3. Generates input-target pairs (prediction task)
4. DataLoader batches samples for training

### Training Configuration
- **Batch Size**: 4 (configurable)
- **Max Sequence Length**: 256 tokens
- **Stride**: 128 tokens (50% overlap between windows)
- **Device**: GPU (if available) or CPU

### Loss Function
- **Cross-entropy loss**: Standard for language modeling
- Minimizes difference between predicted and actual next tokens
- Computed over all positions in the sequence

---

## 📈 Model Capabilities

After training, the model can:
- ✅ Generate text based on a prompt
- ✅ Predict next token probabilities
- ✅ Learn patterns from the training corpus
- ✅ Demonstrate understanding of language structure

### Example Usage (After Training)
```python
# Set model to evaluation mode
model.eval()

# Generate text from a prompt
prompt = "The verdict was"
encoded = tokenizer.encode(prompt)
x = torch.tensor(encoded).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])
```

---

## 🔑 Key Learnings

This project teaches:

1. **Tokenization**: How text is converted to numerical form
   - Byte-level encoding fundamentals
   - BPE algorithm and vocabulary building
   - Token vs. character efficiency

2. **Transformer Architecture**: Core of modern NLP
   - Self-attention mechanism
   - Multi-head attention benefits
   - Residual connections and layer normalization
   - Positional encoding strategies

3. **Deep Learning Practices**:
   - Custom PyTorch datasets and dataloaders
   - Training loops and loss tracking
   - Train/validation splits
   - Model evaluation and checkpointing

4. **Language Modeling**:
   - Next-token prediction task
   - Causal attention masking
   - Context window management
   - Scaling laws (number of parameters vs performance)

---

## 🔬 Advanced Topics Covered

### Attention Patterns
- How multi-head attention learns to focus on relevant tokens
- Positional bias in attention
- Head specialization across layers

### Optimization
- Cross-entropy loss minimization
- Gradient flow through deep networks
- Regularization via dropout

### Scaling
- GPT-2 model size (124M parameters)
- Trade-offs between model capacity and training time
- Computational complexity of self-attention: $O(n^2)$

---

## 📝 File Descriptions

### `model.py` - Core Implementations

```python
# Custom layer implementations
LayerNorm(emb_dim)
GELU()
FeedForward(cfg)
MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
TransformerBlock(cfg)
GPTModel(config)
```

Each class inherits from `nn.Module` for PyTorch integration.

### `archive/the-verdict.txt`
- Training corpus (public domain text)
- ~2,500-3,500 tokens depending on tokenizer
- Sufficient for demonstrating model training

---

## 🎯 Network Capacity & Performance

### Model Parameters
- **Total**: ~124 Million parameters
- **Embeddings**: 2 × 768 × 50,257 (token + position)
- **Transformer Blocks**: 12 layers × (attention + feedforward)
- **Output Layer**: 768 × 50,257

### Training Considerations
- **Memory**: ~500MB-1GB depending on batch size and sequence length
- **Training Time**: Minutes to hours (on GPU) depending on hardware
- **Hardware**: Works on CPU; GPU accelerates significantly
- **Recommended**: NVIDIA GPU with CUDA support for faster training

---

## 🧪 Experimentation Ideas

1. **Vary Model Size**: Reduce `emb_dim` and `n_layers` for faster training
2. **Change Vocabulary**: Use different tokenizers (BPE variations, SentencePiece)
3. **Different Datasets**: Replace the literary text with domain-specific data
4. **Attention Visualization**: Plot attention weights to understand model behavior
5. **Text Generation**: Implement beam search or temperature-based sampling
6. **Fine-tuning**: Adapt pretrained models for specific tasks
7. **Position Embeddings**: Experiment with RoPE or ALiBi alternatives

---

## 📚 References & Further Reading

### Papers
- **Attention Is All You Need** (Vaswani et al., 2017)
  - Original Transformer architecture
  - https://arxiv.org/abs/1706.03762

- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
  - GPT-2 paper with training details
  - https://openai.com/research/gpt-2

### Resources
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **tiktoken GitHub**: https://github.com/openai/tiktoken
- **3Blue1Brown - Attention Visualization**: https://www.3blue1brown.com/

### Similar Projects
- Andrej Karpathy's "nanoGPT": Minimal GPT implementation
- EleutherAI's tools and documentation
- Hugging Face model hub for pretrained models

---

## 🚧 Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
- **Solution**: Reduce batch size or sequence length in training config

**Issue**: `tiktoken not found`
- **Solution**: `pip install tiktoken`

**Issue**: Slow training
- **Solution**: Verify GPU is being used (`torch.cuda.is_available()`)

**Issue**: Data loading errors
- **Solution**: Verify path to `the-verdict.txt` matches notebook configuration

---

## 📄 License & Attribution

This project is an educational implementation based on:
- OpenAI's GPT-2 architecture
- Transformer research community
- Educational frameworks and tutorials

Free to use for learning and research purposes.

---

## 🤝 Contributing & Improvements

Potential enhancements:
- [ ] Add text generation functions
- [ ] Implement attention visualization tools
- [ ] Add model checkpointing and resumption
- [ ] Include inference with temperature sampling
- [ ] Create training metrics dashboard
- [ ] Add multi-GPU support
- [ ] Implement gradient accumulation for large batches

---

## ⚠️ Important Notes

- **Training Data**: Current dataset is small; results may be limited
- **Convergence**: Observable improvement typically after multiple epochs
- **Model Performance**: Not production-ready; for learning GPT-2 internals
- **Computational Cost**: Training even small models requires significant compute
- **Reproducibility**: Set random seeds for consistent results across runs

---

## 📞 Quick Reference

| Task | Location |
|------|----------|
| Understand Tokenization | `1_Byte_Pair_Encoding.ipynb` |
| Learn Attention | `2_Multi-head_Attention.ipynb` |
| Build Transformer | `3_LLM_Scratch.ipynb` |
| Study GPT-2 | `4_GPT_2.ipynb` |
| Train Model | `5_train.ipynb` |
| Model Code | `model.py` |

---

**Last Updated**: March 2025  
**Status**: Active Development & Learning  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+
