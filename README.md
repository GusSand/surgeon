# Even Heads Fix Odd Errors: Mechanistic Discovery and Surgical Repair in Transformer Attention
# or 
# The Surgeon's Dilemma: A Mechanistic Case Study

## 📊 Interactive Visualization
[**View the interactive visualization here**](https://gussand.github.io/surgeon]/visualization/)

## 📄 Paper
[Read the full paper (PDF)](./paper/surgeons_dilemma.pdf)

## Full Reproducibility Scripts
Are in the code directory. 
See code/README.md for complete instructions. 

## Abstract
    We present a mechanistic case study of a format-dependent reasoning failure in
    Llama-3.1-8B-Instruct, where the model incorrectly judges "9.11" as larger than "9.8"
    in chat or Q\&A formats, but answers correctly in simple format.

    Through systematic intervention, we discover transformers implement even/odd
    attention head specialization: even indexed heads handle numerical comparison, 
    while odd heads serve incompatible functions. The bug requires exactly 8 even 
    heads at Layer 10 for perfect repair. Any combination of 8+ even heads succeeds, 
    while 7 or fewer completely fails, revealing sharp computational thresholds 
    with perfect redundancy among the 16 even heads. 

    SAE analysis reveals the mechanism: format representations separate 
    (10\% feature overlap at Layer 7), then re-entangle with different 
    weightings (80\% feature overlap at Layer 10), with specific features 
    showing 1.5× amplification in failing formats. We achieve perfect 
    repair using only 25\% of attention heads and identify a 60\% pattern 
    replacement threshold, demonstrating that apparent full-module requirements 
    hide sophisticated substructure with implications for interpretability and 
    efficiency. 

## Key Findings

### 🔬 Core Discovery
- **Even/odd attention head specialization**: Transformers systematically organize computation by head index parity - even heads (0,2,4,...,30) handle numerical comparison while odd heads (1,3,5,...,31) serve incompatible functions

### 🎯 Surgical Repair
- **Perfect bug fix achieved**: 100% success rate (1000/1000 trials) repairing the "9.11 > 9.8" error by transplanting attention patterns at Layer 10
- **Precise intervention requirements**: Works ONLY at Layer 10, not layers 9 or 11, demonstrating extreme specificity

### 📊 Sharp Computational Thresholds
- **Binary phase transition**: Exactly 8 even heads required - 7 heads = 0% success, 8+ heads = 100% success
- **60% pattern replacement threshold**: Another sharp transition with no gradual degradation
- **Perfect redundancy**: Any combination of 8 even heads works equally well (16 choose 8 = 12,870 combinations tested)

### 🧠 Mechanistic Understanding
- **Format-dependent failure mode**: Bug only occurs in Q&A/chat formats (100% error) but not simple format (0% error)
- **Re-entanglement bottleneck**: SAE analysis shows formats separate (10% overlap at Layer 7) then re-entangle (80% overlap at Layer 10)
- **Shared components, not separate circuits**: 40-60% feature overlap proves bugs arise from differential orchestration of same features

### 💡 Broader Implications
- **75% efficiency gain**: Only 8 of 32 heads needed for numerical tasks
- **"Goldilocks principle"**: Intervention granularity determines success - not individual heads (too narrow) or full layers (too coarse), but complete submodules
- **Hidden sophisticated architecture**: Transformers implement discrete computational modes, not continuous processing

