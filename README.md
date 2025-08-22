# The Surgeon's Dilemma: A Mechanistic Case Study

## 📊 Interactive Visualization
[**View the interactive visualization here**](https://gussand.github.io/surgeon]/visualization/)

## 📄 Paper
[Read the full paper (PDF)](./paper/surgeons_dilemma.pdf)

## 🔬 Key Findings
- The decimal comparison bug (9.8 vs 9.11) is triggered by chat template formatting
- Layer 25 is the critical divergence point where paths split
- The bug is irremediably entangled - neurons causing errors are essential for decimal processing
- Intervention attempts fail: no "sweet spot" exists between wrong answers and incoherence

## 💻 Code
- `code/logitlens.py` - Logit lens analysis revealing Layer 25 divergence
- `code/intervention_pytorch_hooks.py` - Intervention experiments