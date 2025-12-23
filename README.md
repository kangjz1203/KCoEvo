# KCoEvo: Knowledge Graph-Augmented Evolutionary Code Generation

This repository contains the implementation and supplementary materials for the paper  
**"KCoEvo: A Knowledge Graph-Augmented Framework for Evolutionary Code Generation" (DASFAA 2026).**

> ⚠️ The repository is being organized.  
> Full code and datasets will be released shortly after internal review and cleanup.

---

## 1. Overview

**KCoEvo** introduces a knowledge graph (KG)-based framework that augments large language models (LLMs) with structured API evolution knowledge.  
It enables *version-consistent* and *interpretable* code migration by connecting symbolic graph reasoning with neural code generation.

The framework operates in three key stages:
1. **Static Graph Construction:** Extract intra-version API relations from source libraries.
2. **Dynamic Graph Alignment:** Link APIs across versions using rule-based BFS traversal (e.g., `rename`, `deprecate`, `relocate`).
3. **Planning & Reasoning:** Guide LLMs to generate migration-aware code via graph-informed prompts.

---

## 2. Method Highlights

- **Rule-based Alignment:** Automatically matches API nodes across versions based on naming, usage, and documentation patterns.
- **Synthetic Supervision:** Extracts migration paths (`a₁ → rename → a₂`) directly from GitHub diffs without manual annotation.
- **LLM Planning Module:** Utilizes contextual subgraphs to plan migration trajectories before generation.

---

## 3. Experimental Summary

Evaluations are conducted on the **VersiCode** benchmark covering diverse migration types  
(Major→Major, Major→Minor, Minor→Major, Minor→Minor).  
Results show consistent improvements in:
- **CDC@1:** Compilation and functional correctness.
- **EM@1:** Structural and semantic alignment with reference code.

Example gains (KG-enhanced vs. base models):
- DeepSeek-V3: +37→+75 (CDC/EM)
- Llama-3-70B: +41→+66
- GPT-5: +17→+27

---

## 4. Repository Layout
```
KCoEvo/
├── kg_construction/ 
├── scripts/
|   
└── README.md
```

## 5. Data
Knowledge graphs are constructed for popular libraries (e.g., TensorFlow, PyTorch) and stored in here. 
Google Drive:
https://drive.google.com/file/d/1cmKuFmUhXz4xq2ZostgBoFcYgBDfvQXu/view?usp=sharing

## 6. Citation

<!-- If you find this work helpful, please cite:

```bibtex
@inproceedings{kang2026kcoevo,
  title={KCoEvo: A Knowledge Graph Augmented Framework for Evolutionary Code Generation},
  author={Kang et al.},
  booktitle={International Conference on Database Systems for Advanced Applications (DASFAA)},
  year={2026}
} -->