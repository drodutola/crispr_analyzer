# CRISPR Analyzer 🧬

A Python-based tool for analyzing RNA sequences relevant to CRISPR-Cas9 gene editing workflows — supporting sgRNA design, sequence validation, and off-target impact assessment.

## Overview

CRISPR-Cas9 genome editing requires precise RNA sequence analysis to ensure guide RNAs (sgRNAs) target the correct genomic loci with minimal off-target effects. This tool automates key steps in that analysis pipeline, reducing manual effort and improving accuracy in experimental design.

## Features

- RNA sequence input and parsing
- sgRNA candidate identification and scoring
- Off-target site prediction and flagging
- Sequence validation and GC content analysis
- Output reports for downstream experimental use

## Tech Stack

- **Language:** Python
- **Libraries:** Biopython, Pandas, NumPy
- **Input formats:** FASTA, plain text RNA sequences

## How to Run

```bash
git clone https://github.com/drodutola/crispr_analyzer
cd crispr_analyzer
pip install -r requirements.txt
python analyzer.py --input sequence.fasta
```

## Background

Developed alongside hands-on CRISPR-Cas9 laboratory work, including sgRNA design, gel electrophoresis validation, and off-target impact analysis in mammalian systems. Certified in Molecular Gene Editing (CRISPR) — Harvard University, 2021.

## Author

**Dr. Peter Odutola, M.D.** — Physician, AI developer, and genomics researcher.  
[GitHub Profile](https://github.com/drodutola)
