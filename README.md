# CSL7110 Assignment 2 — MinHash and LSH

## Overview
This repository implements **MinHashing** and **Locality Sensitive Hashing (LSH)** for approximate similarity search.

---

## Repository Structure

```
minhash_assignment/
├── data/                        
│   ├── D1.txt
│   ├── D2.txt
│   ├── D3.txt
│   ├── D4.txt
│   └── u.data                   
├── q1_kgrams.py               
├── q2_minhash.py               
├── q3_lsh.py                   
├── q4_movielens_minhash.py      
├── q5_movielens_lsh.py          
├── run_all.py                   
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python 3.8+
- NumPy

```bash
pip install -r requirements.txt
```

### Dataset Setup

**Questions 1–3:** The 4 document files (`D1.txt`–`D4.txt`) are included in the `data/` directory.

**Questions 4–5 (MovieLens):**
1. Download MovieLens 100k from: http://www.grouplens.org/node/73
2. Extract the archive
3. Copy `u.data` into the `data/` directory

---

## Running the Code


### Run individual questions
```bash
python q1_kgrams.py      # K-Gram construction and Jaccard similarity
python q2_minhash.py     # MinHash signature experiments
python q3_lsh.py         # LSH with banding
python q4_movielens_minhash.py   # MinHash on MovieLens 
python q5_movielens_lsh.py       # LSH on MovieLens 
```

