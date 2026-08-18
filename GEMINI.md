# Multi-View HDBSCAN (MV-HDBSCAN)

## 📌 Project Overview
The objective of this project is to develop and implement a **Multi-View version of HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise). 

Multi-view clustering leverages complementary information across multiple feature representations (views) of the same underlying data objects to produce more robust, high-quality clusterings than any single view can achieve alone.

---

## 📂 Project Structure

- **`papers/`**: Research papers providing foundational theory and multi-view clustering methods:
  - `papers/campello2013_hdbscan.pdf` – *Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN)*
  - `papers/daume2011_cotraining.pdf` – *Co-Training for Multi-View Clustering*
  - `papers/kumar2011_coreg_mvc.pdf` – *Co-Regularized Multi-View Spectral Clustering*
  - `papers/yang_wang2018_mvc_survey.pdf` – *Multi-View Clustering: A Survey*
- **`specs/`**: Specifications, technical notes, paper summaries, and algorithmic designs.
- **`src/`**: Source code implementation of the MV-HDBSCAN algorithm and related utilities.
- **`tests/`**: Unit tests, synthetic multi-view dataset generators, and validation benchmarks.

---

## 📐 Project Conventions & Guidelines

- **File Path References**: All references to project files, specifications, documentation, and source code must use **relative paths** (e.g., `papers/campello2013_hdbscan.pdf`, `specs/algorithm_design.md`, `src/mv_hdbscan.py`). Avoid hardcoding absolute system paths.

---

## 🗺️ Roadmap & Workflow

1. **Literature Summaries & Analysis**:
   - Extract and summarize core concepts from papers in `papers/` into detailed design specifications in `specs/`.
   - Analyze integration points for multi-view synergy (e.g., core distance fusion, mutual reachability graphs combination, co-training/co-regularization across minimum spanning trees or cluster hierarchies).

2. **Algorithm Design & Specification**:
   - Formalize the mathematical formulation and algorithmic pipeline for MV-HDBSCAN.
   - Detail stability calculations, condensed tree generation, and multi-view cluster extraction strategies.

3. **Implementation & Testing**:
   - Implement the core algorithm in `src/`.
   - Create unit tests and evaluate on multi-view synthetic and real-world benchmark datasets in `tests/`.
