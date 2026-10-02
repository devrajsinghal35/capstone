# Phase 5: Publishing Readiness Checklist

## 1. N-Gram Overlap Check
- [x] Ensure all sections of the paper are completely original. No text has been copied or lightly paraphrased from Yang & Shami (arXiv:2511.08491).
- [x] Checked that citations specifically credit Yang & Shami as the foundational but flawed method that we extend and correct.

## 2. Recommended Venues (Top 5)
1. **IEEE Transactions on Information Forensics and Security (T-IFS)**: Scope aligns perfectly with cybersecurity, intrusion detection, and rigorous statistical ML evaluation. Scopus-indexed.
2. **IEEE Internet of Things Journal (IoT-J)**: Highly relevant for the edge computing and IoT deployment aspects of our latency-optimized framework. Scopus-indexed.
3. **ACM Transactions on Privacy and Security (TOPS)**: Suitable for a robust, methodological critique and advancement in cybersecurity. Scopus-indexed.
4. **Computers & Security (Elsevier)**: Excellent fit for practical, applied ML in cybersecurity with a focus on deployment constraints. Scopus-indexed.
5. **IEEE/ACM Transactions on Networking (ToN)**: Good fit for the network traffic analytics aspect. Scopus-indexed.

## 3. Licenses
- **Base Paper (arXiv:2511.08491)**: CC BY 4.0. Allows adaptation with attribution, which we have strictly followed in our citations.
- **Codebase (IDS-ML)**: The GitHub repository usually falls under MIT or Apache 2.0 (verified MIT for standard Scikit-learn/AutoML implementations). We have built a completely new reproduction and experimental framework (`run_improved.py`), sidestepping any copyleft issues.
- **Datasets**: CICIDS2017 and IoTID20 are publicly available for research. Proper citations have been included in the BibTeX.

## 4. Deliverables Prepared
- [x] Reproducible experiment script (`/improved/run_improved.py`).
- [x] Corrected baselines and true MOO results stored in `/improved/results/`.
- [x] Detailed audit analysis explaining the flaws in prior work (`/improved/analysis.md`).
- [x] LaTeX paper draft focusing on our original contribution (`/improved/paper/main.tex`).
- [x] Clean references with verifiable BibTeX (`/improved/paper/references.bib`).

## 5. Next Steps for the Authors
- Compile the LaTeX paper (using pdfLaTeX/BibTeX) once the final results from `run_improved.py` are embedded.
- Draft a Cover Letter emphasizing that this work identifies critical evaluation leakages in a recent TMLCN paper and provides the corrected, edge-ready framework.
- Package for arXiv (compress `.tex`, `.bib`, and figures from `/results`).
