---
language:
- sgn
license:
- afl-3.0
multilinguality:
- translation
pretty_name: CISLR
size_categories:
- 1K<n<10K
source_datasets:
- original
tags:
- Indian Sign Language
- Sign Language Recognition
---

# CISLR: Corpus for Indian Sign Language Recognition

This repository contains the Indian Sign Language Dataset proposed in the following paper

> **Paper:** CISLR: Corpus for Indian Sign Language Recognition https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.707/

> **Authors:** Abhinav Joshi, Ashwani Bhat, Pradeep S, Priya Gole, Shashwat Gupta, Shreyansh Agarwal, Ashutosh Modi <br>
>
> **Abstract:** *Indian Sign Language, though used by a diverse community, still lacks well-annotated resources for developing systems that would enable sign language processing. In recent years researchers have actively worked for sign languages like American Sign Languages, however, Indian Sign language is still far from data-driven tasks like machine translation. To address this gap, in this paper, we introduce a new dataset CISLR (Corpus for Indian Sign Language Recognition) for word-level recognition in Indian Sign Language using videos. The corpus has a large vocabulary of around 4700 words covering different topics and domains. Further, we propose a baseline model for word recognition from sign language videos. To handle the low resource problem in the Indian Sign Language, the proposed model consists of a prototype-based one-shot learner that leverages resource rich American Sign Language to learn generalized features for improving predictions in Indian Sign Language. Our experiments show that gesture features learned in another sign language can help perform one-shot predictions in CISLR.*


## Directory Structure

    .
    ├── dataset.csv                 # list of all videos with categorical annotations
    ├── prototype.csv               # files used as prototypes 
    ├── test.csv                    # files used as testset
    ├── CISLR_v1.5-a_videos         # dataset videos
        ├── __Rz2PaTB1c.mp4
        ├── _2TlWc7fctg.mp4
        .
        .
        ├── zZVuyuVTFW0.mp4
    └── I3D_features.pkl            # extracted Inception3D features



## Citation
> Abhinav Joshi, Ashwani Bhat, Pradeep S, Priya Gole, Shashwat Gupta, Shreyansh Agarwal, and Ashutosh Modi. 2022. CISLR: Corpus for Indian Sign Language Recognition. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10357–10366, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

