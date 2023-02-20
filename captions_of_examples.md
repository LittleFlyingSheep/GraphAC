The table below presents the caption examples corresponding to Figure 2 of the paper titled "Graph Attention for Automated Audio Captioning". In example 1, the predicted caption of P-Transformer does not involve "ringing between short pauses", while the proposed GraphAC captioned this content as "repeatedly". In example 2, P-Transformer misrepresents the semantic content "making noises by snoring and groaning" of the ground truth as "barking", while the proposed GraphAC captions this semantic content as "while sleeping". The proposed GraphAC precisely captions audio signals by modelling the contextual association (i.e., long-time dependencies) within audio features. Moreover, the provided examples of GraphAC and GraphAC w/o top-k show that the top-k mask strategy helps to effectively capture the semantic information about acoustic scenes and events to improve the quality of the predicted captions.

<caption><b>Table 1. </b>Example 1.</caption>

| Method            | Caption                                                            |
| ----------------- | ------------------------------------------------------------------ |
| Ground Truth      | Five different sounding bells are **ringing between short pauses** |
| P-Transformer     | A person is tapping on a hard surface                              |
| GraphAC w/o top-k | A person is hitting a metal object with a stick                    |
| **GraphAC**       | A metal object is striking another metal object **repeatedly**     | 

<caption><b>Table 2. </b>Example 2.</caption>

| Method            | Caption                                                               |
| ----------------- | --------------------------------------------------------------------- |
| Ground Truth      | A small dog with a flat face is making noises by snoring and groaning |
| P-Transformer     | A dog is growling and growling as it is *barking*                     |
| GraphAC w/o top-k | A dog is growling **while sleeping**                                  |
| **GraphAC**       | A dog is growling and growling **while sleeping**                     | 
