For better understanding the learnt long-time dependencies of our GraphAC, we provide some caption examples corresponding to Figure 2 of the manuscript in the following Tables. In the example 1, it can be found that the P-Transformer does not describe the content about "ringing between short pauses" in the predicted caption, whereas the proposed GraphAC represent this content as "repeatedly" in the predicted caption. In the example 2, it can be found that the P-Transformer misrepresents the semantic content "making noises by snoring and groaning" of the ground truth as "barking", whereas the proposed GraphAC further summarizes this semantic content and represents it as "while sleeping" in the predicted caption. These examples show that the proposed GraphAC can precisely represent the content of audio signals by modelling the contextual association (i.e., long-time dependencies) within audio features. Moreover, the provided examples of GraphAC and GraphAC w/o top-k can show that the use of the top-k mask strategy can more effectively capture the semantic information about acoustic scenes and events to improve the quality of the predicted captions.

<caption><b>Table 1. </b>Example 1.</caption>

| Method            | Caption                                                            |
| ----------------- | ------------------------------------------------------------------ |
| Ground Truth      | Five different sounding bells are **ringing between short pauses** |
| P-Transformer     | A person is tapping on a hard surface                              |
| GraphAC w/o top-k | A person is hitting a metal object with a stick                    |
| **GraphAC**           | A metal object is striking another metal object **repeatedly**     | 

<caption><b>Table 2. </b>Example 2.</caption>

| Method            | Caption                                                               |
| ----------------- | --------------------------------------------------------------------- |
| Ground Truth      | A small dog with a flat face is making noises by snoring and groaning |
| P-Transformer     | A dog is growling and growling as it is *barking*                     |
| GraphAC w/o top-k | A dog is growling **while sleeping**                                  |
| **GraphAC**       | A dog is growling and growling **while sleeping**                     | 
