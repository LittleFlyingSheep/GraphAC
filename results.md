GraphAC is the audio captioning method proposed in "Graph Attention for Automated Audio Captioning". 
GraphAC-attention is a variant of GraphAC that replaces the graph attention module into the Transformer encoder.
The comparison as follows shows that GraphAC can achieve better SPIDEr performance, 
which verifies the use of graph attention may be better than using Transformer encoder.

| Metrics   | GraphAC | P-attention AC | 
| --------- | --------- | ------------- |
| BLEU$_1$  | 0.5807    | 0.5839        | 
| BLEU$_2$  | 0.3863    | 0.3931        | 
| BLEU$_3$  | 0.2654    | 0.2679        |
| BLEU$_4$  | 0.1806    | 0.1797        |
| METEOR    | 0.1754    | 0.1773        |
| ROUGE$_l$ | 0.3850    | 0.3909        |
| CIDEr     | 0.4366    | 0.4316        |
| SPICE     | 0.1256    | 0.1286        |
| SPIDE$_r$ | **0.2811**    | 0.2801        |
