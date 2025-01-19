# LTRS
### LTRS: Improving Word Sense Disambiguation via Learning to Rank Senses

Word Sense Disambiguation (WSD) is a fundamental task critical for accurate semantic understanding. Conventional training strategies usually only consider predefined senses for target words and learn each of them from relatively limited instances, neglecting the influence of similar ones. To address these problems, we propose the method of Learning to Rank Senses (LTRS) to enhance the task. This method helps a model learn to represent and disambiguate senses from a broadened range of instances via ranking an expanded list of sense definitions. By employing LTRS, our model achieves a SOTA F1 score of 79.6% in Chinese WSD and exhibits robustness in low-resource settings. Moreover, it shows excellent training efficiency, achieving faster convergence than previous methods. This provides a new technical approach to WSD and may also apply to the task for other languages.


## How to Use the Code

### Training the Model
To train the model, use the following command:
```bash
python trainer.py --mode train --encoder-name ./model --bge_model_path ./bge_model --ckpt ./output --device cuda
```

### Evaluating the Model
To evaluate the model, use the following command:
```bash
python trainer.py --mode evaluate --ckpt ./output --device cuda:0
```

### Citation
If you use this code, please cite our paper:
```bibtex
@inproceedings{
}
```

### Acknowledgements
This paper is supported by the National Natural Science Foundation of China (No. 62036001) and the National Social Science Foundation of China (No. 18ZDA295).

For more details, please refer to our [paper](https://github.com/COOLPKU/LTRS/paper.pdf).
