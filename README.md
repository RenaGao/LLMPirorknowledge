# LLMPirorknowledge

This is the code for *Prior Knowledge Matters: LLMs Capability via In Context Learning L2 Dialogues under Information-Theoretic Approach*

### The Distribution Divergence Results

| Languages | Metrics  | Quantifiers & Numerals | Tense & Agreement | Reference & Word | Numbers & Agreement | Speech & Acts | Subject Verb Agreement | Modal Verbs & Expressions | Noun Verb Collocation |
| :----:  |:----:  | :----: |  :----:  | :----: | :----: |  :----:  | :----: | :----: |  :----:  |
| **yue**   | d<sub>bi<sub> | ↑0.145                 | ↓0.275           | ↑0.109          | ↑0.099             | ↑0.145     | ↑0.073               | ↑0.052                  | ↓0.066             |
|           |d<sub>mono<sub>| 0.725                  | 0.027            | 0.188           | 0.489              | 0.203      | 0.318                | 0.123                   | 0.029              |
| **th**    |  d<sub>bi<sub> | ↑0.049                 | ↑0.013           | ↑0.097          | ↑0.130             | ↑0.188     | ↑0.060               | ↑0.120                  | ↑0.121             |
|           | d<sub>mono<sub>| 9.227                  | 0.570            | 0.222           | 0.265              | 0.400      | 0.913                | 0.180                   | 0.190              |
| **ja**    |  d<sub>bi<sub> | ↑0.044                 | ↑0.082           | ↑0.265          | ↑0.190             | ↑0.212     | ↑0.087               | ↑0.053                  | ↑0.073             |
|           | d<sub>mono<sub>| 1.954                  | 0.514            | 0.273           | 0.330              | 0.520      | 0.874                | 0.452                   | 0.232              |
| **ko**    |  d<sub>bi<sub> | ↑0.033                 | ↑0.019           | ↑0.009          | ↑0.051             | ↑0.109     | ↑0.148               | ↓0.131                  | ↑0.183             |
|           |d<sub>mono<sub>| 0.654                  | 0.296            | 0.108           | 0.259              | 0.247      | 0.605                | 0.069                   | 0.295              |
| **ms**    | d<sub>bi<sub> | ↑0.007                 | ↑0.036           | ↑0.027          | ↑0.092             | ↑0.076     | ↑0.026               | ↑0.065                  | ↓0.096             |
|           | d<sub>mono<sub>| 1.039                  | 0.321            | 0.109           | 0.341              | 0.279      | 0.477                | 0.097                   | 0.080              |
| **cmn**   |  d<sub>bi<sub> | ↑0.027                 | ↑0.038           | ↑0.065          | ↑0.037             | ↑0.161     | ↑0.023               | ↑0.082                  | ↑0.059             |
|           | d<sub>mono<sub>| 1.382                  | 0.277            | 0.099           | 0.375              | 0.319      | 0.741                | 0.212                   | 0.108              |
| **ur**    |  d<sub>bi<sub> | ↑0.079                 | ↑0.073           | ↑0.062          | ↑0.050             | ↑0.043     | ↑0.046               | ↓0.126                  | ↑0.044             |
|           | d<sub>mono<sub>| 0.918                  | 0.145            | 0.192           | 0.282              | 0.158      | 0.386                | 0.115                   | 0.046              |

### Dependencies
To fully test our code, make sure you have the following dependencies installed:

* PyTorch
* transformers
* openai
* Matplotlib
* pandas
* numpy
* scipy
* sklearn

### Datasets
We use benchmark [ICNALE](https://language.sakura.ne.jp/icnale/).
```bash
mkdir dataset # you can put datasets and intermedia results here.
```

### In-Context Learning for Target Native Conversations
```bash
cd lib
python gpt_incontext.py  --openai_api_key <your openai api key> --output_dir <Path to the output directory>
```
### Annotation for L2 English Dialogues
For both benchmark data and generated data, you can use the same annotation script.
```bash
cd lib
python annotation.py --openai_api_key <your openai api key> --input_dir <Path to the input directory> --output_dir <Path to the output directory> --samples_dir <Path to the intermedia samples directory>
```

### Distrubutions and Results
You can find some scripts and results in **notebooks/** for distribution calculation. We also provide example results of generated dialogue data along with their annotations, available in the **data/** and **annotations/** directories for reference.


