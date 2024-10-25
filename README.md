## Contents

- **Design_A_log_best**: Contains design A's logs, including TensorBoard charts, model weights, and evaluation results.
- **Design_B_log** and **Design_C_log**: Similar logs for designs B and C respectively.
- **test_run**: Logs of the default run.
- **train.py**: Script for training the model, including the default training run.
- **evaluate.py**: Script for visualizing the performance of a particular model.
- **search_engine.py**: Implementation of the search engine functionality.
- **plot.py**: Code for generating comparison plots of different model designs.
- **utils.py**: Helper code for reading and processing evaluation results.

## Usage

To use search engine specify designs in designs dictionary and run

```sh
python3 search_engine.py
```


## Evaluation

To vizualize certain model use:

```sh
python3 evaluate.py log_dir run_num
```


## Requirements

```sh
pip install -r requirements.txt
```



## Sign
Maksim Aleksandrov
