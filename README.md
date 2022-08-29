This is a fork of the DDSP Singing Vocoders repo with the following
modifications:
* preprocess.py has been altered to generate spectrograms compatible with
  SortAnon's ControllableTalkNet.
* solver.py has been altered (hacky) to start off an existing checkpoint if
  available to allow training to be started and stopped on the same machine
  (i.e. for local training).
* A new training/inference config, `configs/sawsinsub_talknet.yaml` has been
  added for a model that can synthesize audio from
  ControllableTalkNet-compatible spectrograms.
* A checkpoint for a model trained from Twilight Sparkle's singing data has
  been included under exp/ts-full.
* Debuzzing has been added as a command line parameter.

# Usage with ControllableTalkNet
1. Generate spectrograms using my [ControllableTalkNet
   fork.](https://github.com/effusiveperiscope/ControllableTalkNet)
2. Create a folder for containing spectrograms for inference: `mkdir
   talknet_infer`. Move/copy spectrograms to that folder.
3. Create an output folder: `mkdir talknet_out`
4. Run inference:
```
python main.py --config ./configs/sawsinsub_talknet.yaml \
		--stage inference \
		--model SawSinSub \
		--input_dir  ./talknet_infer \
        --output_dir ./talknet_out \
		--model_ckpt ./exp/ts-full/sawsinsub-256/ckpts/vocoder_best_params.pt
```

# DDSP Singing Vocoders
Authors: [Da-Yi Wu](https://github.com/ericwudayi)\*, [Wen-Yi Hsiao](https://github.com/wayne391)\*, [Fu-Rong Yang](https://github.com/furongyang)\*, [Oscar Friedman](https://github.com/OscarFree), Warren Jackson, Scott Bruzenak, Yi-Wen Liu, [Yi-Hsuan Yang](https://github.com/affige)
 
 **equal contribution*
 
 
[**Paper**](https://arxiv.org/abs/2208.04756) | [**Demo**](https://ddspvocoder.github.io/ismir-demo/) 


Official PyTorch Implementation of ISMIR2022 paper "DDSP-based Singing Vocoders: A New Subtractive-based Synthesizer and A Comprehensive Evaluation".

In this repository:
* We propose a novel singing vocoders based on subtractive synthesizer: **SawSing**
* We present a collection of different ddsp singing vocoders
* We demonstrate that ddsp singing vocoders have relatively small model size but can generate satisfying results with limited resources (1 GPU, 3-hour training data). We also report the result of an even more stringent case training the vocoders with only 3-min training recordings for only 3-hour training time.

## A. Installation
```bash
pip install -r requirements.txt 
```
## B. Dataset
Please refer to [dataset.md](./docs/dataset.md) for more details.

## C. Training

Train vocoders from scratch. 
1. Modify the configuration file `..config/<model_name>.yaml`
2. Run the following command:
```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub
```
3. Change `--model` argument to try different vocoders. Currently, we have 5 models: `SawSinSub` (Sawsing), `Sins` (DDSP-Add), ` DWS` (DWTS), `Full`, ` SawSub`. For more details, please refer to our documentation - [DDSP Vocoders](./docs/ddsp_vocoders.md).

Our training resources: single Nvidia RTX 3090 Ti GPU

## D. Validation
Run validation: compute loss and real-time factor (RTF).

1. Modify the configuration file  `..config/<model_name>.yaml`
2. Run the following command:

```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml  \
              --stage validation \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --output_dir ./test_gen
```
## E. Inference
Synthesize audio file from existed mel-spectrograms. The code and specfication for extracting mel-spectrograms can be found in [`preprocess.py`](./preprocess.py). 

```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml  \
              --stage inference \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --input_dir  ./path/to/mel
              --output_dir ./test_gen
```

## F. Post-Processing
In Sawsing, we found there are buzzing artifacts in the harmonic part singals, so we develop a post-processing codes to remove them. The method is simple yet effective --- applying a voiced/unvoiced mask. For more details, please refer to [here](./postprocessing/).


## G. More Information
* Checkpoints
  * **Sins (DDSP-Add)**:  [`./exp/f1-full/sins/ckpts/`](./exp/f1-full/sins/ckpts/)
  * **SawSinSub (Sawsing)**:  [`./exp/f1-full/sawsinsub-256/ckpts/`](./exp/f1-full/sawsinsub-256/ckpts/)
  * The full experimental records, reports and checkpoints can be found under the [`exp`](./exp/) folder.
* Documentation
  * [DDSP Vocoders](./docs/ddsp_vocoders.md)
  * [Synthesizer Design](./docs/synth_demo.ipynb)

## H. Citation
```
@article{sawsing,
  title={DDSP-based Singing Vocoders: A New Subtractive-based Synthesizer and A Comprehensive Evaluation},
  author={Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang},
  journal = {Proc. International Society for Music Information Retrieval},
  year    = {2022},
}
```


