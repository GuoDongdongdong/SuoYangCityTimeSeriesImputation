# Target
For some inevitable reasons, Suo Yang earthen ruins datasets have a lot of missing value, We need find a way to achieve accurate missing value imputation, for subsequent forecasting task.

# Progress
## Preprocess the Datasets
Generate uniformly distributed missing value in train, validation and test dataset.

## Model Selection
We select many models such as:
1. Based on statistics: previous, linear, quadratic
2. Based on RNN: BRITS, GRUD, MRNN
3. Based on CNN: TimesNet
4. Based on Transformer: ImputeFormer, SAITS
5. Based on GAN: USGAN
6. Based on Diffusion: CSDI

## Experiments

## Improve the Model's Struct

## Comparative experiments and ablation Study

# Some Confuse Problems.
1. train and test in one experiment will different with load model and test in result.csv
because train process use some random seed.
