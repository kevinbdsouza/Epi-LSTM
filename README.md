# Epi-LSTM
Latent representation of the human pan-celltype epigenome through a deep recurrent neural network

## Data Preparation
Download the epigenomic data pertaining to histone modifications and chromatin accessibility (ChIP-seq and DNase-seq) from the [Roadmap Epigenomics Consortium](http://www.roadmapepigenomics.org/). Use the arcsinh transformation on the signal to stabilize the variance of these signals and lessen the effect of outliers. The downloaded data consists of different Assays performed in various Cell Types. 

## Epi-LSTM Model 
The autoencoder forms the backbone of Epi-LSTM. The high dimensional sequential input assays are reconstructed via a low dimensional bottleneck using LSTMs coupled with a loss function that forces the output to be as close to the input as possible in euclidean space. The first stage of the framework is a LSTM that acts as an encoder. The encoder reads the input sequence and creates a fixed length low dimensional vector representation in the form of an embedding. The low dimensional
representation at each position in the sequence is then treated as an annotation for that position. The decoder uses the fixed length vector embedding as it’s
initial hidden state and tries to recreate the original sequence. The epigenomic data is fed into the Epi-LSTM in frames of 100 steps to fit the model into memory and to speed up training. As the epigenomic data has a resolution of 25, i.e., a data point for every 25 base pairs, Epi-LSTM is capable of dealing with 2500 positions in a given frame. For more details please refer to the manuscript. 

## Training Epi-LSTM 
The training fuctions are in the ```src/train_fns``` folder. To train a new model with all the epigenomic data, run the ```train_gene.py``` function. 

Pass the Chromosome to train as input to the ```train_iter_gene(config, chr)``` function. Inside ```train_iter_gene```, ```data_ob_gene.prepare_id_dict()``` call prepares the cell type assay combinations to train on. Alter this dictionary to train on smaller subset of data. 

```data_ob_gene.get_data()``` then creates a generator to deliver data frame by frame, which is then fed as input to the LSTM Autoencoder in the ```unroll_loop``` function. Set all the tunable parameters in the ```config.py``` file. 

## Testing Epi-LSTM 

## Downstream Classification



