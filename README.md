# Epi-LSTM
Latent representation of the human pan-celltype epigenome through a deep recurrent neural network

## Data Preparation
Download the epigenomic data pertaining to histone modifications and chromatin accessibility (ChIP-seq and DNase-seq) from the [Roadmap Epigenomics Consortium](http://www.roadmapepigenomics.org/). Use the arcsinh transformation on the signal to stabilize the variance of these signals and lessen the effect of outliers. The downloaded data consists of different Assays performed in various Cell Types. 

## Epi-LSTM Model 
The autoencoder forms the backbone of Epi-LSTM. The high dimensional sequential input assays are reconstructed via a low dimensional bottleneck using LSTMs coupled with a loss function that forces the output to be as close to the input as possible in euclidean space. The first stage of the framework is a LSTM that acts as an encoder. The encoder reads the input sequence and creates a fixed length low dimensional vector representation in the form of an embedding. The low dimensional
representation at each position in the sequence is then treated as an annotation for that position. The decoder uses the fixed length vector embedding as it’s
initial hidden state and tries to recreate the original sequence. The epigenomic data is fed into the Epi-LSTM in frames of 100 steps to fit the model into memory and to speed up training. As the epigenomic data has a resolution of 25, i.e., a data point for every 25 base pairs, Epi-LSTM is capable of dealing with 2500 positions in a given frame. For more details please refer to the ```Methods-Epi-LSTM Autoencoder Framework``` section of the manuscript. The trained models are stored under the ```data``` folder as ```encoder.pth``` and ```decoder.pth```.

The model can be found in the ```src/model.py``` file. The model maninly consists of two parts, encoder which can be found in ```encoder.py``` and the decoder, which can be found in ```decoder.py```. The encoders and decoders can be replaced/manipulated using these files. 

## Training Epi-LSTM 
The training fuctions are in the ```src/train_fns``` folder. To train a new model with all the epigenomic data, run the ```train_gene.py``` function. 

Pass the chromosome to train as input to the ```train_iter_gene(config, chr)``` function. Inside ```train_iter_gene```, ```data_ob_gene.prepare_id_dict()``` call, prepares the cell type assay combinations to train on. Alter this dictionary to train on a smaller subset of data. 

```data_ob_gene.get_data()``` then creates a generator to deliver data frame by frame, which is then fed as input to the LSTM Autoencoder in the ```unroll_loop``` function. Set all the tunable parameters in the ```config.py``` file. 

## Testing Epi-LSTM 
To test an existing model, run the ```test_gene.py``` function with the chromosome as input. Load an existing model using ```model.load_weights()```. The data preparation follows similar steps as training but the LSTM Autoencoder is set to evaluation and does not compute gradients. The test low-dimensional representations can then be stored to be used for downstream tasks as in ```encoder_hidden_states_np = encoder_hidden_states.squeeze(1).cpu().data.numpy()```.  

## Downstream Classification
The stored representations can be used for downstream classification of genomic phenomena like Gene Expression, Promoter-Enhancer Interactions, Frequently Interacting Regions and Replication Timing. The data for these can be obtained from resources pointed to in the ```Methods-Datasets``` section of the manuscript. To start the downstream classification tasks, run ```run_downstream.py``` with the chromosome and trained model as input. Classification is carried out using the XGBoost frameowrk, details of which can be found in the ```Methods-Downstream Classification``` section in the manuscript. 

## Plotting 
Most of the plotting can be done in ```eda/plot_map.py```. The results are pre-stored, they can be loaded and visualized. 



