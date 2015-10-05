# MortgageMachineLearning
This repository is the rough working code for the GOTO CPH 2015 talk by Phil Winder title Modern Fraud Prevension using Deep Learning. It was produced in a rush, so probably has some issues.

# Requirements
## DB requirements
postgres

## Python requirements
Entire Scipy stack
sklearn
keras
pandas
pg8000

# Examples
## MNIST digits example use
Run KerasMNISTTest.py. It should download the data and produce outputs in a ./plots/ folder.

## Speaker example use
Speaker data is available from:  http://web.mit.edu/6.863/share/nltk_lite/timit/

To run the code, start by running speakerPreprocess.py, to compute the STFT of all the audio data. Then, to perform the classification, run speakerDeepLearning.py.

## Mortgate example use
Mortgate data is available from: http://www.freddiemac.com/news/finance/sf_loanlevel_dataset.html

See the loader files for some tips.

Start by loading the database with the data using the load_raw_mac or load_raw_ubuntu files. Then run the classification code in MortgageDeepLearning.py. It should generate some plots in the ./plots/ folder

## Mortgage Decision Tree example
Add the data as per Mortgage Deep Learning instructions. Run MortgageRandomForest.py. Outputs should be in the ./plots/ folder.

# Acknowledgements
This code was based upon code from Todd Schneider, many thanks.
https://github.com/toddwschneider/agency-loan-level

Thanks also to Python, SKLearn, Keras, PG8000 devs. Ta very much.