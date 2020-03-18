![Metabolic_Metro_Map](Metabolic_Metro_Map.png)

_The above picture is exported from_ [Wikipedia](https://en.wikipedia.org/wiki/Metabolic_pathway).

## Basic Description

We present mlLGPR (**m**ulti-**l**abel **L**ogistic Re**G**ression for **P**athway P**R**ediction) is a software package that uses supervised multi-label classification and rich pathway features to infer metabolic networks at the individual, population and community levels of organization.

## Dependencies

- mlLGPR is tested to work under Python 3.5
- [NumPy](http://www.numpy.org/) (>= 1.15)
- [scikit-learn](https://scikit-learn.org/stable/) (>= 0.20)
- [pandas](http://pandas.pydata.org/) (>= 0.23)
- [NetworkX](https://networkx.github.io/) (>= 2.2)

## Demo

### Options
To display mlLGPR's running options, use: `python main.py --help`. It should be self-contained. 

## Installation and Basic Usage
Run the following commands to clone the repository to an approriate location:

``git clone https://github.com/hallamlab/mlLGPR.git``

For all experiments, navigate to ``src`` folder then run the commands of your choice. For example, to display *mlLGPR*'s running options use: `python main.py --help`. It should be self-contained. 


All the command arguments are initiated through [main.py](main.py) file. 
- You need to obtain [MetaCyc](https://metacyc.org/) database in order to extract information. Please modify the content of ``Path.py`` inside utility folder as necessary.
- In addition, please download six database: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc from [biocyc](https://biocyc.org/).


#### Example
To extract information from [MetaCyc](https://metacyc.org/), create golden and synthetic samples, train mlLGPR using elastic-net, evaluate, and predict on dataset, simply set the arguments in the [main.py](main.py) file as:

```python main.py --biocyc --train --evaluate --predict --build_syn_dataset --nSample 15000 --average_item_per_sample 500 --build_synthetic_features --build_golden_dataset --build_golden_features --extract_info_mg --build_mg_features --ds_type "syn_ds" --trained_model "mlLGPR_en_ab_re_pe.pkl" --kbpath "[MetaCyc location]" --dspath "[Location to the processed dataset]" --mdpath "[Location to store or save the model]" --rspath "[Resuls location]" --ospath "[Object location]" --n_jobs 10 --nEpochs 10 --nBatches 5```

where 
- Object location: The location to the data object that contains extracted information from the MetaCyc database and all the datbases.


## Citing
If you find *mlLGPR* useful in your research, please consider citing the following paper:
- M. A. Basher, Abdur Rahman, McLaughlin, Ryan J., and Hallam, Steven J.. **["Metabolic pathway inference using multi-label classification with rich pathway features."](https://www.biorxiv.org/content/10.1101/2020.02.02.919944v1.abstract)**, bioRxiv (2020).

## Contact
For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)

## Upcoming features
- Incorporate graph based learning.
