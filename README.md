![Metabolic_Metro_Map](Metabolic_Metro_Map.svg)

_The above picture is exported from_ [Wikipedia](https://en.wikipedia.org/wiki/Metabolic_pathway).

## WARNING: mlLGPR is currently not active. Please use [triUMPF](https://github.com/hallamlab/triUMPF).

## Basic Description

mlLGPR (**m**ulti-**l**abel **L**ogistic Re**G**ression for **P**athway P**R**ediction) is a novel pathway prediction framework to recover metabolic pathways from large-scale metagenomics datasets and tackle some pathway related obstacles. Unlike the conventional supervised methods that assume each sample is associated with a single class label within a number of candidate classes, a metagenomics dataset usually comprises of multiple pathways per sample, thus, putting the problem in the context of a multi-label classification approach. The originality of our method lies:
- in the search for potential pathways (given enzymatic reactions as inputs)
- in the extraction of pathway/reaction transformation patterns
- in the evaluation strategies and significance tests
- in the large-scale applicability owing to the computational efficiency. 

Our comprehensive analysis using seven designed experimental protocols demonstrates the benefits of our proposed method, enabling fast and relatively efficient prediction on a large-scale metagenomics dataset. 

## Dependencies

- mlLGPR is tested to work under Python 3.5
- [NumPy](http://www.numpy.org/) (>= 1.15)
- [scikit-learn](https://scikit-learn.org/stable/) (>= 0.20)
- [pandas](http://pandas.pydata.org/) (>= 0.23)
- [NetworkX](https://networkx.github.io/) (>= 2.2)

## Demo

### Options
To display mlLGPR's running options, use: `python3 main.py --help`. It should be self-contained. 

### Basic Usage

The mlLGPR comes in two flavours: 
- elastic-net model (mlLGPR-EN)
- fused coefficients (mlLGPR-FC)

All the command arguments are initiated through [main.py](main.py) file.

#### Example 1

To extract information from [MetaCyc](https://metacyc.org/), create golden and synthetic samples, train mlLGPR using elastic-net, evaluate, and predict on dataset, simply set the arguments in the [main.py](main.py) file as:

```python3 main.py --biocyc --metagenomic --train --evaluate --predict --build_syn_dataset --nSample 15000 --average_item_per_sample 500 --build_synthetic_features --build_golden_dataset --build_golden_features --extract_info_mg --build_mg_features --ds_type "syn_ds" --trained_model "mlLGPR_en_ab_re_pe.pkl" --n_jobs 10 --nEpochs 10 --nBatches 5```

#### Example 2

To perform all of the analysis using fused coefficients use:

```python3 main.py --biocyc --metagenomic --train --evaluate --predict --build_syn_dataset --nSample 15000 --average_item_per_sample 500 --build_synthetic_features --build_golden_dataset --build_golden_features --extract_info_mg --build_mg_features --build_pathway_similarities --ds_type "syn_ds" --adjust_by_similarity --trained_model "mlLGPR_en_ab_re_pe.pkl" --n_jobs 10 --nEpochs 10 --nBatches 5```


## Citing

If you employ mlLGPR in your research, please consider citing the following papers presented at PLOS 2019 and [] 2019:

- [To be Added]


## Contact

For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)
