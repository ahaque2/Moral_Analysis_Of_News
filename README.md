# Analyzing Moral Framing in War News Using A Vector Subspace Projection Approach

## Running the code

This repo includes code to conduct moral framing analysis using text. The implementation uses eMFD (enhanced Moral Foundation Dictionary) to get word pairs and identifies moral vector subspaces corresponding to each moral foundation. By default the code uses BERT-base-cased, but can be changed to other models. The current implementation is for news headline analysis, but can be used for any text.

- **1_identify_moral_subspace.py** computes the moral foundation vector subspace. 
To run the code execute python3 1_identify_moral_subspace.py --high 100 --low 100 --num_keep 100 --moral_foundation care --subspace_dim 100
If run without params is runs with default params (200, 200, 200, care, 1). The best params for each moral foundation based on our experiments are included in the code. We recommend using them for optimal performance. 

- **2_Model_Evaluation.py** evaluates the identified vector subspaces against emfd scores.
It computes the Pearson Correlation between moral foundation scores computed by projecting the contextual embeddings of words to the identified vector moral subspaces. To choose a moral foundation use --moral_foundation param as follows. 
python3 2_Model_Evaluation.py --moral_foundation fairness

- **3_Compute_Moral_Scores_for_News.py** computes moral foundation scores for news headlines.
The dataset to be used is under the 'data' folder.

- **4a_explore_results.ipynb** code containing visualizations to explore the differences in coverage of conflicted entities in war news.

- **4b_statistical_tests.ipynb** code for statistical analysis.

- **4c_Granger_causality.ipynb** code to conduct granger causality test to identify cross-publisher influence.
