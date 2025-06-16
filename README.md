# Multi-Policy Test Case Selection (MPTCS)
Multi-policy test case selection for test suites designed to test a wide range of common weaknesses in policy behavior. 

## Dependencies 
All dependencies are given in `requirements.txt`, and can be installed as 
```
python -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt
```
If there is an error in the loading of cuSPARSE, the fix is 
```
pip install nvidia-cusparse-cu12==12.1.0.106 --force-reinstall
```

## RQ1 -- Effect of difficulty-score. 
To investigate the effectiveness of the difficulty score, run 
```
python RQ1-ga-effectiveness.py env_name 50 15 experiment_name
python RQ1-mdpfuzz-effectiveness.py env_name 50 15 experiment_name
```

## RQ2 -- Cost-quality trade-off when varying the number of policies.  
To examine the trade-off between computational costs and quality of test cases in the resulting test suites, run 
```
python RQ2-cost-quality.py env_name 50 20 experiment_name 
python RQ2-cost-quality-single-policy.py env_name 50 20 experiment_name
```
The single-policy run runs many more generations to have a single-policy baseline with high-computational costs. 

## RQ3 -- Diversity analysis 
To examine the effect of the general diversity surface of observation variance and policy confidence, run 
```
python RQ3-diversity_examination.py env_name 50 20 experiment_name
```
After all executions have been run, evaluate the diversity by running of the test suites by running
```
python RQ3-evaluate-diversity.py
```
which saves the results to a `results.csv` file in `results/diversity/`. To look at results: 
```
python RQ3-diversity-analysis.py
```