Introduction
This repository provides source code of IGBFL.

IGBFL is a feature selection approach that aims at handling the high dimension problem rooted in the natural of the program test suite.

Environment
OS: Linux
Python package:
pandas==0.25.1
chardet==3.0.4
numpy==1.16.5
torch==1.9.0
Structure
The structure of the repository is as follows:

calculate_suspiciousness
|____CalculateSuspiciousness.py	:calculate suspiciousness of each statement and give the MFR rank or MAR rank according to the real fault line.
data
|____d4j :Defects4J dataset	
|____manybugs :MangBugs dataset	
|____sir :SIR dataset		
|____motivation :artificial dataset	
data_process
|____data_undersampling :undersampling approaches
|	 |____undersampling.py
|____dimensional_reduction :feature selection
|	 |____Entropy.py		
metrics : SFL and DLFL metrics
|____calc_corr.py
|____dl_metrics.py
|____metrics.py
pipeline
|____Pipeline.py : load different type of data, process data and calculate suspiciousness task
read_data : load data according to args
|____DataLoader.py
|____Defects4JDataLoader.py
|____ManyBugsDataLoader.py
|____SIRDataLoader.py
results : store the results in txt format
utils : some utils during pipeline
|____args_util.py
|____file_util.py
|____read_util.py
|____write_util.py
run.py : program entry
Usage
To run the program, commandline parameters are needed.

**required arguments: **

name	meaning	value
-d	dataset	"d4j", "manybugs","SIR"
-p	program	"Chart", "Closure", "Time", "Lang", ...
-i	bug_id	"1", "2", ...
-m	method	"dstar", "ochiai", "barinel", "MLP-FL", ...
-e	experiment	"origin", "resampling", "undersampling", ...
optional arguments:

To show how to run the program, we give the examples of "illustrative example" in our paper.

original method
run.py -d manybugs -p motivation -i artificial_bug -m GP02 -e origin
After the program end, the MAR rank as the following format will store in result folder.

motivation-artificial_bug   GP02   12 

motivation-artifiial_bug is the program with bug_id, GP02 is the fault localization method and 12 is the final rank that locate the first bug of the program by using the method.

resampling method
run.py -d manybugs -p motivation -i artificial_bug -m GP02 -e resampling
motivation-artificial_bug   GP02   7

undersampling method
run.py -d manybugs -p motivation -i artificial_bug -m GP02 -e undersampling
motivation-artificial_bug   GP02   7  


feature selection
run.py -d manybugs -p motivation -i artificial_bug -m GP02 -e fs
motivation-artificial_bug   GP02   9  



