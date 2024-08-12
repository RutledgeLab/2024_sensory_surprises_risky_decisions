# Surprising sounds influence risky decision making

### FOLDER CONTENTS
* `run_mainfigures.m` is the main script that produces the key analyses and figures in the manuscript

* `/data_mat` contains the anonymized datasets for all seven experiments (total n = 1600). Data collected from 24 Feb 2021 to 05 Sept 2023. These datasets, which are saved as MATLAB struct arrays are loaded in `run_mainfigures.m` to compute all results.

* `/data_csv` contains the same anonymized datasets for all seven experiments (total n = 1600), reformatted to csv table format

* `fitmodel_omnibus_persevrb.m` contains the script that defines and runs the main models. This script uses Maximum Likelihood Estimation (MLE) for parameter estimation.

* `fitmodel_omnibus_rb.m` contains the script that defines and runs the Risky Bias model. This script uses Maximum Likelihood Estimation (MLE) for parameter estimation.

* `fitmodel_pt.m` contains the script that defines and runs a basic Prospect Theory model

* `fitmodel_dLapsemodel.m` contains the script that defines the dLapse model featured in figure 4

* `ttest_bf.m` is the function called to run Bayes Factor calculations

* `sigstar.m` is the function called to generate significance stars for plotting purposes

* `setFigureDefaults.m` is the function that sets the default aesthetics for figures in the paper


### SYSTEM REQUIREMENTS

- MATLAB Version: 9.13.0.2049777 (R2022b)
- Operating System: macOS  Version: 13.5.2 Build: 22G91 
- Java Version: Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
- Optimization Toolbox Version 9.4 (R2022b)  
- Statistics and Machine Learning Toolbox Version 12.4 (R2022b)  

#### Versions tested: 

- MATLAB 9.13.0.2049777 (R2022b)
- MATLAB 9.8.0.1873465 (R2020a) Update 8
  -'swarmchart' function in 'run_mainfigures.m' requires R2020b or newer versions of MATLAB
- No required non-standard hardware

### INSTALLATION GUIDE

* MATLAB Quickstart guide: https://www.mathworks.com/matlabcentral/answers/897247-matlab-installation-quick-start-guide

* Typical install time: 30-45 minutes 

### DEMO & INSTRUCTIONS FOR USE

1. Open 'run_mainfigures.m'. 
2. Run the script 
	* Expected run time: 3.5 minutes
	* Expected output: All Figures (2A,2B,2C,2D,3A,3B,3C,3D,4,5B,5C,6B,6C,7) from the manuscript. Key statistics from figures are printed in the console
3. Different studies can be selected in the '%% select a single study to make plots for' section (line 46-50) to produce plots and analyses for different datasets. 
	* For example: '%% Figure 5 & 6: Both Persev & Bias Effects on the same plot.' can produce Figures 5C1, 5C2, 6C1, 6C2 from the manuscript if Experiments 3,4,5,6,or 7 are assigned to 'alldata' variable on line 44 and the analysis script is rerun.














