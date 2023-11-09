## Surprising sounds bias risky decision making
## DBPR NCOMMS-23-08295

OVERVIEW
-----------------------------------------------------------------------------------------------------
Data collected from 24 Feb 2021 to 05 Sept 2023


FOLDER CONTENTS
-----------------------------------------------------------------------------------------------------
* 'run_mainfigures.m' is the main script that produces the key analyses and figures in the manuscript

* 'Exp[1-7]_cleaned.mat' contain the data for each of the seven experiments (total n = 1600) discussed in the manuscript

* 'fitmodel_omnibus_OBdiff_model.m' contains the script that defines and runs the main models. This script uses Maximum Likelihood Estimation (MLE) for parameter estimation.

* 'fitmodel_pt.m' contains the script that defines and runs a basic Prospect Theory model

* 'fitmodel_pt_dLapsemodel.m' contains the script that defines the dLapse model featured in figure 4

* 'ttest_bf.m' is the function called to run Bayes Factor calculations

* 'sigstar.m' is the function called to generate significance stars for plotting purposes

* 'setFigureDefaults.m' is the function that sets the default aesthetics for figures in the paper


SYSTEM REQUIREMENTS
-----------------------------------------------------------------------------------------------------

MATLAB Version: 9.13.0.2049777 (R2022b)
Operating System: macOS  Version: 13.5.2 Build: 22G91 
Java Version: Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
-----
MATLAB                                                Version 9.13        (R2022b)  
Optimization Toolbox                                  Version 9.4         (R2022b)  
Statistics and Machine Learning Toolbox               Version 12.4        (R2022b)  

* Versions tested: 
	MATLAB 9.13.0.2049777 (R2022b)
	MATLAB 9.8.0.1873465 (R2020a) Update 8
		-'swarmchart' function in 'run_mainfigures.m' requires R2020b or newer versions of MATLAB

* No required non-standard hardware

INSTALLATION GUIDE
-----------------------------------------------------------------------------------------------------

* MATLAB Quickstart guide: https://www.mathworks.com/matlabcentral/answers/897247-matlab-installation-quick-start-guide

* Typical install time: 30-45 minutes 

DEMO & INSTRUCTIONS FOR USE
-----------------------------------------------------------------------------------------------------

1. Open 'run_mainfigures.m'. 
2. Run the script 
	* Expected run time: 5.5 minutes
	* Expected output: All Figures (2A,2B,2C,2D,3A,3B,3C,3D,4,5,6) from the manuscript. Key statistics from figures are printed in the console
3. Different studies can be selected in the '%% select a single study to make plots for' section (line 40-50) to produce plots and analyses for different datasets. 
	* For example: '%% Figure 5 & 6: Both Persev & Bias Effects on the same plot.' can produce Figures 5 and 6 from the manuscript if Exp 3-6 are assigned to'alldata' variable on line 40 and the analysis script is rerun.














