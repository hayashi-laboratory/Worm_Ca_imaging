# Worm_Ca_imaging

====

This script is for analyzing Ca imaging data in C. elegans.  

Input data  
1. single plane multiple neuron Ca imaging data  
(.csv, which is pre-analyzed by Trackmate in Fiji) 
2. representative image   
(.png, for identifying the neurons) 

## Usage  
1. run the script  
2. select the directory which contain CSV and PNG files. CSV and PNG files should be same name (Experiment1.csv and Experiment1.png)  
3. Wait for calculation  

## Outputs  
Analyzed_yyyy_mm_dd  
	Experiment-001   
		datas   
			FFT_data    
			motion_bout_analysis  
				ID0   
				ID1  
			quiescent_bout_analysis  
				ID0  
				ID1  
			rawdata  
				ID0  
				ID1  
			correlation.csv  
			locomotor_activity.csv  
			locomotion_bool.csv  
		figures  
		Neural_positions  
	readme.txt  


---  
## Licence  
[MIT]  

---
## Author  
[Shinichi Miyazaki] https://github.com/Rukaume

