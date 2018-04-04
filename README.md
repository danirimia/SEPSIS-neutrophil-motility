# SEPSIS-neutrophil-motility
Microfluidic assay measures neutrophil spontaneous motility in whole blood for sepsis diagnosis

Code developed by Julianne Jorgensen : June 2017 – Massachusetts General Hospital - Boston

Data analysis
Files were converted to standard AVI format	Nikon Elements or ImageJ	-

**Step** |**Software Package** |**Code**
--- |--- |---
Initial processing for removing the background. |Python	|Cell_counter.py, Cell_counter_unstained_new.py, Cell-counter-unstained.py
Cell tracking was performed automatically from brightfield images	ImageCV and SciKit-Learn packages in Python. |Python |Cell_counter.py, Cell_counter_unstained_new.py, Cell-counter-unstained.py
Cell motility pattern identification (track number, video frame, cell diameter, x position, y position, distance, and velocity - definitions in Table S1.) 	|ImageCV, TrackPy, and SciKit-Learn packages in Python.	|Cell_counter.py, Cell_counter_unstained_new.py, Cell-counter-unstained.py


Machine learning

**Step** |**Software Package** |**Code**
--- |--- |---
Convert data for processing	|Python	 |Convert_burns.py, Convert_data.py, Convert_to_format.py
data split 1:2:1 (training data: testing data: held-out set) by patient – part of cross-validation |Python	|Extract_draws.py, Load_data.py, Process_burns.py (originally written for the burns patients, also run on trauma patients)
Support vector machines (SVM) algorithm to separate septic – non-septic patients |Python |Load_data.py, Process.py
Algorithm training on the training data	|Python |Convert_to_format.py
Final graphs and results were produced with the held-out set	|Python	| Extract_draws.py,  Process_burns.py
Regularized linear discriminant analysis via hold-out analysis with cross validation and multiple resampling |Python |Process.py, Process_burns.py 
tSNE graphs	|Python	|Load_data.py
Histogram of the AUC values from the held-out data is graphed |Python	| Process.py, Process_burns.py

