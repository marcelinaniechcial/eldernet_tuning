# Transfer Learning of Self-supervised Network for Parkinson’s Disease Digital Biomarker

## Abstract
Clinical assessments provide limited insight into Parkinson’s disease (PD) digital biomarkers. More generalizable data could be collected in real-life measurements from wrist‑worn sensors (Evers et al., 2020). The reduced arm swing is one of the early motor signs of Parkinson's disease that can be used as a digital biomarker. The key part of providing an accurate quantification of an arm swing is detecting gait without other arm activities. In this work, we focus on using transfer learning for that task by tuning the self-supervised ElderNet model (Brand et al., 2024) on the free-living Parkinson@Home annotated dataset (Post E., 2025).  We tune two models, first to recognize gait in Parkinson’s Patients and second to detect gait segments without other arm activities. The models achieved high AUC scores of over 0.97 and 0.9 on PD Patients. Additionally, we compare the tuned models with established classifiers such as Logistic regression and Random Forest  (Post et al., 2025). Our comparison shows the tuned models do not outperform the traditional


## Methods

### Data Processing
The Parkinson@Home dataset contains annotated free-living accelerometer data from 25 Parkinson’s Patients and 25 controls (Post E., 2025). In the tuning process we only used data from PD Patients, however, the model was tested also on the controls. The labels include the position of the sensor, type of currently performed activity and information about the medications intake. In the processing, we downsampled and interpolated signals from 200 Hz to 30 Hz. Afterwards, the data was split into 10s windows and normalized. Each window was given a label based on the dominant activity at that period. 

### Tuning ElderNet

The image below shows the pre-trained ElderNet architecture and training pipelines  (Brand et al., 2024). We tune this model on the labeled dataset for 15–25 epochs using Cross-entropy loss and a learning rate scheduler. Tuning was performed with Leave-One-Out Cross Validation across multiple sets of hyperparameters.


## Results 

Please find the detailed numerical comparisons and tables [here](Poster.pdf).

### Model 1
The network shows consistent performance across the less and more affected sides of Parkinson's Patients. As expected, performance on controls is lower, since none were included in the tuning process. Comparison with traditional models shows performance close to Logistic Regression and a wider gap to the leading Random Forest.

### Model 2
The task of recognizing gait without other arm activities is significantly harder than recognizing any gait segment. The network performs noticeably worse than the other classifiers, with larger differences seen between LAS and MAS. Note that the dataset does not contain arm activity labels for controls.


## Discussion

There is a limited amount of available annotated accelerometer data from Parkinson’s Patients. Nevertheless, the use of self-supervised pre-trained weights from the model trained on the older adults allowed the network to achieve accurate classification on PD Patients. The first task of gait recognition showed ElderNet to perform comparably to standard classifiers. A larger performance gap, indicating the superiority of simpler models, was observed in the more complex task of recognizing gait without other arm activities. This is likely because the task diverges significantly from ElderNet's pre-trained objective. Further research is necessary to generalize these results to other neural network architectures.

Brand YE, Kluge F, Palmerini L, Paraschiv-Ionescu A, Becker C, Cereatti A, Maetzler W, Sharrack B, Vereijken B, Yarnall AJ, Rochester L, Del Din S, Muller A, Buchman AS, Hausdorff JM, Perlman O. Automated Gait Detection in Older Adults during Daily-Living using Self-Supervised Learning of Wrist-Worn Accelerometer Data: Development and Validation of ElderNet. Res Sq [Preprint]. 2024 Mar 15:rs.3.rs-4102403. doi: 10.21203/rs.3.rs-4102403/v1. Update in: Sci Rep. 2024 Sep 6;14(1):20854. doi: 10.1038/s41598-024-71491-3. PMID: 38559043; PMCID: PMC10980143.

Brand, Y.E., Kluge, F., Palmerini, L. et al. Self-supervised learning of wrist-worn daily living accelerometer data improves the automated detection of gait in older adults. Sci Rep 14, 20854 (2024). https://doi.org/10.1038/s41598-024-71491-3

Evers LJ, Raykov YP, Krijthe JH, Silva de Lima AL, Badawy R, Claes K, Heskes TM, Little MA, Meinders MJ, Bloem BR. Real-Life Gait Performance as a Digital Biomarker for Motor Fluctuations: The Parkinson@Home Validation Study. J Med Internet Res. 2020 Oct 9;22(10):e19068. doi: 10.2196/19068. PMID: 33034562; PMCID: PMC7584982 \\

Post E, Laarhoven TV, Raykov YP, Little MA, Nonnekes J, Heskes TM, Bloem BR, Evers LJW. Quantifying arm swing in Parkinson's disease: a method accounting for arm activities during free-living gait. J Neuroeng Rehabil. 2025 Feb 26;22(1):37. doi: 10.1186/s12984-025-01578-z. PMID: 40011957; PMCID: PMC11863854.
