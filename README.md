# Effectivness of semi-supervised NN for Parkinson's biomarkers
This study investigated efectivness of semi-supervised in learning for Parkinson's disease biomarker detection under limited data conditions. We used pre-trainted ElderNet neural network (Brand et al, 2024) and tuned it using a dataset from the Parkinson@home study (Evers et al, 2020). Its performance was then compared directly to a Random Forest and Logistic Regression baseline models (Brand, 2024).

We developed two separate models. The first model focused on general gait detection. It was trained to identify walking periods from wrist-worn accelerometer data. This tuned network achieved an Area Under the Curve (AUC) of 0.97. 

Our second model addressed a more difficult task: detecting gait without other arm activities. This required the model to isolate pure gait from other complex movements. It achieved a n AUC of 0.91. 

Brand YE, Kluge F, Palmerini L, Paraschiv-Ionescu A, Becker C, Cereatti A, Maetzler W, Sharrack B, Vereijken B, Yarnall AJ, Rochester L, Del Din S, Muller A, Buchman AS, Hausdorff JM, Perlman O. Automated Gait Detection in Older Adults during Daily-Living using Self-Supervised Learning of Wrist-Worn Accelerometer Data: Development and Validation of ElderNet. Res Sq [Preprint]. 2024 Mar 15:rs.3.rs-4102403. doi: 10.21203/rs.3.rs-4102403/v1. Update in: Sci Rep. 2024 Sep 6;14(1):20854. doi: 10.1038/s41598-024-71491-3. PMID: 38559043; PMCID: PMC10980143.

Brand, Y.E., Kluge, F., Palmerini, L. et al. Self-supervised learning of wrist-worn daily living accelerometer data improves the automated detection of gait in older adults. Sci Rep 14, 20854 (2024). https://doi.org/10.1038/s41598-024-71491-3

Evers LJ, Raykov YP, Krijthe JH, Silva de Lima AL, Badawy R, Claes K, Heskes TM, Little MA, Meinders MJ, Bloem BR. Real-Life Gait Performance as a Digital Biomarker for Motor Fluctuations: The Parkinson@Home Validation Study. J Med Internet Res. 2020 Oct 9;22(10):e19068. doi: 10.2196/19068. PMID: 33034562; PMCID: PMC7584982 \\

Post E, Laarhoven TV, Raykov YP, Little MA, Nonnekes J, Heskes TM, Bloem BR, Evers LJW. Quantifying arm swing in Parkinson's disease: a method accounting for arm activities during free-living gait. J Neuroeng Rehabil. 2025 Feb 26;22(1):37. doi: 10.1186/s12984-025-01578-z. PMID: 40011957; PMCID: PMC11863854.
