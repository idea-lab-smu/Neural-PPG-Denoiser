# Neural PPG Denoiser (NPD) replication

### TTSR

Paper

Yang, F.; Yang, H.; Fu, J.; Lu, H.; Guo, B. Learning Texture Transformer Network for Image Super-resolution. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition; IEEE: Piscataway, NJ, USA, 2020; pp. 5791–5800. [CrossRef](https://doi.org/10.48550/arXiv.2006.04139)

Code

Yang, F.; Yang, H. TTSR, (2020), Github repository, https://github.com/researchmm/TTSR

### Dataset

Siam, A.; Abd El-Samie, F.; Abu Elazm, A.; El-Bahnasawy, N.; Elbanby, G. Real-World PPG Dataset, Version 1; Mendeley Data;
Mendeley: London, UK, 2019. [CrossRef](http://doi.org/10.17632/yynb8t9x3d.1)

Pimentel, M.A.; Johnson, A.E.; Charlton, P.H.; Birrenkott, D.; Watkinson, P.J.; Tarassenko, L.; Clifton, D.A. Toward a Robust
Estimation of Respiratory Rate from Pulse Oximeters. IEEE Trans. Biomed. Eng. 2016, 64, 1914–1923. [CrossRef](http://doi.org/10.1109/TBME.2016.2613124) 

Delaram Jarchi and Alexander J. Casson. Description of a Database Containing Wrist PPG Signals Recorded during Physical Exercise with Both Accelerometer and Gyroscope Measures of Motion. Data 2017, 2(1), 1. [CrossRef](http://doi.org/10.3390/data2010001)

Reiss, A.; Indlekofer, I.; Schmidt, P.; Van Laerhoven, K. Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural
Networks. Sensors 2019, 19, 3079. [CrossRef](https://doi.org/10.3390/s19143079)

## Original Neural PPG Denoiser (NPD)

Kwon, J.H.; Kim, S.E.; Kim, N.H.; Lee, E.C.; Lee, J.H. Preeminently Robust Neural PPG Denoiser. Sensors 2022, 22, 2082. https://doi.org/10.3390/s22062082

Github
https://github.com/juhuk98/Neural-PPG-Denoiser.git

## Quick test
1. Clone this github repo

2. Download models from [GoogleDrive](https://drive.google.com/drive/folders/1IwWNDqcMlRnFNLt6Blxzs-O8i8LHvafT).

3. Add to the following path `./model`

4. Run test
```
sh test.sh
```
5. The results are in (`./test/demo/output`)

## Evaluation
1. run evalutation
```
sh eval.sh
```

## Train
1. run train

```
sh train.sh
```

2. The results are in (‘./train/CUFED/TTSR/model/’ )


