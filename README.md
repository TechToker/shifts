![img](https://github.com/yandex-research/uncertainty-challenge/blob/main/Logoshifts_full_white.png)

# Shifts Challenge

This repository contains data readers and examples for the different datasets provided by the [Shifts Project](https://shifts.ai). 

The Shifts Dataset contains curated and labeled examples of real, 'in-the-wild' distributional shifts across three large-scale tasks. Specifically, it contains white matter multiple sclerosis lesions segmentation and vessel power estimation tasks' data currently used in [Shifts Challenge 2022](https://shifts.grand-challenge.org/), as well as tabular weather prediction, machine translation, and vehicle motion prediction tasks' data used in Shifts Challenge 2021. Dataset shift is ubiquitous in all of these tasks and modalities. 

The dataset, assessment metrics and benchmark results are detailed in our associated papers:

* [Shifts 2.0: Extending The Dataset of Real Distributional Shifts](https://arxiv.org/pdf/2206.15407) (2022)
* [Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks](https://arxiv.org/pdf/2107.07455.pdf) (2021)


If you use Shifts datasets in your work, please cite our papers using the following Bibtex:
```
@misc{https://doi.org/10.48550/arxiv.2206.15407,
  author = {Malinin, Andrey and Athanasopoulos, Andreas and Barakovic, Muhamed and Cuadra, Meritxell Bach and Gales, Mark J. F. and Granziera, Cristina and Graziani, Mara and Kartashev, Nikolay and Kyriakopoulos, Konstantinos and Lu, Po-Jui and Molchanova, Nataliia and Nikitakis, Antonis and Raina, Vatsal and La Rosa, Francesco and Sivena, Eli and Tsarsitalidis, Vasileios and Tsompopoulou, Efi and Volf, Elena},
  title = {Shifts 2.0: Extending The Dataset of Real Distributional Shifts},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2206.15407}
  url = {https://arxiv.org/abs/2206.15407}
}
```
```
@article{shifts2021,
  author    = {Malinin, Andrey and Band, Neil and Ganshin, Alexander, and Chesnokov, German and Gal, Yarin, and Gales, Mark J. F. and Noskov, Alexey and Ploskonosov, Andrey and Prokhorenkova, Liudmila and Provilkov, Ivan and Raina, Vatsal and Raina, Vyas and Roginskiy, Denis and Shmatova, Mariya and Tigar, Panos and Yangel, Boris},
  title     = {Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks},
  journal   =  {arXiv preprint arXiv:2107.07455},
  year      = {2021},
}
```

If you have any questions about the Shifts Dataset, the paper or the benchmarks, please contact `am969@yandex-team.ru` . 


# Dataset Download And Licenses

## License
The Shifts datasets are released under different license.

### White matter multiple sclerosis lesions segmentation

Data is distributed under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. Data can be downloaded after signing [OFSEP](https://www.ofsep.org/fr/) data usage agreement.

### Motion Prediction
  
Shifts SDC Motion Prediction Dataset is released under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.

## Download links

**By downloading the Shifts Dataset, you automatically agree to the licenses described above.**

### Motion Prediction

Canonical parition of the training and development data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/canonical-trn-dev-data.tar). The canonical parition of the evaluation data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/canonical-eval-data.tar). The full, unpartitioned dataset is available [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/full-unpartitioned-data.tar). Baseline models can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/baseline-models.tar).





