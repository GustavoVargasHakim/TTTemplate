# TTTemplate
Pipeline template for Test-Time Training (TTT) and Test-Time-Adaptation (TTA). Fully customizable code based on PyTorch, featuring the following characteristics:

1. Based on Timm models for standardization and ease of use. 
2. It automates the execution on multiple-GPUs using DistributedDataParallel.
3. Configurations are already done to easily work on popular datasets such as: CIFAR-10/100-C, CIFAR-10.1, OfficeHome and VisDA-C.
4. Highly customizable, only focusing on modifying very specific functions that are inserted in the standard pipeline.
5. Includes the following functionalities: source training, joint training, adaptation to a single dataset/corruption, multiple-corruption adaptation (for CIFAR-10/100-C).

## How is the TTTemplate organized?

The TTTemplate's main files are the following:

* `source_training.py`: performs cross-entropy training on a source dataset (e.g. CIFAR-10), without including any secondary task (Test-Time Training). This code is suitable to measure baseline accuracy and for Test-Time Adaptation methods. Notice that for VisDA-C and OfficeHome datasets, the pretrained weights of ImageNet are used.
* `joint_training.py`: performs a joint training task, normally combining cross-entropy loss for classification, and a secondary task. The pipeline is similar to that for source training, but requires modifying some other files in order to add an auxiliary task (see more details below).
* `test.py`: evaluates the model on the testing data (i.e. target domain), without performing any type of adaptation. Use this to measure baselines.
* `test_adapt.py`: adapts the model according to the custom TTT/TTA method. Similar pipeline as in testing, but requires slight modification in other files, similarly as for joint training if an auxiliary task is used.

The main goal of the TTTemplate is to have a strong code baseline and starting point for any TTT/TTA research project. To do this, we need to minimize the amount of code that needs to be modified, as well as the amount of new code to be written. The previously mentioned files would require minimal modifications, mostly related to root names, adding parsed arguments, or passing additional parameters to functions.

Each project would require different tools and ingredients that are very specialized. However, most of these tools can be added on top of a general skeleton. In the TTTemplate, these tools can mostly been added inside the different files that compose the `utils` folder:

* `dist_utils.py`: utils related to distributed processes, and mostly related to displaying messages and results.
* `utils.py`: a generalization of distributed tools, but used when only a single GPU is used. 
* `model_utils.py`: tools for model building and customization. They allow creating standard models from Timm, as well as adding new components depending on the project. We ensure that each model is able to return both classification logits and features. 
* `train_utils.py`: tools for the source and join training processes. The whole sub-pipeline of loading batches, forward and backward passes, and metric computation can all be grouped into more compact functions that can be called every in the main files. They also allow for customization of specific forward passes, as well as creating custom loss functions. 
* `test_utils.py`: tools for target evaluation and adaptation, which automate the process of evaluating a loss function on a batch, as well as adapting to it according to custom loss functions. 

The folder `dataset` contains some additional utils related to dataset loading. They completely set up to work with the most common datasets in TTT/TTA. In general, they would require minimal modification, basing on the state-of-the-art as of 2023.

## Source Training


