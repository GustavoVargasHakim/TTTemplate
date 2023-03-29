# TTTemplate
Pipeline template for Test-Time Training (TTT) and Test-Time-Adaptation (TTA). Fully customizable code based on PyTorch, featuring the following characteristics:

1. Based on Timm models for standardization and ease of use. 
2. It automates the execution on multiple-GPUs using DistributedDataParallel.
3. Configurations are already done to easily work on popular datasets such as: CIFAR-10/100-C, CIFAR-10.1, OfficeHome and VisDA-C.
4. Highly customizable, only focusing on modifying very specific functions that are inserted in the standard pipeline.
5. Includes the following functionalities: source training, joint training, adaptation to a single dataset/corruption, multiple-corruption adaptation (for CIFAR-10/100-C).

## Instructions of usage

The TTTemplate's main files are the following:

* `source_training.py`: performs cross-entropy training on a source dataset (e.g. CIFAR-10), without including any secondary task (Test-Time Training). This code is suitable to measure baseline accuracy and for Test-Time Adaptation methods. Notice that for VisDA-C and OfficeHome datasets, the pretrained weights of ImageNet are used.
* `joint_training.py`: performs a joint training task, normally combining cross-entropy loss for classification, and a secondary task. The pipeline is similar to that for source training, but requires modifying some other files in order to add an auxiliary task (see more details below).
* `test.py`: evaluates the model on the testing data (i.e. target domain), without performing any type of adaptation. Use this to measure baselines.
* `test_adapt.py`: adapts the model according to the custom TTT/TTA method. Similar pipeline as in testing, but requires slight modification in other files, similarly as for joint training if an auxiliary task is used.
