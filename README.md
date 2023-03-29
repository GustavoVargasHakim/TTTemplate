# TTTemplate
Pipeline template for Test-Time Training (TTT) and Test-Time-Adaptation (TTA). Fully customizable code based on PyTorch, featuring the following characteristics:

1. Based on Timm models for standardization and ease of use. 
2. It automates the execution on multiple-GPUs using DistributedDataParallel.
3. Configurations are already done to easily work on popular datasets such as: CIFAR-10/100-C, CIFAR-10.1, OfficeHome and VisDA-C.
4. Highly customizable, only focusing on modifying very specific functions that are inserted in the standard pipeline.
5. Includes the following functionalities: source training, joint training, adaptation to a single dataset/corruption, multiple-corruption adaptation (for CIFAR-10/100-C).
