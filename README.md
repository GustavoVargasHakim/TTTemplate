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

## Configuration through arguments

The whole configuration for any project can be defined inside the `configuration.py` file. A quite simple parsed arguments system is used, where the different parameters are also organized in categories such as "Directories", "Dataset" or "Model". These are the minimum requirements for the whole TTTemplate pipeline to work, so please avoid changing them. Any new configuration related to your method should be added after the section "Method configuration". 

Please carefully study all the available configurations to familiarize with what you need to change and which arguments you need to specify when running your programs. 

## Model customization

Model customization can be done inside `utils/model_utils.py`. The most important function is `create_model`, which receives the parsed arguments, (optional) pretrained weights, and a boolean argument called `augment` that determines whether to augment/custom a model or not, depending on your needs. Notice that the Timm model is loaded without a classifier, while it, along the final global pooling layer, are manually added after creation.

When customizing a model, you need to take of two main things:

* Model augmentation: adding new components to a Pytorch class-based model is done through a function called `augment_model`. Normally, this function receives the timm model, the name of the dataset, and a series of optional `**kwargs`. NOTE: the `**kwargs` must have been ideally received by the `create_model` function. Please see an example of how to add a new component to a model:

```python
def augment_model(model, **recipe):
    '''
    Augmenting model with additional modules (e.g., Y-shaped architectures, projectors, etc.)
    :param model: original timm model
    :param recipe: auxiliar information to add the new components (e.g. layer indices)
    :return: augmented model
    '''
    #Example
    layers = recipe['layers']
    if layers[1]:
        model.projector1 = nn.Conv2d(128, 128, 1)

    return model
```

You can receive anything you need to augment your model in the form of a `**recipe`. 

* Forward pass augmentation: when adding new components to your model (e.g., second head for a self-supervised task), the forward pass needs to also be modified. In this case, the `types` library is used to override the model's `__forward__` method. If you intend working with small datasets (i.e., CIFAR-10/100-C, 32 x 32 images), you should focus on the function `forward_small`. If you intend working with large datasets (i.e., VisDA-C, OfficeHome, 224 x 224 images), you should focus on the function `forward_large`. Notice that overriding this methods also help returning the feature maps of different layers (very useful in Deep Learning methods) along with the classification logits. 



