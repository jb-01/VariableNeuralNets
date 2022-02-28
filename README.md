# VariableNeuralNets - attention-based parameter swapping for continual learning
official submission for the 2022 AWS Deep Learning Challenge

Brain Tumor   |  Nuclei |  Colonoscopy
:-------------------------:|:-------------------------: |:-------------------------:
![](https://github.com/jb-01/VariableNeuralNets/blob/main/images/merged_image.jpg?raw=true)  |  ![](https://github.com/jb-01/VariableNeuralNets/blob/main/images/merged_image%201.jpg?raw=true)  |  ![](https://github.com/jb-01/VariableNeuralNets/blob/main/images/merged_image%202.jpg?raw=true)
This project explores the use of 

## Set-up Steps 

0. Set up DL1 Instance / Choose [Habana® Deep Learning Base AMI (Ubuntu 20.04)](https://aws.amazon.com/marketplace/pp/prodview-fw46rwuxrtfse?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) image

1. Select a TensorFlow framework image to run on the above Habana image:
	
> docker pull public.ecr.aws/habanalabs/tensorflow-installer:2.8.0-ubuntu20.04-1.3.0-latest

Run the docker image:

>docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host public.ecr.aws/habanalabs/tensorflow-installer:2.8.0-ubuntu20.04-1.3.0-latest


## Run the model
Create a working directory for python scripts on EC2 and execute: 
- ```trainUNet.py``` to train the MainNet (UNet) 
- ```AttentionNet.py``` to train the AttentionNet (CNN)
- ```Mechanics.py``` with given input image to test entire model on given task for either brain tumor, colonoscopy, or cell nuclei segmentation

## DL1 / Efficiency Notes
The above models rely on Habana modules to train using AWS DL1 instances. Training across multiple devices results in ~10 epochs under 60 seconds on the AttentionNet (CNN) and under 5 minutes for the MainNet (UNet) using the corresponding datasets referenced below. Both models converge to > 0.85 accuracy.
- For documentation on TensorFlow migration and training see [Habana Documentation](https://docs.habana.ai/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html# )

### Datasets
**Nuclei Segmentation:**
https://www.kaggle.com/c/data-science-bowl-2018/overview

**Brain Tumor:**
https://www.med.upenn.edu/cbica/brats2020/data.html

**Colonoscopy:**
https://paperswithcode.com/dataset/kvasir

---
## Submission
https://docs.habana.ai/en/latest/index.html
