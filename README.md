# Building-Roof-Inspection-System
To ensure the safety and reliability of roof inspections, it is imperative to implement comprehensive measures and protocols that prioritize the well-being of personnel and adhere to industry standards and best practices, thereby fostering a secure and dependable environment for conducting thorough examinations of roofing structures. The approach is based on the UNet network with transfer learning on the two popular architectures: VGG16 and Resnet101. The result shows that a large crack segmentation dataset helps improve the performance of the model in diverse cases that could happen in practice.


# Overview
An essential step in solving structural investigation issues is inspecting buildings and roofs. As part of the house inquiry project, for instance, a drone is programmed to fly throughout the house and take images of various surfaces. After that, a computer will analyze the images to identify any possible damage areas on the outside of the home. The less human labor required to process these photos, the more accurate the model is. If not, the operators will be forced to verify each and every image, which is tedious and prone to mistakes. The model's sensitivity to noise and other things, like title lines, etc., presents a problem in this work. 

# Dependencies
Create conda environment from yaml file:
...

or explicity using:
```python
conda create --name crackseg_env python=3
conda activate crackseg_env
conda install -c anaconda pytorch-gpu 
conda install -c conda-forge opencv 
conda install matplotlib scipy numpy tqdm pillow
```

# Inference
- download the pre-trained model [unet_vgg16](https://drive.google.com/open?id=1wA2eAsyFZArG3Zc9OaKvnBuxSAPyDl08) or 
[unet_resnet_101]().
- put the downloaded model under the folder ./models
- run the code
```pythonstub
python inference_unet.py  -img_dir ./test_images -model_path ./models/model_vgg_best.pt -model_type vgg16 -out_pred_dir ./test_result
```

# Citation
https://github.com/khanhha/crack_segmentation
