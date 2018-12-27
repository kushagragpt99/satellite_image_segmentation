# satellite_image_segmentation
multi class segmentation of satellite images. Part of the the competition 'Eye in the Sky' in Inter IIT Tech meet 2018
# Dataset
the dataset used in the repository is available at https://drive.google.com/drive/folders/1a8rFjqAc7fHBjpzCgnLwdI1lvjhinjSA
# Repo description
1. model.h5 and model.json contain weights and structures for already trained models.
2. The script unet.py can be used to train the model again. 
3. The script model.py can be used to measure the accuracy and to segment new satellite images. 
4. model.png contains the structure of the UNet used.
5. Some of the predictions can be viewed in the predicted_images folder. gndt are the ground truth (given) images and pred are the predicted images.
