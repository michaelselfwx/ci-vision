Model Versions README

v1 (ex. unet_v1.2, dino_v1.5):

Used "sampled" frames for the training set, "unsampled" for val/test

Selected the images in the val/test by random ("rand") assignment

- Split-type=rand, label-usage=train_samp

v2 (ex. unet_v2.3, dino_v2.3):

Used ONLY the "sampled" frames for train/val/test, no "unsampled" frames were used at all

Split the images up 70% 15% 15% based on day

- Split-type=day, label-usage=only_samp

v3 (ex. unet_v3.3, dino_v3.3):

Used ONLY the "sampled" frames like in v2

Split the images randomly

- Split-type=rand, label-usage=only_samp

Notes:

dino_v1.5 and dino_v1.5_2 configurations are slightly different, using large dino and base dino, respectively (still yielded better results with the updated script and base dino).

AFTER v3.3 for both model types, they all used updated script (not denoted by \_2 anymore)