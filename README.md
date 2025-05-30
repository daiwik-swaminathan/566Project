# Version 2 Documentation

## Audio Scraping

1. Download the ffmpeg from https://ffmpeg.org/download.html
2. In utils.ffmpeg.py change the ffmpeg path to where you downloaded ffmpeg - for me it looks like:
   ```
   self.ffmpeg_path = r"C:\Users\joelp\OneDrive - Cal Poly\CalPoly\CSC 566\ffmpeg\ffmpeg\bin"
   ```
3. From the repo home directory run
   ```
   python -m dataset_scraping.audio_scraping
   ```
4. Go to dataset_scraping.audio_scraping.py and change your name and go to the save_clip function. Change your name to be unique so we don't overwrite other folders. Then create a folder structure in dataset_scraping
   ![alt text](image-1.png)
5. A Tkinter GUI will pop up with a button to save a non-highlight or highlight. This should be recording computer audio not your mic. (Tested on Windows not Mac yet)
   ![alt text](image.png)
6. Watch sports and record what you think is a highlight. It should save the clip to the respective folder.
7. Test the audio to see if its being saved correctly. If so we can repeat and hopefully get a decently sized dataset.

## Model Prediction

- To run training

```
python -m model_v2.model
```

### Model.py

- Uses a CNN with a couple more layer and no PCA downsampling
- Results are pretty unpredictable, we can test with more data

### Model2.py

- Attempts to use a variational autoenconder with classifier loss to help push the diff class latents away
- The idea is that we train a good encoder to encode the audio and run a classifier on the latents
- By training with the classifier it helps push the latents away but also can lead to overfitting
- We can visualize the change on Daiwiks training set
  ![alt text](image-2.png)
- Hopefully with more data it becomes more robust and we can test the impact between classifcation loss and reconstruction loss

# The predictor

- We use ffmpeg to record computer audio output in 10 sec segments and live feed it to the model
- If the probability is past a threshold we classify it as a highlight or not and display it
- Right now its really finicky - it thinks no audio could be a highlight - but the hope is that with some data collected in the same way that we will use it, it will become more robust.
- Some problems might be highlight delay (we are taking the last 10 seconds) and too many videos (rn its doing it every second) and probabilites tend to be close to 0.5

- example:
  ![alt text](image-3.png)
