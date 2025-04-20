# CDS Project (Team RPP) - Deepfake News Detector
<br> 

# Code Files 
<li> ImgModelCode : code for the image preprocessing + model</li>
<li> TextModelCode : code for the text preprocessing + model</li>
<li> data visualtion : code for the data visualistion for both datasets</li>
<li> interface : code for the streamlit app</li>
<br> 

# Dataset 
<li> Text Model - Dataset is included in the repository (Fake.csv and True.csv) </li>
<li> Image Model - Dataset is downloaded from kaggle in the first few cells of Image Model. <br> You can also download it from the link here:  https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images</li>

# Model
- `img_model.keras` — Pre-trained CNN model for classifying real vs. deepfake human images.
- `fake_news_model.pkl` — Logistic Regression model for text-based fake news detection.
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer for transforming news articles before classification.

