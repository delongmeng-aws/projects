## Statistical and Machine Learning Projects




[Implementation of logistic regression with neural network mindset](https://github.com/delongmeng/projects/tree/master/Implementation_Logistic_Regression_with_Neural_Network_mindset)
- build a general learning architecture including parameter initialization, loss and gradient calculation, and optimization from scratch
- packages: Python numpy, scipy, PIL, h5py
- keywords: logistic regression, image classification, computer vision


[Implementation of neural network with one hidden layer](https://github.com/delongmeng/projects#:~:text=Implementation_Neural_Network_One_Hidden_Layer)
- build a shallow neural network architecture
- implement the parameter initialization, forward and backward propagation, model training and prediction, hyperparameter tuning modules 
- packages: Python numpy, sklearn
- keywords: neural network, classification


[Implementation of deep neural network from scratch](https://github.com/delongmeng/projects/tree/master/Implementation_Deep_Neural_Network)
- implement the forward propagation, loss calculation, and backward propagation modules of a fully connected neural network from scratch
- build a multi-layer model and apply to an image classification task
- packages: Python numpy, PIL, scipy, h5py
- keywords: DNN, image classification, computer vision


[Neural network modeling for SIGNS dataset using TensorFlow](https://github.com/delongmeng/projects/tree/master/NN_SIGNS_TensorFlow)
- build a neural network using Tensorflow
- apply this model to the SIGNS dataset to recognize numbers from hand signs
- packages: Python tensorflow, h5py
- keywords: image classification, computer vision


[Implementation of Convolutional Neural Networks (CNN) from scratch](https://github.com/delongmeng/projects/tree/master/Implementation_CNN)
- implement convolutional and pooling layers for forward propagation of Convolutional Neural Networks (CNN) from scratch
- implement a TensorFlow CNN model and apply to the SIGNS dataset to recognize numbers from hand signs
- packages: Python numpy, tensorflow, PIL, h5py
- keywords: CNN, image classification, computer vision


[Implementation of face recognition models](https://github.com/delongmeng/projects/tree/master/Implementation_Face_Recognition)
- map face images into encodings using a pretained Inception model
- implement the triplet loss function among the positive, negative, and anchor images
- build models for face verification and face recognition 
- packages: Python keras, tensorflow
- keywords: face recognition, computer vision


[Implementation of residual networks (ResNets) and application on SIGNS dataset](https://github.com/delongmeng/projects/tree/master/Implementation_ResNets_SIGNS)
- build a 50-layer convolutional networks using residual networks (ResNets)
- implement the identity block and convolutional block of a ResNet
- apply this ResNet50 model on the SIGNS dataset of hand signs
- packages: Python keras, tensorflow
- keywords: CNN, ResNet, computer vision


[Implementation of Art generation by neural style transfer](https://github.com/delongmeng/projects/tree/master/Implementation_Art_Generation_Neural_Style_Transfer)
- implement the neural style transfer algorithm and generate novel artistic images using this algorithm
- neural style transfer merges a "content" image and a "style" image to create a "generated" image
- build the TensorFlow graph based on the pre-trained VGG-19 model
- build content cost function and style cost function and combine them to get the total cost
- packages: Python tensorflow, PIL, scipy
- keywords: generative AI, computer vision


[Implementation of YOLO for car detection and autonomous driving](https://github.com/delongmeng/projects/tree/master/Implementation_YOLO_Car_detection_Autonomous_Driving)
- implement the You Only Look Once (YOLO) algorithm for object detection
- output filtering using non-max suppression (NMS) 
- packages: Python numpy, tensorflow, keras, yad2k, PIL, scipy
- keywords: CNN, computer vision, object detection, autonomous driving


[Word vector representation using GloVe](https://github.com/delongmeng/projects/tree/master/Word_Vector_Representation_GloVe)
- similarity measurement between words based on the pre-trained GloVe vectors
- implement word analogy prediction, debiasing (neutralization and equalization) algorithms
- packages: Python numpy
- keywords: NLP, word embedding


[Build an Emojifier using word vector representation and LSTM](https://github.com/delongmeng/projects/tree/master/Emojify_word_vector_LSTM)
- convert words to word vector representations using pre-trained GloVe embeddings
- further feed word embeddings into an LSTM to account for word ordering
- packages: Python emoji, keras
- keywords: NLP, word embedding


[Implementation of Recurrent Neural Networks (RNN) from scratch](https://github.com/delongmeng/projects/tree/master/Implementation_RNN_from_scratch)
- implement the forward and backward pass for a basic Recurrent Neural Network (RNN) model from scratch
- implement the forward and backward pass a Long Short-Term Memory (LSTM) model from scratch
- packages: Python numpy
- keywords: NLP, RNN, LSTM


[Neural machine translation using LSTM with attention](https://github.com/delongmeng/projects/tree/master/Neural_Machine_Translation_LSTM_attention)
- build a neural machine translation (NMT) model to translate human-readable dates into machine-readable dates
- implement the attention step, and the pre-attention and post-attention LSTMs
- visualization of the attention weights
- packages: Python tensorflow
- keywords: NLP, LSTM, attention, machine translation


[RNN LSTM music generation](https://github.com/delongmeng/projects/tree/master/RNN_LSTM_music_generation_Jazz_improvisation)
- build a model to learn musical patterns and then use this model to generate music similar to the given jazz music
- packages: Python keras
- keywords: RNN, LSTM, music generation, generative AI


[RNN LSTM text generation character level language model](https://github.com/delongmeng/projects/tree/master/RNN_LSTM_text_generation_Character_level_language_model)
- build a character-level text generation RNN model to synthesize new text
- implement the gradient clipping and sampling techniques
- generate dinosaur names according to a training set of actual dinosaur names using a RNN model
- generate Shakespeare poems after training on a collection of Shakespearian poems using a LSTM model
- packages: Python numpy, keras
- keywords: NLP, RNN, LSTM, text generation, generative AI


[Trigger word detection for speech recognition](https://github.com/delongmeng/projects/tree/master/Trigger_word_detection_speech_recognition)
- build a audio spectrograms dataset for speech recognition using data synthesis technique
- build a speech recognition architecture including convolutional, GRU, and dense layers
- packages: Python pydub, keras
- keywords: speech recognition, data synthesis


[A simulation study of statistical power in linear regression](http://htmlpreview.github.io/?https://github.com/delongmeng/projects/blob/master/power_analysis_simulation.html)  
- perform a simulation study to demonstrate the factors that affect statistical power in linear regression
- specifically, investigate the effects of sample size, signal strength, and noise level
- visualization and explanation of how power is affected by these factors
- packages: R scales
- keywords: statistical power analysis, linear regression


[Linear regression prediction of breast cancer patient overall survival: modeling building and assumption testing](http://htmlpreview.github.io/?https://github.com/delongmeng/projects/blob/master/breast_cancer_survival_multi_linear.html) 
- build a linear regresion model for survival prediction using various model building techniques such as AIC model search, Box-Cox response transformation
- collinearity checking, and diagnostics of model assumptions including residual analysis, Q-Q plot, Breusch-Pagan test, and Shapiro-Wilk normality test
- Kaplan-Meier survival probability and Cox proportional hazards regression
- packages: R tidyverse, survival, lm, lmtest
- keywords: survival analysis, linear regression


[Predict the housing prices in Ames: linear regression models and tree-based regression models](https://github.com/delongmeng/projects/blob/master/housing_price_regression_report.pdf)  
- build a linear regression model and a tree-based model for housing price prediction
- compare different regularization methods for the linear regression model: L1-norm penalty (Lasso), L2-norm penalty (Ridge) or a combination of L1-norm and L2-norm penalty (Elastic Net)
- data pre-processing techniques such as one-hot encoding and Winsorization
- packages: R glmnet, xgboost
- keywords: regularization, Elastic Net, Gradient Boosting Machine (GBM)


[Walmart Store Sales Forecasting: using SVD in linear regression](https://github.com/delongmeng/projects/blob/master/sales_forecasting_SVD_linear_regression_report.pdf)  
- train a linear regression model to predict sales based on historical weekly sales data
- use singular value decomposition (SVD) for feature extraction
- package: R lm
- keywords: sales forecasting, SVD


[Movie Review Sentiment Analysis: bag-of-words, n-gram, and logistic regression](https://github.com/delongmeng/projects/blob/master/review_sentiment_NLP_report.pdf)  
- construct the vocabulary based on "bag-of-words" and "n-gram" strategy, build the document-term matrix (DTM), and use a logistic regression model with Lasso regularization to trim the vocabulary
- process the input movie review text to embed the text into matrix
- train a cross-validation logistic regression model with Ridge regularization to predict the sentiment
- package: R text2vec, glmnet, proc
- keywords: NLP, word embedding, sentiment analysis


[Clinical risk modeling](https://github.com/delongmeng/projects/tree/master/Clinical_Risk_modeling)
- logistic regression based risk modeling for diabetic retinopathy
- decision tree and random forests for risk modeling
- risk model evaluation using C-index
- hyperparameter grid search
- missing data processing: imputation
- model interpertation: SHAP analysis
- packages: Python sklearn, shap
- keywords: healthcare


[Clinical survival analysis](https://github.com/delongmeng/projects/tree/master/Clinical_Survival_analysis)
- Kaplan-Meier analysis and Log-Rank test
- Cox Proportional Hazards modeling
- Harrell's C-index for Cox model evaluation
- random survival forests
- VIMP score for model interpertation
- packages: Python lifelines, r2py, R randomForestSRC
- keywords: healthcare


[Clinical treatment effect analysis](https://github.com/delongmeng/projects/tree/master/Clinical_Treatment_Effect_analysis)
- evaluate the results of randomized control trial (RCT)
- T-learner: learning treatment risk and control risk
- evaluation metric: C-statistic-for-benefit
- packages: Python lifelines
- keywords: healthcare




*Note that html files can be viewed by adding `http://htmlpreview.github.io/?` to the front of the url. For example:*

*Power analysis simulation:  
http://htmlpreview.github.io/?https://github.com/delongmeng/projects/blob/master/power_analysis_simulation.html*

