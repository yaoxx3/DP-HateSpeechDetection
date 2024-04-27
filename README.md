# Privacy-Preserving Text Analysis for Hate Speech Detection
We utilize Opacus and TensorFlow Privacy libraries to train differentially private hate speech detection models based on huggingface BERTSequenceClassifier. Our models currently focus on binary classification task.
## Target
The project helps you apply differential privacy techniques to a carefully selected social media dataset, which includes state-of-the-art, annotated datasets specific to hate speech. 
By employing a comprehensive corpus of hate speech, we intend to implement privacy-preserving mechanisms to detect such speech while safeguarding the privacy of individuals involved, 
regardless of the nature of their rhetoric.
## Getting Started
### Dependancies
The models utilize [Opacus](https://github.com/pytorch/opacus) and [TensorFlow Privacy](https://github.com/tensorflow/privacy) libraries to achieve Differential Privacy. Installing these libraries is a pre-requisite. 
Detailed instructions can be found in the main page of the libraries.
### Data Source
[Hate Speech Dataset from a White Supremacy Forum](https://github.com/Vicomtech/hate-speech-dataset), the dataset being utilized in the research project on Privacy-Preserving Text Analysis for Hate Speech Detection, 
is sourced from a white supremacist forum known as Stormfront. 
This dataset consists of hate speech annotated on Internet forum posts in English at the sentence level. 
Specifically, it comprises 10,568 sentences extracted from Stormfront and classified as conveying hate speech.

You can safely utilize our model training workflow for another hate speech dataset.
### Experimental: Data Undersampling and Oversampling
EDA reveals a significant class imbalance within this dataset, as well as most of [hate speech dataset online](https://hatespeechdata.com/). In order to mitigate the bias caused by the skewness 
of dataset, we also provide [data undersampling](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) and [oversampling](https://github.com/makcedward/nlpaug) 
techniques. You can comment them out if it's not needed.
### Model training
We utilize [BERTForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification) as the base model. To save computational resources, we currently freeze most of the layers and only fine tune a few upper layers (BertPooler and Classifier).
You can comment out the freezing code snippet if you want to fine tune the full BERT model.
## Future work
Further enhancements to our project could involve the adoption of more sophisticated differential privacy techniques such as personalized privacy, 
where individual user settings could dictate the level of privacy applied to their data. 
Additionally, incorporating other forms of privacy-preserving data transformations or employing advanced neural network architectures that are inherently more robust to
noise may yield better performance.

