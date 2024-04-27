# Privacy-Preserving Text Analysis for Hate Speech Detection
We utilize differential privacy ML Libraries to train differentially private hate speech detection models based on huggingface [BERTForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification). Our models currently focus on binary classification task. We first implement a non-private baseline model with the BERT model, deploy [Opacus](https://github.com/pytorch/opacus) and [Tensorflow Privacy](https://github.com/tensorflow/privacy), and then further apply DP algorithm by leveraging [Differentially Private Stochastic Gradient Descent (DP-SGD)](https://arxiv.org/abs/1607.00133).
## Target
This project aims to delve into a comprehensive exploration of differential privacy methodologies, specifically focusing on the [Opacus](https://github.com/pytorch/opacus) framework and [TensorFlow privacy](https://github.com/tensorflow/privacy), with the overarching goal of scrutinizing online hate speech prevalent in social media datasets. In an era marked by heightened concerns over data privacy infringements by social media platforms, governmental bodies, and third-party applications, the adoption of such techniques becomes imperative to uphold individual privacy rights while addressing pertinent issues. These methodologies play a pivotal role in identifying potential threats of violence adeptly, all the while ensuring the preservation of individual privacy and civil liberties. This significance is underscored by the pivotal role of social media platforms in facilitating communication, organization, and potentially harmful activities.

The relevance of this project extends to its potential utility for law enforcement and security personnel. The insights gleaned from our analysis hold promise in assisting authorities in making informed decisions, enabling them to identify potential risks and implement timely mitigative measures effectively.
## Getting Started
### Dependancies
We utilize [Opacus](https://github.com/pytorch/opacus) and [TensorFlow Privacy](https://github.com/tensorflow/privacy) libraries to achieve Differential Privacy. Installing these libraries is a pre-requisite. 
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
### Training
We utilize [BERTForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification) as the base model. To save computational resources, we currently freeze most of the layers and only fine tune a few upper layers (BertPooler and Classifier).
You can comment out the freezing code snippet if you want to fine tune the full BERT model.
## Future work
Further enhancements to our project could involve the adoption of more sophisticated differential privacy techniques such as personalized privacy, 
where individual user settings could dictate the level of privacy applied to their data. 
Additionally, incorporating other forms of privacy-preserving data transformations or employing advanced neural network architectures that are inherently more robust to
noise may yield better performance.

Future work that would significantly contribute to this domain includes the expansion of differentially private models to detect nuances within hate speech across different languages and cultures, as well as the integration of user feedback loops to improve model sensitivity to context and mitigate biases. Moreover, the development of a privacy-preserving model monitoring framework that could continuously assess and manage the trade-offs between privacy protection and model utility in a production environment would be a noteworthy progression. Implementing federated learning approaches, where training can be performed on decentralized data, might also enhance privacy by design and could be a pivotal step towards a new standard in privacy-preserving text analysis.
## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/CindyKan"><img src="https://avatars.githubusercontent.com/u/17608784?v=4" width="100px;" alt=""/><br /><sub><b>Cindy Kan</b></sub></a><br /></td>
  </tr>
</table>

