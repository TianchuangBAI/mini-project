# mini-project

Here is the mini project 1 from IPEN 5810, and I finished this project with my teammate Mr. Zihao QU

In this project, we will try to use some mechine learning and NLP technologies to achieve following questions.

* Preprocess the text, such as removing punctuation marks, stop words, etc.
* Use natural language processing techniques to classify the topics in these employee reviews.
* Visualize and interpret the topic classification results.
* Use machine learning tools to rank the correlation between different topics and identify those that are mostly correlated with income/wage/salary.

In the Code file, Our team sperated those questions into four files.
* [Q1](https://github.com/TianchuangBAI/mini-project/blob/72617f78ce8f539e58f2a6dc60cbbc218b3f8ac7/code/Q1.py): We finished preprocess the text.
* [Q2](https://github.com/TianchuangBAI/mini-project/blob/72617f78ce8f539e58f2a6dc60cbbc218b3f8ac7/code/Q2.py): We use the NLP technology to classify the topics in the employee reviews. We choose the review_cons field to anaylize and classify.
* [Q3](https://github.com/TianchuangBAI/mini-project/blob/72617f78ce8f539e58f2a6dc60cbbc218b3f8ac7/code/Q3.py): We try to visualize the topic classification result. In this file, I also try some methods to improve the model performance and accuracy, I will list my work later.
* [Q4](https://github.com/TianchuangBAI/mini-project/blob/72617f78ce8f539e58f2a6dc60cbbc218b3f8ac7/code/Q4.py): We use ML technologies to rank the corrlation and print them out.

All the code and dataset can be found in the github and thanks for the helping from Dr. Luoye CHEN, TAs and ChatGPT.


Q1 is a basic task, we used some libraries like pandas and nltk, as well as specific modules and functions. 
1. Define a fuction which can preprocess the text including the skills like removing the empty values, filters out words starting with "http", "#", etc.
2. Then we applied the fuction to some fields like 'review_summary','review_advice', etc.
3. At last, we save the preprocessed data into a new file.


Q2 is a NLP task, we use pandas, nltk and scikit-learn for this task.
1. We preprocessed the data like Q1 and use PorterStemmer from nltk to process the word stemming.
2. Then we did the feature extraction and text classification. I utilized the TfidfVectorizer from scikit-learn to convert the text data into a matrix of TF-IDF features and splited the data into training and testing sets using the train_test_split function. we use a Multinomial Naive Bayes classifier to train a model on the training data and make predictions on the test data.
3. We evaluate our model by using the accuracy_score fuction and print the result and report.

Here is the result of 1st Version.
![result](https://github.com/TianchuangBAI/mini-project/blob/ce5228d9fb706454e05fa9267fc573eccf5f4dc0/pics/Q2.jpg)

Q3 we have two tasks, the first one is visualize and interpret the topic classification result. The second one is improve the accuracy of the topic detction.
![result]()
