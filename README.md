# mini-project

Here is the mini project 1 from IPEN 5810, and I finished this project with my teammate Mr. Zihao QU

In this project, we will try to use some mechine learning and NLP technologies to achieve following questions.

* Preprocess the text, such as removing punctuation marks, stop words, etc.
* Use natural language processing techniques to classify the topics in these employee reviews.
* Visualize and interpret the topic classification results.
* Use machine learning tools to rank the correlation between different topics and identify those that are mostly correlated with income/wage/salary.

In the Code file, Our team sperated those questions into four files.
* Q1: We finished preprocess the text.
* Q2: We use the NLP technology to classify the topics in the employee reviews. We choose the review_cons field to anaylize and classify.
* Q3: We try to visualize the topic classification result. In this file, I also try some methods to improve the model performance and accuracy, I will list my work later.
* Q4: We use ML technologies to rank the corrlation and print them out.

All the code and dataset can be found in the github and thanks for the helping from Dr. Luoye CHEN, TAs and ChatGPT.


Q1 is a basic task, we used some libraries like pandas and nltk, as well as specific modules and functions. 
1, Define a fuction which can preprocess the text including the skills like removing the empty values, filters out words starting with "http", "#",etc.
2, Then we applied the fuction to some fields like 'review_summary','review_advice', etc.
3, At last, we save the preprocessed data into a new file.
