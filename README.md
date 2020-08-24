# A Naive Bayes Classifier that detects Ham and Spam emails
Naive Bayes Classifier written in Python to classify emails as ham or spam.

There are 3 files: 
* nblearn uses the training data to learn the Naive Bayes model and saves the model into a file called nbmodel. 
Use `python3 nblearn.py /path/to/input` to run it.

* nbclassify uses the model that was learnt to classify the data in the dev set, and writes it into a file nboutput. Use `python3 nbclassify.py /path/to/input` to run it.

* nbevaluate reads the predicted data and evaluates it, and outputs the precision, recall, accuracy and f1 scores. 
Use `python3 nbevaluate.py nboutput_filename` to run it.

The other two folders are used to experiment with some tweaks. They contain the same structure. 
* One uses only 10% of the training data to see if it makes a huge difference 
* The other performs some feature modifications by preprocessing the text before training it

The report.txt contains the detailed evaluations of each of the methods. 
