# ham-spam
Naive Bayes Classifier written in Python to classify emails as ham or spam.

There are 3 files: 
- nblearn uses the training data to learn the Naive Bayes model and saves the model into a file called nbmodel.
- nbclassify uses the model that was learnt to classify the data in the dev set, and writes it into a file nboutput.
- nbevaluate reads the predicted data and evaluates it, and outputs the precision, recall, accuracy and f1 scores. 

The other two folders contain the same structure: one uses only 10% of the training data to see if it makes a huge difference, and the other performs some feature modifications by preprocessing the text before training it. 

The report.txt contains the detailed evaluations of each of the methods. 
