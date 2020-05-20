import sys


def read_classified_data(path_to_data):
    data = []
    output_file = open(path_to_data, "r")
    line = output_file.readline()
    
    while line:
        data.append(line)
        line = output_file.readline()
    return data
    
        


def evaluate_data(data_list):
    
    number_of_spam = 0
    number_of_ham = 0
    no_of_spam_classifications = 0
    no_of_correct_spam_classifications = 0

    no_of_ham_classifications = 0
    no_of_correct_ham_classifications = 0
    
    for item in data_list:
        item_list = item.split()
        #print(item_list)
        predicted_label = item_list[0]
        print("predicted", predicted_label, end = " ")
        actual_label = item_list.pop()
        if "ham" in actual_label:
            actual_label = "ham"
            number_of_ham += 1
        elif "spam" in actual_label:
            actual_label = "spam"
            number_of_spam += 1
        else:
            continue
            
        print("; actual",actual_label)
        
        #spam & ham precision:
        if predicted_label == 'spam':
            no_of_spam_classifications += 1
            #if the actual label is also spam
            if actual_label == 'spam':
                no_of_correct_spam_classifications += 1
                
        elif predicted_label == 'ham':
            no_of_ham_classifications += 1
            if actual_label == 'ham':
                no_of_correct_ham_classifications += 1
                
    
    
    
    try:
        spam_precision = no_of_correct_spam_classifications/no_of_spam_classifications
    except ZeroDivisionError:
        spam_precision = 0
    try:
        ham_precision = no_of_correct_ham_classifications/no_of_ham_classifications
    except ZeroDivisionError:
        ham_precision = 0


    #spam and ham recall 
    try:
        spam_recall = no_of_correct_spam_classifications/number_of_spam
    except ZeroDivisionError:
        spam_recall = 0
    try:
        ham_recall = no_of_correct_ham_classifications/number_of_ham
    except:
        ham_recall = 0
        
        
    #F-1 scores
    try:
        spam_f1 = (2*spam_precision*spam_recall)/(spam_precision+spam_recall)
    except ZeroDivisionError:
        spam_f1 = 0
    try:
        ham_f1 = (2*ham_precision*ham_recall)/(ham_precision+ham_recall)
    except ZeroDivisionError:
        ham_f1 = 0


    print("Spam P,R,F1: ", spam_precision, ", ", spam_recall,", ", spam_f1)
    print("Ham P,R,F1: ", ham_precision, ", ", ham_recall,", ", ham_f1)

    

        

def main():
	path_to_data = sys.argv[1]
	data = read_classified_data(path_to_data)
	evaluate_data(data)


if __name__ == "__main__":
	main()