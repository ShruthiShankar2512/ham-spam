import sys 
import os 
import string
import json

from timeit import default_timer as timer



#function to read the data from the text files. 
def read_data(data_path):
    text = []
    target = []
    
    spam_ham_dict = {"spam": [], "ham": []}
    
    #walk through all the directories and subdirectories in the training directory
    for dirpath, dirs, files in os.walk(data_path):
        for filename in files:
            fname = os.path.join(dirpath,filename)
            
            #open the file, read it and store it in the respective list in the dictionary, 
            #find the right dictionary key (label) by checking whether it is in the ham or spam directory. 
            with open(fname, "r", encoding = "latin1") as f:
                if dirpath[-3:] == "ham":
                	try:
                		spam_ham_dict["ham"].append(f.read())
                	except:
                		print("couldnt read ", fname)
                		continue
                elif dirpath[-4:] == "spam":
                	try:
                		spam_ham_dict["spam"].append(f.read())
                	except:
                		print("couldnt read ", fname)
                		continue
                else:
                	continue    
    return spam_ham_dict






#function to read the data from the text files. 
def read_data_10percent(data_path):
    text = []
    target = []
    
    spam_ham_dict = {"spam": [], "ham": []}
    spam_count = 0
    ham_count = 0
    
    #walk through all the directories and subdirectories in the training directory
    for dirpath, dirs, files in os.walk(data_path):
        count = 0
        for filename in files:
            count+=1
            fname = os.path.join(dirpath,filename)
            
            #open the file, read it and store it in the respective list in the dictionary, 
            #find the right dictionary key (label) by checking whether it is in the ham or spam directory. 
            with open(fname, encoding = "latin1") as f:
                if dirpath[-3:] == "ham":
                    ham_count+=1
                    if ham_count > 950:
                        #if both spam and ham are greater than the required numbers, break
                        if spam_count > 750:
                            return spam_ham_dict
                        else:
                            continue
                    else:
                        spam_ham_dict["ham"].append(f.read())

                elif dirpath[-4:] == "spam":
                    spam_count+=1
                    if spam_count > 750:
                        if ham_count > 950:
                            return spam_ham_dict
                        else:
                            continue
                    else:
                        spam_ham_dict["spam"].append(f.read())
                else:
                    continue
    return spam_ham_dict



#removes punctuation from the text 
def remove_new_line(text):
    mystring = text.replace('\n', ' ').replace('\r', '')
    return mystring

#calls remove_punctuation to remove punctuation and ten tokenizes the text
def tokenize(text):
    text = remove_new_line(text).lower()
    return text.split()

#cleans data by removing punctuation and tokenizing the text.
def get_clean_data(text_list):
    clean_data = []
    for t in text_list:
        clean_data.append(tokenize(t))
    return clean_data

#gets all the words from te sentence list
def get_words(sentence_list):
    word_list = [item for sublist in sentence_list for item in sublist]
    return word_list

#gets a set of vocabulary from the word list.
def get_vocabulary(word_list):
    return set(word_list)


def get_all_word_counts(word_list):
    word_count_dict = {}
    for word in word_list:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
    return word_count_dict








#takes the data dictionary that is read from the files, computes the vocabulary, prior probabilities and the conditional probabilities and returns them.
def train_NB_model(spam_ham_dict):
    
    #clean the data
    for k in spam_ham_dict.keys():
        clean_data = get_clean_data(spam_ham_dict[k])
        spam_ham_dict[k] = clean_data
        
    prior_probabilities = {"spam":0, "ham":0}

    
    #get the counts and a prior probabilities
    spam_count = len(spam_ham_dict["spam"])
    ham_count = len(spam_ham_dict["ham"])
    total_count = spam_count + ham_count
    
    prior_probabilities["spam"] = spam_count/total_count
    prior_probabilities["ham"] = ham_count/total_count
    
    
    #get all the words.
    spam_words = get_words(spam_ham_dict["spam"])
    ham_words = get_words(spam_ham_dict["ham"])
    spam_words_total_count = len(spam_words)
    ham_words_total_count = len(ham_words)
    all_words = spam_words + ham_words

    vocabulary = list(get_vocabulary(all_words))
    
    
    #get the word counts
    spam_word_count = get_all_word_counts(spam_words)
    ham_word_count = get_all_word_counts(ham_words)
    
    
    #get all the conditional probabilities.
    spam_conditional_probabilities = {k: v/spam_count for k,v in spam_word_count.items()}
    ham_conditional_probabilities = {k: v/ham_count for k,v in ham_word_count.items()}
    
    #get all the laplace smoothed conditional probabilities
    spam_conditional_probabilities_smoothed = {k: (v+1)/(spam_words_total_count+len(vocabulary)) for k,v in spam_word_count.items()}
    ham_conditional_probabilities_smoothed = {k: (v+1)/(ham_words_total_count+len(vocabulary)) for k,v in ham_word_count.items()}

    spam_conditional_probabilities_smoothed["total_spam_word_count"] = spam_words_total_count
    ham_conditional_probabilities_smoothed["total_ham_word_count"] = ham_words_total_count
    
    
    #conditional probabilities of spam and ham for every token
    conditional_probabilites = [spam_conditional_probabilities, ham_conditional_probabilities]
    smoothed_conditional_probabilities = [spam_conditional_probabilities_smoothed, ham_conditional_probabilities_smoothed]
    
    return vocabulary, prior_probabilities, smoothed_conditional_probabilities
    
    



def main():
	path_to_data = sys.argv[1]

	
	start_reading = timer()
	data_dict = read_data_10percent(path_to_data)
	finish_reading = timer()
	print("time to read", finish_reading-start_reading)


	print(len(data_dict["spam"]))
	print(len(data_dict["ham"]))

	start_training = timer()
	V,PP,CP = train_NB_model(data_dict)
	end_training = timer()
	print("time to train ", end_training - start_training)
	print("trained model")

	model_data = {"vocabulary": V, "prior_probabilities": PP, "conditional_probabilities":CP}
	json.dump(model_data, open("nbmodel.txt",'w', encoding = "latin1"))
	print("json dump done")





if __name__ == '__main__':
	main()

