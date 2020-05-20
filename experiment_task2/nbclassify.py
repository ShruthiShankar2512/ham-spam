import sys 
import os 
import string
import json
import math

from timeit import default_timer as timer

def read_from_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data

def process_json(json_data):
    return set(json_data["vocabulary"]), json_data["prior_probabilities"], json_data["conditional_probabilities"]


def read_data_test(data_path):
    text_dict = {}
    #walk through all the directories and subdirectories in the training directory
    for dirpath, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(".txt"):
                #count+=1
                fname = os.path.join(dirpath,filename)
                #open the file, read it and store it in the the data array
                try:
                    with open(fname, encoding = "latin1") as f:
                        text_dict[fname]=[f.read()]
                except:
                    print("Couldnt read ", fname)
                    continue

    return text_dict



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


def apply_NB_model(text, vocabulary, conditional_probabilities, pp_spam, pp_ham, spam_length, ham_length, vocab_length):
    #calculate spam score
    prob_score_spam = pp_spam
    prob_score_ham = pp_ham

    

    #for each work in the email
    for token in text:
        #if the token is not in the vocabulary, skip it.
        if token in vocabulary:
            try:
                prob_score_spam += math.log( conditional_probabilities[0][token] )
                #print("\n\n\n\n\n\n\nupdated spam score", prob_score_spam)
            except KeyError:
                prob_score_spam += math.log((0+1)/( spam_length + vocab_length ))
        
            try:
                prob_score_ham += math.log(conditional_probabilities[1][token] )
                #print("\n\n\n\n\n\n\nupdated spam score", prob_score_spam)
            except KeyError:
                prob_score_ham += math.log((0+1)/( ham_length + vocab_length ))
        else:
            #print(token)
            #print("token not in vocab")
            continue
    label = "null"
    if prob_score_spam >= prob_score_ham:
        label = "spam"
    else:
        label = "ham"
    return label
    




def NB_classify(data_dict, vocabulary, prior_probabilities, conditional_probabilities):
    for k in data_dict.keys():   
        clean_data = get_clean_data(data_dict[k])
        data_dict[k] = clean_data.pop()
    
    output_list = []
    #for each email

    spam_length = conditional_probabilities[0]["total_spam_word_count"]
    vocab_length = len(vocabulary)
    ham_length = conditional_probabilities[1]["total_ham_word_count"]

    pp_spam = math.log(prior_probabilities["spam"])
    pp_ham = math.log(prior_probabilities["ham"])

    pp_spam = prior_probabilities["spam"]
    pp_ham = prior_probabilities["ham"]
    count = 0
    for k in data_dict.keys():
    	count+=1
    	email = data_dict.get(k)
    	label = apply_NB_model(email, vocabulary, conditional_probabilities, pp_spam, pp_ham, spam_length, ham_length, vocab_length)

    	output_list.append(''.join([label,"\t",k]))
    	#if count%100 == 0:
    		#print(count, label, k)
    
    with open("nboutput.txt", "w") as outfile:
        outfile.write("\n".join(output_list))
        


def main():
	path_to_data = sys.argv[1]

	start_reading = timer()
	data_dict = read_data_test(path_to_data)
	finish_reading = timer()
	print("time to read", finish_reading-start_reading)
	print("read test data")

	#get the NB model from the stored text file.
	vocabulary, PP, CP = process_json(read_from_json("nbmodel.txt"))
	print("read nb model")

	#classify the test data
	start_classifying = timer()
	NB_classify(data_dict, vocabulary, PP, CP)
	end_classifying = timer()
	print("time to classify all data: ", end_classifying - start_classifying)
	print("classification done")







if __name__ == "__main__":
	main()



            
            
            
        