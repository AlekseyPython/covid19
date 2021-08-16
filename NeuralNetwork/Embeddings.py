import os, sys
import multiprocessing
import nltk
from gensim.models import Word2Vec
from .Lemmatizator import Lemmatizator
import Settings, Functions
from collections import Counter


class Embeddings:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.directory_with_medical_books = Settings.FOLDER_WITH_MEDICAL_BOOKS
        self.embeddings = Settings.PATH_EMBEDDINGS
    
    @staticmethod 
    def discarding_weights_output_layer(model): 
        #freezes the model, keeping the hidden layer weights
        #and discards output weights predicting co-occurrence of words
        #further training of the model after discarding the weights of the output layer is not can
        model.init_sims(replace=True)
        return model
    
    def get_model(self):
        if os.path.exists(self.embeddings):
            return self._load_model()
            
        sentences = self._get_sentences()
        model = self._learning_word2vec(sentences)
        self._save_model(model)
        return model
        
    def _get_sentences(self):
        medical_books = [file_name for file_name in os.listdir(self.directory_with_medical_books) if file_name.endswith('.txt')]
        with multiprocessing.Pool() as pool:
            sentences_of_books = pool.map(self._get_sentences_book, medical_books)
        
        # sentences_of_books = []
        # for book in medical_books:
        #     sentences_of_books.append(self._get_sentences_book(book))
            
        return Functions.concatenate_list_of_lists(sentences_of_books)
                  
    def _get_sentences_book(self, file_name):
        def this_line_hit_on_table(line):
            if line.find('   ') >= 0:
                return True
            
            first_duble_space = line.find('  ')
            if first_duble_space >= 0:
                if line.find('  ', first_duble_space+2) >= 0:
                    return True
            return False
                    
        text = ''
        lemmatizator = Lemmatizator(remove_end_punctuation_sentences=False)
        full_path = self.directory_with_medical_books + '/' + file_name
        with open(full_path, mode='rt') as f:
            lines = f.readlines() # list containing lines of file
            
            start_line = ''
            for line in lines: 
                line = start_line + line
                start_line = ''
                
                line = line.strip()
                if not line:
                    continue
                
                if this_line_hit_on_table(line):
                    #this is a table, in which several different sentences are mixed 
                    continue
                        
                if line.endswith('-'):
                    last_space = line.rfind(' ')
                    if last_space == -1:
                        start_line = line[:-1]
                        line = ''
                        
                    else:
                        start_line = line[last_space+1:-1]
                        line = line[:last_space]
                
                elif len(line) < 50 and line[-1] != '.':
                    #short lines accept as a sentence
                    line += '.'
                
                if line:
                    lemmatize_line = lemmatizator.lemmatize_sentence(line)
                    text += lemmatize_line + ' '
        
        #split by sentences    
        sentences = nltk.tokenize.sent_tokenize(text)
        
        #split sentences by words
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentence = nltk.word_tokenize(sentence)
            tokenized_sentence = tokenized_sentence[:-1]
            if len(tokenized_sentence) > 3:
                tokenized_sentences.append(tokenized_sentence)
            
        return tokenized_sentences
                     
    def _learning_word2vec(self, sentences):
        model = Word2Vec(sentences,
                        seed        = self.random_seed,
                        min_count   = 10,
                        vector_size = 64, 
                        workers     = multiprocessing.cpu_count(),
                        window      = 4,
                        epochs      = 100
                        )  
        return model
    
    def _save_model(self, model):
        model.save(self.embeddings)
    
    def _load_model(self):
        return Word2Vec.load(self.embeddings)

        