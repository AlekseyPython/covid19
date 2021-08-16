import multiprocessing
import pymorphy2
import nltk
    

class Lemmatizator:
    def __init__(self, remove_end_punctuation_sentences=True):
        self.remove_end_punctuation_sentences = remove_end_punctuation_sentences
        
        self.lemmatizator = pymorphy2.MorphAnalyzer()
        self.table_of_unnecessary_characters = self._create_table_of_unnecessary_characters()

    def lemmatize_data(self, data, columns):
        for column in columns:
            data_column = data[column]
            with multiprocessing.Pool() as pool:
                lemmatized_column = pool.map(self.lemmatize_sentence, data_column)
                
            data[column] = lemmatized_column
        return data
    
    def lemmatize_sentence(self, sentence):
        if not sentence:
            return sentence
        
        sentence = self._remove_unnecessary_characters(sentence)
        words = nltk.word_tokenize(sentence)
        lemmatized_words = self._lemmatize_words(words)
        sentence = ' '.join(lemmatized_words)
        sentence = sentence.strip()
        return sentence
    
    def _create_table_of_unnecessary_characters(self):
        table_of_simbols = {}
        
        table_of_simbols[ord('@')] = ord(' ')
        table_of_simbols[ord('#')] = ord(' ')
        table_of_simbols[ord('№')] = ord(' ')
        table_of_simbols[ord('$')] = ord(' ')
        table_of_simbols[ord('%')] = ord(' ')
        table_of_simbols[ord('^')] = ord(' ')
        table_of_simbols[ord('&')] = ord(' ')
        table_of_simbols[ord('*')] = ord(' ')
        table_of_simbols[ord('(')] = ord(' ')
        table_of_simbols[ord(')')] = ord(' ')
        table_of_simbols[ord('-')] = ord(' ')
        table_of_simbols[ord('_')] = ord(' ')
        table_of_simbols[ord('=')] = ord(' ')
        table_of_simbols[ord('+')] = ord(' ')
        table_of_simbols[ord('[')] = ord(' ')
        table_of_simbols[ord(']')] = ord(' ')
        table_of_simbols[ord('{')] = ord(' ')
        table_of_simbols[ord('}')] = ord(' ')
        table_of_simbols[ord(':')] = ord(' ')
        
        table_of_simbols[ord('"')] = ord(' ')
        table_of_simbols[ord("'")] = ord(' ')
        table_of_simbols[ord('/')] = ord(' ')
        table_of_simbols[ord('|')] = ord(' ')
        table_of_simbols[ord('\\')] = ord(' ')
        table_of_simbols[ord('\t')] = ord(' ')
        table_of_simbols[ord(',')] = ord(' ')
        table_of_simbols[ord('<')] = ord(' ')
        table_of_simbols[ord('>')] = ord(' ')
        
        if self.remove_end_punctuation_sentences:
            table_of_simbols[ord('?')] = ord(' ')
            table_of_simbols[ord('!')] = ord(' ')
            table_of_simbols[ord('.')] = ord(' ')
            table_of_simbols[ord(';')] = ord(' ')
            
        for index in range(10):
            table_of_simbols[ord(str(index))] = ord(' ')
            
        return table_of_simbols
    
    def _remove_unnecessary_characters(self, sentence):
        return sentence.translate(self.table_of_unnecessary_characters)
            
    def _lemmatize_words(self, words):
        lemmatized_words = []
        
        if self.remove_end_punctuation_sentences:
            stored_words = set(['не', 'Не'])
        else:
            stored_words = set(['не', 'Не', '.', '!', '?', ';'])
            
        for word in words:
            if len(word) < 5:
                if word.isupper():
                    #this is an abbreviation that, after decryption, will be of sufficient length
                    pass
                
                elif word in stored_words:
                    pass
                
                else:
                    continue
            
            word = self.lemmatizator.parse(word)[0].normal_form
            lemmatized_words.append(word)
        return  lemmatized_words 
         
