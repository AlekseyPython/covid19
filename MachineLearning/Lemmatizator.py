import pymorphy2
    

class Lemmatizator:
    def __init__(self, data):
        self.data = data
        self.lemmatizator = pymorphy2.MorphAnalyzer()

    def lemmatize(self, columns):
        data = self.data  
        lenght = len(data)
        for column in columns:
            for index_row in range(lenght):
                sentence = data.loc[index_row, column]
                if not sentence:
                    continue
                
                data.loc[index_row, column] = self._lemmatize_sentence(sentence)
        return data
    
    def _lemmatize_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = self._remove_unnecessary_characters(sentence)
        words = sentence.split()
        lemmatized_words = self._lemmatize_words(words)
        sentence = ' '.join(lemmatized_words)
        sentence = sentence.strip()
        return sentence
        
    def _remove_unnecessary_characters(self, sentence):
        table_of_simbols = {}
        table_of_simbols[ord('!')] = ord(' ')
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
        table_of_simbols[ord(';')] = ord(' ')
        table_of_simbols[ord('"')] = ord(' ')
        table_of_simbols[ord("'")] = ord(' ')
        table_of_simbols[ord('/')] = ord(' ')
        table_of_simbols[ord('|')] = ord(' ')
        table_of_simbols[ord('\\')] = ord(' ')
        table_of_simbols[ord('\t')] = ord(' ')
        table_of_simbols[ord(',')] = ord(' ')
        table_of_simbols[ord('.')] = ord(' ')
        table_of_simbols[ord('<')] = ord(' ')
        table_of_simbols[ord('>')] = ord(' ')
        
        for index in range(10):
            table_of_simbols[ord(str(index))] = ord(' ')
        
        sentence = sentence.translate(table_of_simbols)
        return sentence       
            
    def _lemmatize_words(self, words):
        lemmatized_words = []
        for word in words:
            word = self.lemmatizator.parse(word)[0].normal_form
            if len(word)>=5:
                lemmatized_words.append(word)
        return  lemmatized_words      
