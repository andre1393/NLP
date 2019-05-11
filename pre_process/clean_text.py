import re
import nltk
import spacy
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.stem.rslp import RSLPStemmer
import re
nltk.download('rslp', quiet = True)
nltk.download('stopwords', quiet = True)

STEM = "Stemizacao"
LEM = "Lemmatizacao"
    
class TextCleaner:
    
    def __init__(self, lan = 'english', norm = None, load_v = None, clean_regex = "([aA-zZ]+)", stemmer = RSLPStemmer(), custom_func = []):
        self.regex = clean_regex
        self.stops = nltk.corpus.stopwords.words(lan)
        
        if load_v == None:
            if lan == "english":
                self.nlp = spacy.load('en')
            elif lan == "portuguese":
                self.nlp = spacy.load('pt_core_news_sm')
            else:
                raise("idioma invalido")
        else:
            self.nlp = spacy.load(load_v)
        self.ps = stemmer
        
        if (norm != STEM) and (norm != LEM) and (norm != None):
            raise("Normalizacao invalida")
        self.norm = norm
        self.custom_func = custom_func
        
    def clean(self, text_input, stops = None):
        if stops != None:
            self.stops = stops
        
        if type(text_input) == pd.Series:
            result = text_input.apply(self.remove_special_char).apply(lambda x: self.remove_stop_words(x, self.stops)).apply(self.normalize)
            for f in self.custom_func:
                result = result.apply(f)
            return result
        elif type(text_input) == str:
            return self.clean_one(text_input)
        elif type(text_input) == list:
            result = []
            for text in text_input:
                result.append(self.clean_one(text))
            return result
        else:
            raise('Tipo de input de texto nao reconhecido')

    def clean_one(self, text):
        result = self.normalize(self.remove_stop_words(self.remove_special_char(text), self.stops))
        for f in self.custom_func:
            result = f(result)
        return result

    def remove_special_char(self, text):
        return " ".join(["".join(re.findall(self.regex, word)).lower() for word in text.split(" ")])
    
    def remove_stop_words(self, text, stops):
        if stops != None:
            return " ".join([word if word not in self.stops else "" for word in text.split(" ")])
        else:
            return text
        
    def normalize(self, text):
        text = re.sub(' +', ' ', text)
        text = re.sub('^ | $', '', text)
        if len(text) > 0:
            if self.norm == LEM:
                doc = self.nlp(text)
                return " ".join([token.lemma_ for token in doc])
            elif self.norm == STEM:
                return " ".join([self.ps.stem(word) for word in text.split(" ")])
        return text