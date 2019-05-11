import re
import string
from collections import Counter

class OrtographyCorrector:
    
    def __init__(self, dictionary, encoding = 'utf-8'):
        palavras = self.words(open(dictionary, encoding = encoding).read())
        self.WORDS = Counter(palavras)

    def words(self, text): return re.findall(r'\w+', text.lower())

    def P(self, word):
        N=sum(self.WORDS.values())
        "Probability of `word`."
        #print('\nP(%s): %f' % (word, (self.WORDS[word] / N)))
        return self.WORDS[word] / N

    def correction(self, word):
        #import pdb; pdb.set_trace()
        "Most probable spelling correction for word."
        word = word.lower()
        return max(self.candidates(word), key=self.P)

    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        #print('\nknow: ',set(w for w in words if w in WORDS))
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        #print ('\nedits1: ', set(deletes + transposes + replaces + inserts))
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))