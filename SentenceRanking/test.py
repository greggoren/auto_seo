from krovetzstemmer import Stemmer
from nltk.stem import PorterStemmer
a = " word is complexity in form of ability merging"
stemmer = Stemmer()
p_stemmer = PorterStemmer()
print([stemmer.stem(j) for j in a.split()])
print([p_stemmer.stem(j) for j in a.split()])