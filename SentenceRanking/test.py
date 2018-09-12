from krovetzstemmer import Stemmer
from nltk.stem import PorterStemmer
a = " Word is complexity in form of ability InMotion"
stemmer = Stemmer()
p_stemmer = PorterStemmer()
print([stemmer.stem(j) for j in a.split()])
print([p_stemmer.stem(j) for j in a.split()])