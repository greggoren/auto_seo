import os
import gensim
import logging.handlers

logger = logging.getLogger('train_word2vec')
logger.setLevel(logging.DEBUG)

BASE_FILE_PATH    = os.path.dirname(os.path.abspath(__file__))
FILES_FILE_PATH   = os.path.join(BASE_FILE_PATH, "files/")
RESULTS_FILE_PATH = os.path.join(BASE_FILE_PATH, "results/")
STOP_WORDS_FILE = ""
class Sentences(object):
    
    def __init__(self, dirname):
        
        logger.info("Sentences: init")

        self.dirname      = dirname


        self.stopwords_list = []
        with open(os.path.join(FILES_FILE_PATH, STOP_WORDS_FILE), 'r') as f:
            for line in f.readlines():
                self.stopwords_list.append(line.rstrip("\n"))

    def __iter__(self):
        """
        Iterate over sentences from the corpus.
        Get file with corpus documents. Each document is a sentence for the model.
        Build a list of sentences where each sentence converts to a list of words.
        For example: "The dog ate my HW" ---> [The, dog, ate, my, HW]
        """

        for fname in os.listdir(os.path.join(FILES_FILE_PATH, self.dirname)):
            # Each file line is a single document
            for line in open(os.path.join(FILES_FILE_PATH, self.dirname, fname)).readlines():
                yield line.rstrip('\n').split()


    """
    def pre_processing_sentences(self):
        
        logger.info("Sentences: pre_processing_sentences ------------------->")
        
        try:
            for fname in os.listdir(self.dirname):
                with open(os.path.join(FILES_FILE_PATH, 'preprocessing', 'preprocess_' + fname), 'w') as outfile:
                    
                    logger.info("pre_processing_sentences: going over file %r" % (fname))
                    fname = os.path.join(self.dirname, fname)
                    if not os.path.isfile(fname):
                        continue
                    # Each file line is a single document
                    with io.open(fname, 'r', encoding='utf-8') as f:
                        for line in f.readlines():
                            words = []
                            for term in line.rstrip('\n').split():
                                # Ignore terms that are not words  
                                if term == '[OOV]':
                                    continue 
                                if len(term) > 30:
                                        # logger.info("check_legal_word: length of word " + term + " is bigger than 30 characters")
                                    continue
                                elif term in self.stopwords_list:
                                    continue

                                # Clear text from special characters
                                term = unicodedata.normalize('NFKD', unicode(term)).encode('ascii','ignore')
                                term = term.encode(encoding='UTF-8',errors='strict')
                                term = re.sub("[!-/]|[:-@]|[[-`]|[{-~]", "", term)
                                words.append(term)
                            
                            outfile.write(' '.join(words))
                            outfile.write('\n')
                    
                outfile.close()
                
        except Exception as e:
            logger.error("pre_processing_sentences: Exception: " + e.message)
            
        logger.info("Sentences: pre_processing_sentences <-------------------")

    def check_legal_word(self, word):
        
        logger.info("check_legal_word: -------------------> ")
        
        # Check if length of word <= 45
        if len(word) > 40:
            logger.info("check_legal_word: length of word " + word + " is bigger than 40 characters")
            word = word[:40]
        
        # Check if the word is spelled correct    
        is_correct_spelling = self.spellchecker.check(word)
        if is_correct_spelling == False:
            suggested_list = self.spellchecker.suggest(word)
            if len(suggested_list) != 0:
                logger.info("check_legal_word: the word " + word + " is misspelled, suggest new word " + suggested_list[0])
                word = suggested_list[0] if word != suggested_list[0].lower() else word
        
        logger.info("check_legal_word: <------------------- ")
    """
class WordToVec(object):
    
    def __init__(self):
        
        logger.info("WordToVec: init")
        

        self.dirname = os.path.join(FILES_FILE_PATH,"docFiles")

    def train_model(self):
    
        logger.info("train_model: -------------------> ")
        
        try:
        
            sentences = Sentences(dirname = self.dirname)

            model = gensim.models.Word2Vec(sentences = sentences, 
                                           size      = 300, 
                                           window    = 8,
                                           min_count = 0, 
                                           workers   = 4,compute_loss=True)
            
            model.train(sentences      = sentences, 
                        total_examples = model.corpus_count, 
                        epochs         = 2)
    
            # Normalized vectors
            model.wv.init_sims(replace = True)
            # Save model to directory
            logger.info("train_model: Save model to directory")
            model.wv.save_word2vec_format(os.path.join(FILES_FILE_PATH,'word2vec_model'),
                                          binary = True)
        
        except Exception as e:
            logger.error("train_model: Exception: " + str(e))
            
        logger.info("train_model: <------------------- ")
        
    def load_model(self):
        
        logger.info("load_model: -------------------> ")
        
        model = gensim.models.KeyedVectors.load_word2vec_format(
                    fname  = os.path.join(FILES_FILE_PATH, self.corpus_name + '_word2vec'), 
                    binary = True,
                    limit  = 700000)
        
        queries = self.queries_map.values()
        words = set()
        for query in queries:
            for word in query.split():
                words.add(self.stemmer.stem(word))
                
        for word in words:
            if word not in model.wv.vocab:
                logger.info("load_model: the word " + str(word) + " is missing in the model")
        
        logger.info("load_model: <------------------- ")
        
if __name__ == '__main__':
    
    try:

        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        LOGS_FILE_PATH = os.path.join(BASE_FILE_PATH, "logs/")
        LOG_FILENAME   = os.path.join(LOGS_FILE_PATH, 'train_word2vec.log') 
        

        fh = logging.handlers.RotatingFileHandler(
                      filename    = LOG_FILENAME, 
                      maxBytes    = 10485760, 
                      backupCount = 10)
        
        # Add log formatter with timestamp
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # add the handler to the root logger
        logger = logging.getLogger('train_word2vec')
        logger.addHandler(fh)
        
        # Set up a specific logger with our desired output level
        logger.setLevel(logging.DEBUG)
        
        # Create class instance
        train_word2vec = WordToVec()
        train_word2vec.train_model()

    except Exception as e:
        print (str(e))
