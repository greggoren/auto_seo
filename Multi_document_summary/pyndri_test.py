import pyndri
import params
index = pyndri.Index(params.path_to_index)
token2id, id2token, id2df = index.get_dictionary()