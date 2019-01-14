from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences

text = """French Lick Resort and Casino Sued by Colorado Artist for Copyright Infringement in Commissioned Artwork.Indianapolis, IN - French Lick Resorts and Casino Group has been sued in the Southern District of Indiana by Pamela Mougin for Copyright Infringement .In the 2,000's it was restored in connection and converted into a hotel / casino in an effort to revitalize the economically depressed region."""
print(retrieve_sentences(text))
