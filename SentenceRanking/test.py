from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences

text = """Top 10 New York (NY) hotels:
There is a big number of great hotels in the "Big Apple", but as future New York visitors we want only the best the New York can give us. So, here they are the best hotels in New York:
1) Chelsea Pines Inn.
2) Casablanca Hotel Times Square.
3) Library Hotel.
4) The Sherry-Netherland Hotel.
5) Hotel Giraffe.
6) 414 Hotel.
7) EVEN Hotel Times Square South.
8) NobleDEN Hotel.
9) The Bryant Park Hotel.
10) The Martime Hotel."""
print(retrieve_sentences(text))
