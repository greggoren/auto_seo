from SpamTechniques.weaving import create_new_document_by_weaving

query = "New York Hotel"
text = "Hello this is a test to check what happens inside our function. A good example will be tested here and hopefully work"
new_text = create_new_document_by_weaving(text,query,0.6)
print(new_text)