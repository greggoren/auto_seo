from Preprocess.preprocess import retrieve_sentences,tokenize_sentence
# test = "Get website hosting services easily- Get your business online FAST! Our best-in-class industry themes and website hosting design make it possible for your website to be online on the internet in days. All with our website design hosting services for the World Wide Web. The relation with World Wide Web and the internet is also very important! Monitor your site along the way with 24/7 access to your Website Design Manager. View designs, send files and leave feedback Keep your site fresh without having to make changes yourself. Our staff will make 6 hours of updates during the first 12 months, at your request (Website plan only). Simple Sites can purchase updates. No hidden fees. With our website design hosting services, you pay a one-time fee to build your site and a small monthly hosting fee. Everything else is included, so there are no surprises."
test = "Choose Web.com for your Small Business Web Design and Small Business Website needs. Improve online presence with Online Marketing Services.\
Everything You Need to be Successful Online\
Customized website design and monthly marketing to connect with customers.\
Professional design including photos and copy created specifically for your business\
Setup of Google™ Local and monthly marketing submissions to top search engines and local directories\
Monthly website updates handled for you with just a phone call to your web expert\
Mobile website setup for smartphone capability\
Tracking tools to view your online success"
s=retrieve_sentences(test)
print(s)
for si in s:
    tokenize_sentence(si)