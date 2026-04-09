import wikipedia as wiki

print(wiki.search("Python"))
print(wiki.suggest("Pyth"))
print(wiki.summary("Python"))

wiki.set_lang("fr")
print(wiki.summary("Python"))

wiki.set_lang("en")
p = wiki.page("Python")

#To get the Title
print(p.title)

#To get the url of the article
print(p.url)

#To scrape the full article
print(p.content)

#To get all the images in the article
print(p.images)

#And to get all the referals used by Wikipedia in the article
print(p.links)