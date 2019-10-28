# news_collector
collect news from google news and group the news by category

News are group by one of Topic Modelling Algoritm called Latent Dirichlet Allocation [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

Requirements
- Python 3 only
- sklearn : to load the LatentDirichletAllocation model and CountVectorizer
- numpy : to extract the value from Dataframe
- wordcloud : to create the wordcloud for collected text
- pyLDAvis : to draw topic distribution graph
- pandas : to create Dataframe for collected news 
- bs4 : to crawl the words from designated url
