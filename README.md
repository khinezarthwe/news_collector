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


#### Start the project 
 ```pip install -r requirements.txt```
 
#### modify search keywords:
 ##### google search rss url 
 ```https://news.google.com/rss/search?q={query}```
 
 ##### for example 
 
- search keyword: ironman
    - Default: 
        - ```https://news.google.com/rss/search?q=ironman```
    - For Japanese version:
        - ```https://news.google.com/rss/search?q=ironman&hl=ja&gl=JP&ceid=JP:ja```
    -  For English version:
        - ```https://news.google.com/rss/search?q=ironman&hl=en-US&gl=US&ceid=US:en```

- search keyword: ant-man and the wasp
    - Default: 
        - ```https://news.google.com/rss/search?q=ant-man%20and%20the%20wasp```
    - For english version:
        - ```https://news.google.com/rss/search?q=ant-man+and+the+wasp&hl=en-US&gl=US&ceid=US:en```
    - For japanese version:
        - ```https://news.google.com/rss/search?q=ant-man+and+the+wasp&hl=ja&gl=JP&ceid=JP:ja```
 
 

### Issues
-   library warning error (pandas, matplotlib)
-   adding logging
-   error handling 
-   pandas future warning 
-   adding evaluation development to Result
-   adding explanation to current task

### Feature Extension
-   can add search keyword by user 
-   can customize the topic number
