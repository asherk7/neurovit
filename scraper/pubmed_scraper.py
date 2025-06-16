from pymed import PubMed

def fetch_articles(query):
    pubmed = PubMed(tool="MyTool", email="your.email@example.com")
    results = pubmed.query(query, max_results=5)
    articles = []
    for article in results:
        article_dict = article.toDict()
        articles.append({
            'title': article_dict['title'],
            'abstract': article_dict['abstract'],
            'authors': ', '.join([author['lastname'] for author in article_dict['authors']]),
            'doi': article_dict['doi']
        })
    return articles
