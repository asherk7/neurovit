from pymed import PubMed
import requests
import os
import time

pubmed = PubMed(tool="MyTool", email="masherk03@outlook.com")
SAVE_DIR = "rag/papers"

def fetch_articles(query, max_results=10):
    results = pubmed.query(query, max_results=max_results)
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

def get_free_pdf_url(doi):
    """Uses Unpaywall to check if there's a free fulltext PDF for a DOI"""
    headers = {"Accept": "application/json"}
    url = f"https://api.unpaywall.org/v2/{doi}?email=masherk03@outlook.com"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            data = response.json()
            if data.get("is_oa"):
                pdf_url = data.get("best_oa_location", {}).get("url_for_pdf")
                return pdf_url
    except Exception as e:
        print(f"Error checking Unpaywall for DOI {doi}: {e}")
    return None

def download_pdf(pdf_url, filename):
    try:
        response = requests.get(pdf_url, stream=True, timeout=20)
        if response.ok and "application/pdf" in response.headers.get("Content-Type", ""):
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"No valid PDF at: {pdf_url}")
    except Exception as e:
        print(f"Error downloading: {e}")

def run():
    queries = ["glioma tumor", "meningioma tumor", "pituitary tumor"]
    for query in queries:
        print(f"\nSearching for: {query}")
        articles = fetch_articles(query)
        for article in articles:
            print(f"\nTitle: {article['title']}")
            doi = article["doi"]
            if not doi:
                print("Skipping (no DOI)")
                continue

            doi = doi.split()[0].strip()

            pdf_url = get_free_pdf_url(doi)
            if pdf_url:
                safe_title = "".join(c if c.isalnum() else "_" for c in article["title"])[:80]
                filename = os.path.join(SAVE_DIR, f"{safe_title}.pdf")
                print(filename)
                download_pdf(pdf_url, filename)
            else:
                print("No free PDF found.")

            time.sleep(1)

if __name__ == "__main__":
    run()
