import arxiv

# Create arXiv client
client = arxiv.Client()

# Create search
search = arxiv.Search(
    query="large language models",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

# Get results
results = client.results(search)

# Print results
for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Title:", result.title)
    print("Authors:", ", ".join(author.name for author in result.authors))
    print("Summary:", result.summary[:500])
    print("URL:", result.entry_id)