import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    number_pgs = len(corpus)
    probability_distribution = {}

    # Probability of choosing a link from the current page
    lnk_prob = damping_factor / len(corpus[page]) if len(corpus[page]) > 0 else 0

    for p in corpus:
        # Probability of choosing a random link from any page
        rndm_prob = (1 - damping_factor) / number_pgs
        # Total probability for the current page
        if p in corpus[page]:
            sum_prob = lnk_prob 
        else: 
            sum_prob = 0
        sum_prob += rndm_prob
        probability_distribution[p] = sum_prob

    return probability_distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #Initialization
    Pge_Rnk = {page: 0 for page in corpus}

    # Choose a random page to start
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        Pge_Rnk[current_page] += 1
        # Use the transition model to get the next page
        probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(probabilities.keys()), weights=probabilities.values())[0]

    # Normalize the PageRank values
    total_samples = sum(Pge_Rnk.values())
    Pge_Rnk = {page: rank / total_samples for page, rank in Pge_Rnk.items()}

    return Pge_Rnk



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialization of PageRank values
    page_rank = {page: 1 / len(corpus) for page in corpus}

    # Iteration - PageRank calculation
    while True:
        new_Pge_Rnk = {page: (1 - damping_factor) / len(corpus) for page in corpus}

        for page in corpus:
            for lnkng_page, Lnks in corpus.items():
                if page in Lnks:
                    new_Pge_Rnk[page] += damping_factor * (page_rank[lnkng_page] / len(Lnks))

        # Check for convergence (using a small threshold)
        if all(abs(new_Pge_Rnk[page] - page_rank[page]) < 0.001 for page in corpus):
            break

        page_rank = new_Pge_Rnk

    return page_rank



if __name__ == "__main__":
    main()
