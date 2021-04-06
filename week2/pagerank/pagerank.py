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
    distribution = dict()
    # populating distribution with corpus keys
    for key in corpus:
        distribution[key] = 0
    # getting possible next pages
    possible_links_on_page = corpus[page]
    # distributing damping_factor among possible links
    for link in distribution:
        if link in possible_links_on_page:
            distribution[link] = ((1 - damping_factor) / len(distribution)) + (damping_factor / len(possible_links_on_page))
            continue
        distribution[link] = (1 - damping_factor) / len(distribution)

    return distribution

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = dict()
    # populating distribution with corpus keys
    for key in corpus:
        distribution[key] = 0
    # choosing first sample randomly
    choice = random.choice(list(corpus.keys()))
    # sampling n times
    for i in range(1, n):
        sample = transition_model(corpus, choice, damping_factor)

        for link in sample:
            distribution[link] = ((i - 1) * distribution[link] + sample[link]) / i

        choice = random.choices(list(distribution.keys()), list(distribution.values()), k=1)[0]

    return distribution

def sigma(corpus, distribution, link):
    total = 0
    for key in corpus:
        if link in corpus[key]:
            total += distribution[key] / len(corpus[key])
    return total

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank_variation = float('inf')
    n = len(corpus)
    distribution = dict()
    # assigning each page a rank of 1 / N
    for key in corpus:
        distribution[key] = 1 / n
    # the process should repeat until no PageRank value changes by more than 0.001
    while rank_variation >= 0.001:
        for link in distribution:
            old_distribution = distribution[link]
            distribution[link] = ((1 - damping_factor) / n) + (damping_factor * sigma(corpus, distribution, link))
            rank_variation = abs(distribution[link] - old_distribution)

    return distribution


if __name__ == "__main__":
    main()
