import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S Conj VP
NP -> N | Det NP | AP NP | PP NP
VP -> V | V NP | VP PP | Adv VP | VP Adv
AP -> Adj | Adj AP
PP -> P NP | P S
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # converting all characters to lowercase
    sentence = sentence.lower()
    # tokenizing sentence
    sentence = nltk.tokenize.word_tokenize(sentence)
    print(sentence)
    # list of words to remove
    remove_list = []

    for word in sentence:
        not_alpha = 0
        for w in word:
            if not w.isalpha():
                not_alpha += 1
        if not_alpha == len(word):
            remove_list.append(word)

    return [word for word in sentence if word not in remove_list]


def has_np_within(subtree):
    """Check if the given subtree has a NP within."""
    for leaf in subtree.subtrees():
        # ignore itself
        if leaf is subtree:
            continue
        if leaf.label() == 'NP':
            return True
    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_list = []

    for subtree in tree.subtrees():
        if subtree.label() == 'NP' and not has_np_within(subtree):
            np_list.append(subtree)
    return np_list


if __name__ == "__main__":
    main()
