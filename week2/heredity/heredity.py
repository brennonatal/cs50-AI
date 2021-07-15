import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def num_of_genes(person, one_gene, two_genes):
    '''
    Returns the number of genes for a given person
    '''
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    probabilities = []

    for person_name, person_data in people.items():
        # getting the number of genes for this person
        n_genes = num_of_genes(person_name, one_gene, two_genes)

        # for anyone with no parents listed in the dataset
        if person_data['mother'] is None and person_data['father'] is None:
            # unconditional probabilities of having the genes and the trait(or not)
            probabilities.append(PROBS['gene'][n_genes] * PROBS['trait'][n_genes][person_name in have_trait])

        # for those who have parents listed in the dataset
        else:
            # getting the number of genes from the parents
            n_genes_father = num_of_genes(person_data['father'], one_gene, two_genes)
            n_genes_mother = num_of_genes(person_data['mother'], one_gene, two_genes)
            # initializing individual probability (for good pratice)
            prob = 1

            # if the person only has one copy of the gene, it may have been inherited from the father or mother 
            if n_genes == 1:
                if n_genes_mother == 2 and n_genes_father == 0:
                    prob = PROBS['mutation'] * PROBS['mutation']                # inherited from the mother: both must mutate
                    prob += (1 - PROBS['mutation']) * (1 - PROBS['mutation'])   # inherited from the father: none will mutate
                elif n_genes_mother == 1 and n_genes_father == 0:
                    prob = 0.5 * (1 - PROBS['mutation'])    # inherited from the mother: 50% for mom and 99% for dad (father's gene must not mutate)
                    prob += 0.5 * PROBS['mutation']         # inherited from the father: 50% for mom 1% for dad (father's gene must mutate)
                elif n_genes_mother == 0 and n_genes_father == 0:
                    prob = PROBS['mutation'] * (1 - PROBS['mutation'])    # inherited from the mother: 1% for mom and 99% for dad (mother's gene must mutate)
                    prob += (1 - PROBS['mutation']) * PROBS['mutation']   # inherited from the father: 99% for mom 1% for dad (father's gene must mutate)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif n_genes_mother == 2 and n_genes_father == 1:
                    prob = (1 - PROBS['mutation']) * 0.5    # inherited from the mother: 99% for mom 50% for dad
                    prob += PROBS['mutation'] * 0.5         # inherited from the father: 1% for mom 50% for dad (mother's gene must mutate)
                elif n_genes_mother == 1 and n_genes_father == 1:
                    prob = 0.5 * 0.5    # inherited from the mother: 50% for both parents
                    prob += 0.5 * 0.5   # inherited from the father: 50% for both parents
                elif n_genes_mother == 0 and n_genes_father == 1:
                    prob = (1 - PROBS['mutation']) * 0.5  # inherited from the mother: 99% for mom and 50% for dad
                    prob += PROBS['mutation'] * 0.5     # inherited from the father: 1% for mom 50% for dad (mother's gene must mutate)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif n_genes_mother == 2 and n_genes_father == 2:
                    prob = (1 - PROBS['mutation']) * PROBS['mutation']    # inherited from the mother: 99% for mom 1% for dad (father's gene must mutate)
                    prob += PROBS['mutation'] * (1 - PROBS['mutation'])   # inherited from the father: 1% for mom 99% for dad (mother's gene must mutate)
                elif n_genes_mother == 1 and n_genes_father == 2:
                    prob = 0.5 * PROBS['mutation']          # inherited from the mother: 50% for mom and 1% for dad (father's gene must mutate)
                    prob += 0.5 * (1 - PROBS['mutation'])   # inherited from the father: 50% for mom 99% for dad 
                elif n_genes_mother == 0 and n_genes_father == 2:
                    prob = PROBS['mutation'] * PROBS['mutation']                # inherited from the mother: 1% for mom and 1% for dad (both must mutate)
                    prob += (1 - PROBS['mutation']) * (1 - PROBS['mutation'])   # inherited from the father: 99% for mom 99% for dad
                
                # probability that the person have the trait or not
                prob *= PROBS['trait'][n_genes][person_name in have_trait]

                probabilities.append(prob)

            # the child must receive 1 gene from each parent
            elif n_genes == 2:
                if n_genes_mother == 2:
                    prob = 1 - PROBS['mutation']    # the child will receive 1 gene from mother with a 99% chance
                elif n_genes_mother == 1:
                    prob = 0.5    # the child will receive 1 gene from mother with a 50% chance
                elif n_genes_mother == 0:
                    prob = PROBS['mutation']    # the child will receive 1 gene from mother with a 1% chance (gene must mutate)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if n_genes_father == 2:
                    prob *= 1 - PROBS['mutation']    # the child will receive 1 gene from father with a 99% chance
                elif n_genes_father == 1:
                    prob *= 0.5    # the child will receive 1 gene from father with a 50% chance
                elif n_genes_father == 0:
                    prob *= PROBS['mutation']    # the child will receive 1 gene from father with a 1% chance (gene must mutate)
                
                # probability that the person have the trait or not
                prob *= PROBS['trait'][n_genes][person_name in have_trait]

                probabilities.append(prob)
            
            # the child must not receive any genes
            elif n_genes == 0:
                if n_genes_mother == 2:
                    prob = PROBS['mutation']    # the child will not receive the gene from mother with a 1% chance (gene must mutate)
                elif n_genes_mother == 1:
                    prob = 0.5    # the child will not receive the gene from mother with a 50% chance
                elif n_genes_mother == 0:
                    prob = 1 - PROBS['mutation']    # the child will not receive the gene from mother with a 99% chance
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if n_genes_father == 2:
                    prob *= PROBS['mutation']    # the child will not receive the gene from father with a 1% chance (gene must mutate)
                elif n_genes_father == 1:
                    prob *= 0.5    # the child will not receive the gene from father with a 50% chance
                elif n_genes_father == 0:
                    prob *= 1 - PROBS['mutation']    # the child will not receive the gene from father with a 99% chance
                
                # probability that the person have the trait or not
                prob *= PROBS['trait'][n_genes][person_name in have_trait]

                probabilities.append(prob)

    # calculating joint probability
    joint_prob = 1
    for i in probabilities:
         joint_prob *= i
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    for person in probabilities:

        n_genes = num_of_genes(person, one_gene, two_genes)
        probabilities[person]['gene'][n_genes] += p
        
        probabilities[person]['trait'][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        trait_probs_sum = sum(probabilities[person]['trait'].values())
        probabilities[person]['trait'][True] /= trait_probs_sum
        probabilities[person]['trait'][False] /= trait_probs_sum

        gene_probs_sum = sum(probabilities[person]['gene'].values())

        for i in range(3):
            probabilities[person]['gene'][i] /= gene_probs_sum


if __name__ == "__main__":
    main()
