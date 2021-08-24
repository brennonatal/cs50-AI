import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # iterating over domains
        for var in self.domains:
            # list to store worlds to be removed
            remove_list = []
            # iterating over worlds inside domains
            for world in self.domains[var]:
                # verifying node consistency
                if len(world) != var.length:
                    remove_list.append(world)
            # remove worlds that do not satisfy node consistency
            for w in remove_list:
                self.domains[var].remove(w)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False

        remove_list = []

        if self.crossword.overlaps[x, y] is not None:
            # for each domain of x check if domains of y can be placed
            i, j = self.crossword.overlaps[x, y]
            for v1 in self.domains[x]:
                success_matches = len(self.domains[y])
                for v2 in self.domains[y]:
                    if v1[i] != v2[j]:
                        success_matches -= 1
                if success_matches == 0:
                    remove_list.append(v1)

        # remove worlds that do not satisfy arc consistency
        for w in remove_list:
            self.domains[x].remove(w)
            revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = []
        # if arcs is none we combine all variables in self.domains
        if arcs is None:
            for x in self.domains:
                for y in self.crossword.neighbors(x):
                    if self.crossword.overlaps[x, y] is not None:
                        queue.append((x, y))

        while queue != []:
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(self.domains) == len(assignment):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # checking if words are unique
        words = list(assignment.values())
        # sets removes duplicates
        if len(words) != len(set(words)):
            return False

        # checking if every value is the correct length
        for variable in assignment:
            if variable.length != len(assignment[variable]):
                return False

        # looking for conflicts between neighbouring variables
        for variable in assignment:
            for neighbour in self.crossword.neighbors(variable):
                if neighbour in assignment:
                    i, j = self.crossword.overlaps[variable, neighbour]
                    if assignment[variable][i] != assignment[neighbour][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        order = []

        neighbors = self.crossword.neighbors(var)
        for var_value in self.domains[var]:
            n_conflicts = 0
            for neighbor in neighbors:
                if neighbor in assignment:
                    continue
                else:
                    i, j = self.crossword.overlaps[var, neighbor]
                    for neighbor_value in self.domains[neighbor]:
                        if var_value[i] != neighbor_value[j]:
                            n_conflicts += 1
            order.append((var_value, n_conflicts))

        order = [word[0] for word in sorted(order, key=lambda item: item[1])]

        return order

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        variables = self.domains.keys()

        # list of dicts that contains unassigned variables details
        unassigned = []

        for var in variables:
            if var not in assignment:
                unassigned.append({
                    'var': var,
                    # select by fewest
                    'n_values': len(self.domains[var]),
                    # select by highest
                    'degree': len(self.crossword.neighbors(var))
                })

        # ordering by fewest values in domain
        order = [item['var'] for item in sorted(
            unassigned, key=lambda details: details['n_values'])]

        return order[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # check if assignment is complete
        if self.assignment_complete(assignment):
            return assignment

        # select an unassigned variable
        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result != None:
                    return result
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
