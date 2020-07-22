# HACK: THIS IS USED TO LOAD UP THE CHAR DICT -- GET RID OF IT BY 
# EMBEDDING THE SYMBOLTABLE LOAD IN A PACKAGE

class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols
    :param starting_symbol: Starting index of symbol.
    :type starting_symbol: int
    :param unknown_symbol: Index of unknown symbol.
    :type unknown_symbol: int
    """

    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s = starting_symbol
        self.unknown = unknown_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in self.d.items()}
