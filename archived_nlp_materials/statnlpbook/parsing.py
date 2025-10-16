from graphviz import Digraph
from collections import defaultdict
import statnlpbook.util as util
import math


def render_forest(trees):
    """
    Renders a (parse) tree forest using graphiz
    Args:
        trees: list of recursive tree structure: tuple (node_label, children_nodes)
               for non-terminals or a string for terminals.

    Returns:
        the Digraph object representing the forest. Can be rendered in notebook.
    """
    nodes = []
    edges = []

    def collect_graph(current):
        if isinstance(current, tuple):
            children = [collect_graph(child) for child in current[1]]
            node_id = str(len(nodes))
            nodes.append((node_id, current[0]))
            for child_id, _ in children:
                edges.append((node_id, child_id))
        else:
            node_id = str(len(nodes))
            nodes.append((node_id, current))
        return nodes[-1]

    for tree in trees:
        collect_graph(tree)
    dot = Digraph(comment='The Round Table')
    for node_id, node_label in nodes:
        dot.node(node_id, node_label)
    for arg1_id, arg2_id in edges:
        dot.edge(arg1_id, arg2_id)

    return dot


def render_tree(tree):
    """
    Renders a (parse) tree using graphiz
    Args:
        tree: recursive tree structure: tuple (node_label, children_nodes) for non-terminals or a string for terminals.

    Returns:
        the Digraph object representing the tree. Can be rendered in notebook.
    """
    return render_forest([tree])


def filter_non_terminals(tree, allowed_non_terminals, collapse_if_singleton=True):
    if isinstance(tree, str):
        return tree if collapse_if_singleton else [tree]
    else:
        non_terminal, children = tree
        filtered_children = sum([filter_non_terminals(child, allowed_non_terminals, False) for child in children], [])
        if non_terminal in allowed_non_terminals:
            result = [(non_terminal, filtered_children)]
        else:
            result = filtered_children
        return result[0] if collapse_if_singleton and len(result) == 1 else result


def render_transitions(transitions):
    class Output:
        def _repr_html_(self):
            rows = ["<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                " ".join(state.buffer),
                render_forest(state.stack)._repr_svg_(),
                action)
                    for state, action in transitions]
            return "<table>{}</table>".format("\n".join(rows))

    return Output()


def get_label(node):
    if isinstance(node, str):
        return node
    else:
        return node[0]


class Chart:
    """
    A CYK Chart representation, optmised for visualisation of the chart and the individual CYK steps.
    """

    def __init__(self, sentence):
        self.sentence = sentence
        self.entries = defaultdict(list)  # map from (Int,Int,NonTerminal) to (list of (int,int,str) cells)
        self.cell_to_entries = defaultdict(list)
        self.target_cell_labels = set()
        self.source_cell_labels = set()
        self.target_cells = set()
        self.source_cells = set()
        self.scores = defaultdict(lambda: -math.inf)

    def __getitem__(self, key):
        return self.entries[key]

    def __setitem__(self, key, value):
        self.entries[key] = value
        begin, end, label = key
        self.cell_to_entries[begin, end] += [(label, value)]

    def score(self, begin, end, label):
        return self.scores[begin, end, label]

    def append_label(self, begin, end, label, sources=None):
        self[begin, end, label] += [[]] if sources is None else [sources]

    def update_label(self, begin, end, label, score=0.0, sources=None):
        self[begin, end, label] = [[]] if sources is None else [sources]
        self.scores[begin, end, label] = score

    def entries_at_cell(self, begin, end):
        return self.cell_to_entries[begin, end]

    def labels_at_cell(self, begin, end):
        return util.distinct_list([label for label, _ in self.cell_to_entries[begin, end]])

    def mark_target_label(self, begin, end, label):
        self.target_cell_labels.add((begin, end, label))

    def mark_source_label(self, begin, end, label):
        self.source_cell_labels.add((begin, end, label))

    def mark_source(self, begin, end):
        self.source_cells.add((begin, end))

    def mark_target(self, begin, end):
        self.target_cells.add((begin, end))

    def clear_label_marks(self):
        self.target_cell_labels.clear()
        self.source_cell_labels.clear()

    def clear_cell_marks(self):
        self.target_cells.clear()
        self.source_cells.clear()

    def clear_marks(self):
        self.clear_cell_marks()
        self.clear_label_marks()

    def mark_cyk_focus(self, begin, middle, end):
        self.clear_marks()
        self.mark_target(begin, end)
        self.mark_source(begin, middle)
        self.mark_source(middle + 1, end)
        return self

    def mark_cyk_rule(self, begin, end, alpha):
        self.clear_label_marks()
        self.mark(begin, end, alpha)
        return self

    def mark_cyk_source_focus(self, begin, middle, end, beta, gamma):
        self.clear_label_marks()
        self.mark_source_label(begin, middle, beta)
        self.mark_source_label(middle + 1, end, gamma)
        return self

    def mark_cyk_terminal(self, i, non_terminal):
        self.clear_label_marks()
        self.mark_target_label(i, i, non_terminal)
        return self

    def mark(self, begin, end, label, source_index=0):
        self.mark_target_label(begin, end, label)
        for b, e, l in self[begin, end, label][source_index]:
            self.mark_source_label(b, e, l)

    def derive_trees(self, begin=None, end=None, label='S'):
        b = begin if begin is not None else 0
        e = end if end is not None else len(self.sentence) - 1
        result = []
        for sources in self[b, e, label]:
            if len(sources) == 2:
                (bc1, ec1, lc1), (bc2, ec2, lc2) = sources
                for child_1 in self.derive_trees(bc1, ec1, lc1):
                    for child_2 in self.derive_trees(bc2, ec2, lc2):
                        tree = (label, [child_1, child_2])
                        result.append(tree)
            else:
                tree = (label, [self.sentence[begin]])
                result.append(tree)
        return result

    def _repr_html_(self):
        header = "<tr><td></td>{}</tr>".format(
            "".join(["<td>{}: {}</td>".format(i, self.sentence[i]) for i in range(0, len(self.sentence))]))
        rows = [header]

        def color_for_label(b, e, l):
            if (b, e, l) in self.target_cell_labels:
                return 'blue'
            elif (b, e, l) in self.source_cell_labels:
                return 'red'
            else:
                return 'black'

        def color_for_cell(b, e):
            if (b, e) in self.target_cells:
                return '#C3C3C3'
            elif (b, e) in self.source_cells:
                return '#E1E1E1'
            else:
                return 'white'

        def text_for_cell(row_index, col_index, label):
            if (row_index, col_index, label) in self.scores:
                score = self.scores[row_index, col_index, label]
                return '<font color="{0}">{1}: {2:.2f}</font>'.format(color_for_label(row_index, col_index, label), label,
                                                               score)
            else:
                return '<font color="{}">{}</font>'.format(color_for_label(row_index, col_index, label), label)

        for row_index in range(0, len(self.sentence)):
            cells = []
            for col_index in range(0, len(self.sentence)):
                relevant_entries = self.cell_to_entries[row_index, col_index]
                relevant_labels = util.distinct_list([label for label, _ in relevant_entries])
                border = 'solid' if col_index >= row_index else 'none'
                labels = [text_for_cell(row_index, col_index, label) for
                          label in relevant_labels]
                cell_color = color_for_cell(row_index, col_index)
                cell = """<td style="border:{};" bgcolor="{}">{}</td>""".format(border, cell_color,
                                                                                ", ".join(labels))
                cells.append(cell)
            row = "<tr><td>{}: {}</td>{}</tr>".format(row_index, self.sentence[row_index], "".join(cells))
            rows.append(row)
        return "<table>{}</table>".format("\n".join(rows))
