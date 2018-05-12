import base64
import functools
import hashlib

from collections import defaultdict
from operator import itemgetter

from typing import Iterable, Optional, Tuple, NamedTuple, FrozenSet

import os
import six

from delphin.mrs import Pred, Xmrs
from delphin.mrs.config import IVARG_ROLE
from delphin.mrs.eds import Eds
from delphin.mrs.components import nodes as mrs_nodes, links as mrs_links


@functools.lru_cache(maxsize=65536)
def strip_category(cat):
    if cat.endswith("u_unknown"):
        lemma, pos_and_sense = cat.rsplit("/", 1)
        pos_part, sense_part = pos_and_sense.split("_", 1)
        lemma_part = "X"
    else:
        pred_obj = Pred.stringpred(cat)
        lemma_part = "X" if cat.startswith("_") else pred_obj.lemma
        pos_part = str(pred_obj.pos)
        sense_part = str(pred_obj.sense)
    return lemma_part + "_" + pos_part + "_" + sense_part


@six.python_2_unicode_compatible
class GraphNode(object):
    def __init__(self, id_=None, is_root=False):
        self.name = id_ or base64.b64encode(os.urandom(15)).decode("ascii")
        self.is_root = is_root

    def __str__(self):
        return "GraphNode: {}".format(self.name)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@six.python_2_unicode_compatible
class HyperEdge(object):
    def __init__(self,
                 nodes,  # type: Iterable[GraphNode]
                 label,  # type: str
                 is_terminal,  # type: bool
                 span=None  # type: Optional[Tuple[int, int]]
                 ):
        self.nodes = tuple(nodes)  # as immutable list
        self.label = label  # type: str
        self.is_terminal = is_terminal  # type: bool
        self.span = span  # type: Optional[Tuple[int, int]]

    def __str__(self):
        return "{}{}: {}".format(self.span if self.span is not None else "", self.label,
                                 " -- ".join(i.name for i in self.nodes))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.nodes) ^ hash(self.label) ^ hash(self.span)

    def __eq__(self, other):
        return isinstance(other, HyperEdge) and self.nodes == other.nodes and \
               self.label == other.label and self.span == other.span


class PredEdge(HyperEdge):
    def __init__(self,
                 pred_node,  # type: GraphNode
                 span,  # type: Tuple[int, int]
                 label  # type: str
                 ):
        super(PredEdge, self).__init__([pred_node], label, True, span)

    @classmethod
    def as_new(cls,
               span,  # type: Tuple[int, int]
               label,  # type: str
               name  # type: str
               ):
        pred_node = GraphNode(name)
        return pred_node, cls(pred_node, span, label)

    @classmethod
    def from_eds_node(cls,
                      eds_node,  # type:
                      lemma_to_x=False
                      ):
        name = str(eds_node.pred)
        if lemma_to_x:
            name = strip_category(name)
        return cls.as_new(eds_node.lnk.data, name,
                          eds_node.nodeid)


class HyperGraph(NamedTuple("HyperGraph", [("nodes", FrozenSet[GraphNode]),
                                           ("edges", FrozenSet[HyperEdge])])):
    """
    :type nodes: FrozenSet[GraphNode]
    :type edges: FrozenSet[HyperEdge]
    """

    @classmethod
    def from_eds(cls,
                 e,  # type: Eds
                 lemma_to_x=False
                 ):
        """:rtype: HyperGraph"""
        nodes = []
        nodes_by_pred_label = {}
        edges = []
        for node in e.nodes():
            graph_node, edge = PredEdge.from_eds_node(node, lemma_to_x)
            graph_node.is_root = (node.nodeid == e.top)
            nodes_by_pred_label[node.nodeid] = graph_node
            nodes.append(graph_node)
            edges.append(edge)

        for node in e.nodes():
            for label, target in e.edges(node.nodeid).items():
                edges.append(HyperEdge([nodes_by_pred_label[node.nodeid],
                                        nodes_by_pred_label[target]], label=label,
                                       is_terminal=True))

        return cls(frozenset(nodes), frozenset(edges)).to_standardized_node_names()

    @classmethod
    def from_mrs(cls,
                 m,  # type: Xmrs
                 lemma_to_x=False
                 ):
        """:rtype: HyperGraph"""
        nodes = []
        name_to_number = {}
        nodes_by_pred_label = {}
        edges = []
        for node in m.eps():
            graph_node, edge = PredEdge.from_eds_node(node, lemma_to_x)
            graph_node.is_root = (node.label == m.top)
            nodes_by_pred_label[node.nodeid] = graph_node
            name_to_number[node.label] = node.nodeid
            nodes.append(graph_node)
            edges.append(edge)

        for start, end, rargname, post in mrs_links(m):
            if start == 0:
                continue
            edges.append(HyperEdge([nodes_by_pred_label[start],
                                    nodes_by_pred_label[end]],
                                   label=rargname + "/" + post,
                                   is_terminal=True))

        return cls(frozenset(nodes), frozenset(edges)).to_standardized_node_names()

    def draw(self, output, file_format="pdf", attr_map=None, node_name_map=None, show_span=True):
        from graphviz import Digraph
        dot = Digraph()
        if attr_map is None:
            attr_map = {}
        if node_name_map is None:
            node_name_map = {}

        for node in self.nodes:
            attr = attr_map.get(node) or {}
            attr.update({"width": "0.075", "height": "0.075", "fixedsize": "true"})
            label = node_name_map.get(node) or ""
            if label != "":
                attr.update({"width": "0.25", "height": "0.25"})
            dot.node(node.name, label=label, _attributes=attr)

        for edge in self.edges:
            attr = attr_map.get(edge) or {}
            attr.update({"arrowsize": "0.5"})
            if edge.span is not None:
                label = "{}({},{})".format(edge.label, edge.span[0], edge.span[1]) if show_span else edge.label
            else:
                label = edge.label
            if len(edge.nodes) == 1:
                fake_end = edge.nodes[0].name + label + "_end"
                dot.node(fake_end, label="",
                         _attributes={"width": "0.005", "height": "0.005", "fixedsize": "true",
                                      "color": "white"})
                dot.edge(edge.nodes[0].name, fake_end, label=label,
                         _attributes=attr)
            elif len(edge.nodes) == 2:
                dot.edge(edge.nodes[0].name, edge.nodes[1].name, label,
                         _attributes=attr)
            else:
                for idx, end_point in enumerate(edge.nodes[1:], 1):
                    dot.edge(edge.nodes[0].name, end_point.name,
                             label="{} (0 to {})".format(label, idx),
                             _attributes=attr)

        if file_format == "source":
            return dot.source
        else:
            dot.format = file_format
            dot.render(output, cleanup=True)

    def to_hgraph(self):
        from common.hgraph.hgraph import Hgraph
        hgraph = Hgraph()
        hgraph.my_hyper_graph = self

        for node in self.nodes:  # type: GraphNode
            label = ""
            ext_id = None
            ident = "_" + node.name

            # Insert a node into the AMR
            ignoreme = hgraph[ident]  # Initialize dictionary for this node
            hgraph.node_to_concepts[ident] = label
            if ext_id is not None:
                if ident in hgraph.external_nodes and hgraph.external_nodes[ident] != ext_id:
                    raise Exception("Incompatible external node IDs for node %s." % ident)
                hgraph.external_nodes[ident] = ext_id
                hgraph.rev_external_nodes[ext_id] = ident
            if node.is_root:
                hgraph.roots.append(ident)

        for edge in self.edges:  # type: HyperEdge
            hyperchild = tuple("_" + node.name for node in edge.nodes[1:])
            ident = "_" + edge.nodes[0].name
            new_edge = edge.label
            hgraph._add_triple(ident, new_edge, hyperchild)

        return hgraph

    def to_standardized_node_names(self, return_mapping=False):
        edge_by_node = defaultdict(list)
        for edge in self.edges:
            for idx, node in enumerate(edge.nodes):
                edge_by_node[node].append((edge, idx))

        default_hash = hashlib.md5(b"13").digest()
        node_hashes = {node: default_hash for node in self.nodes}

        def get_edge_hashes(edge,  # type: HyperEdge
                            idx  # type: int
                            ):
            md5_obj = hashlib.md5((edge.label + "#" + str(idx)).encode())
            for adj_node in edge.nodes:
                md5_obj.update(node_hashes[adj_node] + b"#")
            return md5_obj.digest()

        def get_sibling_hashes(node  # type: GraphNode
                               ):
            md5_obj = hashlib.md5()
            edge_hashes = sorted(get_edge_hashes(edge, idx)
                                 for edge, idx in edge_by_node[node])
            for h in edge_hashes:
                md5_obj.update(h)
            return md5_obj.digest()

        def recalculate_hashs():
            new_node_hashes = {}
            for node in self.nodes:
                md5_obj = hashlib.md5()
                md5_obj.update(get_sibling_hashes(node))
                new_node_hashes[node] = md5_obj.digest()
            return new_node_hashes

        for cycle in range(10):
            node_hashes = recalculate_hashs()

        nodes_in_order = sorted(node_hashes.items(), key=itemgetter(1))

        node_rename_map = {}
        for node_idx, (node, hash_value) in enumerate(nodes_in_order):
            node_rename_map[node] = GraphNode(str(node_idx))

        new_edges = []
        for edge in self.edges:
            new_edges.append(
                HyperEdge((node_rename_map[node] for node in edge.nodes),
                          edge.label, edge.is_terminal, edge.span))

        ret = self.__class__(frozenset(node_rename_map.values()),
                              frozenset(new_edges))
        if return_mapping:
            return ret, node_rename_map
        else:
            return ret

    def to_eds(self):
        node_names = {i: None for i in self.nodes}
        edges = []
