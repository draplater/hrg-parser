from __future__ import print_function
import hashlib
from collections import defaultdict, Iterable
from operator import itemgetter

import sys

from itertools import permutations
from typing import NamedTuple, Set, List, Tuple, Union, Optional, Sequence, Dict

from hrgguru.const_tree import ConstTree, Lexicon
from hrgguru.hyper_graph import HyperEdge, GraphNode, HyperGraph


class HRGRule(NamedTuple("HRGRule", [("lhs", HyperEdge),
                                     ("rhs", HyperGraph)])):
    """
    :type lhs: HyperEdge
    :type rhs: HyperGraph
    """

    def __str__(self):
        return "{} -> \n{}\n".format(self.lhs, "\n".join(str(i) for i in self.rhs.edges))

    def __repr__(self):
        return self.__str__()

    @classmethod
    def extract(cls,
                edges,  # type: Set[HyperEdge]
                internal_nodes,  # type: Set[GraphNode]
                external_nodes,  # type: Set[GraphNode]
                label,  # type: str
                cfg_rule=None
                ):
        nodes = internal_nodes.union(external_nodes)
        edge_by_node = defaultdict(list)  # node -> (edge, index of this node in this edge)
        for edge in edges:
            for idx, node in enumerate(edge.nodes):
                edge_by_node[node].append((edge, idx))

        default_hash = hashlib.md5(b"13").digest()
        node_hashes = {node: default_hash for node in nodes}  # node -> hash

        def get_edge_hashes(
                node_hashes,  # type: Dict[GraphNode, bytes]
                edge,  # type: HyperEdge
                idx  # type: int
        ):
            md5_obj = hashlib.md5((edge.label + "#" + str(idx)).encode())
            for adj_node in edge.nodes:
                md5_obj.update(node_hashes[adj_node] + b"#")
            return md5_obj.digest()

        def get_sibling_hashes(
                node_hashes,  # type: Dict[GraphNode, bytes]
                node  # type: GraphNode
        ):
            md5_obj = hashlib.md5()
            edge_hashes = sorted(get_edge_hashes(node_hashes, edge, idx)
                                 for edge, idx in edge_by_node[node])
            for h in edge_hashes:
                md5_obj.update(h)
            return md5_obj.digest()

        for cycle in range(10):
            new_node_hashes = {}
            # recalculate hashes
            for node in nodes:
                md5_obj = hashlib.md5()
                md5_obj.update(get_sibling_hashes(node_hashes, node))
                md5_obj.update(b'\x01' if node in external_nodes else b'\x00')
                new_node_hashes[node] = md5_obj.digest()
            node_hashes = new_node_hashes

        nodes_in_order = sorted(node_hashes.items(), key=itemgetter(1))

        node_rename_map = {}
        for node_idx, (node, hash_value) in enumerate(nodes_in_order):
            node_rename_map[node] = GraphNode(str(node_idx))

        # get rhs
        new_edges = []
        for edge in edges:
            new_edges.append(
                HyperEdge((node_rename_map[node] for node in edge.nodes),
                          edge.label, edge.is_terminal))
        rhs = HyperGraph(frozenset(node_rename_map.values()),
                         frozenset(new_edges))

        # determine external nodes permutation
        def get_external_nodes_permutation():
            if len(external_nodes) == 2:
                for permutation in permutations(external_nodes):
                    if any(edge.nodes == permutation for edge in edges):
                        return [node_rename_map[i] for i in permutation]
                if cfg_rule is not None and len(cfg_rule.child) == 2:
                    left_span = cfg_rule.child[0].span
                    right_span = cfg_rule.child[1].span
                    left_node = [edge.nodes[0] for edge in edges
                                 if len(edge.nodes) == 1 and edge.span == left_span]
                    right_node = [edge.nodes[0] for edge in edges
                                  if len(edge.nodes) == 1 and edge.span == right_span]
                    if left_node and right_node and {left_node[0], right_node[0]} == external_nodes:
                        # print("Permutation rule 2 used")
                        return [node_rename_map[left_node[0]], node_rename_map[right_node[0]]]
            return sorted(
                (node_rename_map[i] for i in external_nodes),
                key=lambda x: int(x.name)
            )

        # get lhs
        lhs = HyperEdge(get_external_nodes_permutation(),
                        label=label,
                        is_terminal=False)
        return node_rename_map, cls(lhs, rhs)

    def apply(self,
              hg,  # type: HyperGraph
              edge  # type: HyperEdge
              ):
        assert edge in hg
        assert edge.label == self.lhs.label
        assert len(edge.nodes) == len(self.lhs.nodes)

    def draw_in_graph(self):
        raise NotImplementedError

    def rhs_to_hgraph(self):
        from common.cfg import NonterminalLabel
        from common.hgraph.hgraph import Hgraph
        nt_id_count = 0
        hgraph = Hgraph()

        for node in self.rhs.nodes:  # type: GraphNode
            label = ""
            try:
                ext_id = self.lhs.nodes.index(node)
            except ValueError:
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
            if ext_id == 0:
                hgraph.roots.append(ident)

        for edge in self.rhs.edges:  # type: HyperEdge
            hyperchild = tuple("_" + node.name for node in edge.nodes[1:])
            ident = "_" + edge.nodes[0].name
            if "_" not in edge.label and not edge.label.startswith("ARG") \
                    and not edge.label.startswith("BV"):
                # this is a nonterminal Edge
                new_edge = NonterminalLabel(edge.label)
                if not new_edge.index:
                    new_edge.index = "_%i" % nt_id_count
                    nt_id_count = nt_id_count + 1
            else:
                new_edge = edge.label

            hgraph._add_triple(ident, new_edge, hyperchild)

        return hgraph

    def to_grammar(self, rule_id):
        from parser.vo_rule import VoRule
        return VoRule(rule_id, self.lhs.label, 0.0,
                      self.rhs_to_hgraph(), None,
                      nodelabels=False, logprob=False)

    def draw(self, save_path, draw_format="png"):
        attrs = {}
        node_name_map = {node: str(idx) for idx, node in enumerate(self.lhs.nodes)}
        for node in self.lhs.nodes:
            attrs[node] = {"color": "red"}
        return self.rhs.draw(save_path, draw_format, attrs, node_name_map)


class HRGDerivation(list):
    @staticmethod
    def detect_small(hg, rule):
        local_spans = set([i.span for i in rule.child] + [rule.span])
        direct_edges = set(i for i in hg.edges if i.span in local_spans)
        if not direct_edges:
            return None
        related_nodes = set(i for edge in direct_edges for i in edge.nodes)
        related_edges = direct_edges.union(
            set(i for i in hg.edges
                if i.span is None and all(j in related_nodes for j in i.nodes)))
        internal_nodes = set(node for node in related_nodes
                             if all(edge in related_edges for edge in hg.edges
                                    if node in edge.nodes))
        external_nodes = related_nodes - internal_nodes

        if not external_nodes:
            # if not external nodes, random select one
            node = internal_nodes.pop()
            external_nodes.add(node)

        return related_edges, internal_nodes, external_nodes

    @staticmethod
    def detect_lexicalized(hg, rule):
        is_lexical = isinstance(rule.child[0], Lexicon)
        local_spans = set([i.span for i in rule.child] + [rule.span])
        direct_edges = set(i for i in hg.edges if i.span in local_spans)
        if not direct_edges:
            return None
        related_nodes = set(i for edge in direct_edges for i in edge.nodes)
        related_edges = direct_edges.union(
            set(i for i in hg.edges
                if i.span is None and all(j in related_nodes for j in i.nodes)))

        def get_outgoing_edges(node):
            # if some external node only have internal edges and outgoing edges,
            # it can be converted into internal node
            ret = []
            for edge in hg.edges:
                if edge.span is None and node == edge.nodes[0]:
                    ret.append(edge)
            return ret

        # edges that start with related_nodes
        if is_lexical:
            outgoing_edges = set(edge for node in related_nodes
                                 for edge in get_outgoing_edges(node))
            outgoing_nodes = set(i.nodes[1] for i in outgoing_edges)
            all_edges = related_edges.union(outgoing_edges)
            all_nodes = related_nodes.union(outgoing_nodes)
        else:
            all_edges = related_edges
            all_nodes = related_nodes

        internal_nodes = set(node for node in all_nodes
                             if all(edge in all_edges for edge in hg.edges
                                    if node in edge.nodes))
        external_nodes = all_nodes - internal_nodes

        if not external_nodes:
            # if not external nodes, random select one
            node = internal_nodes.pop()
            external_nodes.add(node)

        return all_edges, internal_nodes, external_nodes

    @staticmethod
    def detect_large(hg,  # type: HyperGraph
                     rule):
        local_spans = set([i.span for i in rule.child] + [rule.span])
        # edge that match certain span
        direct_edges = set(i for i in hg.edges if i.span in local_spans)
        if not direct_edges:
            return None
        # node that connected with direct_edges
        related_nodes = set(i for edge in direct_edges for i in edge.nodes)
        # edge that connects internally
        internal_edges = set(i for i in hg.edges
                             if i.span is None and all(j in related_nodes for j in i.nodes))

        all_edges_0 = direct_edges | internal_edges
        internal_nodes_0 = set(node for node in related_nodes
                               if all(edge in all_edges_0 for edge in hg.edges
                                      if node in edge.nodes))
        external_nodes_0 = related_nodes - internal_nodes_0

        def can_be_internal(node):
            # if some external node only have internal edges and outgoing edges,
            # it can be converted into internal node
            ret = []
            for edge in hg.edges:
                if node not in edge.nodes:
                    continue
                if edge in all_edges_0:
                    continue
                if edge.span is None and node == edge.nodes[0]:
                    ret.append(edge)
                    continue
                return []
            return ret

        # edges that start with related_nodes
        outgoing_edges = set(edge for node in external_nodes_0
                             for edge in can_be_internal(node))
        outgoing_nodes = set(i.nodes[1] for i in outgoing_edges)

        # all edges
        all_edges = direct_edges | internal_edges | outgoing_edges
        all_nodes = related_nodes | outgoing_nodes

        internal_nodes = set(node for node in all_nodes
                             if all(edge in all_edges for edge in hg.edges
                                    if node in edge.nodes))
        external_nodes = all_nodes - internal_nodes

        if not external_nodes:
            # if not external nodes, random select one
            node = internal_nodes.pop()
            external_nodes.add(node)

        return all_edges, internal_nodes, external_nodes

    @classmethod
    def convert_cfg_node(cls, node):
        if isinstance(node, Lexicon):
            return node
        ret = ConstTree(node.tag)
        for i in node.child:
            if isinstance(i, Lexicon) or i.has_semantics:
                ret.child.append(i)
            else:
                ret.child.extend(i.generate_words())
        ret.span = node.span
        return ret

    @classmethod
    def extract_compat(cls,
                       hg,  # type: HyperGraph
                       cfg, spans,
                       draw=False,
                       sent_id=None,
                       draw_format="png",
                       detect_func=None):
        # TODO: remove draw
        if detect_func is None:
            detect_func = cls.detect_large
        pics = []

        def generate_derivation(hg  # type: HyperGraph
                                ):
            lexicons = list(cfg.generate_words())
            assert len(lexicons) == len(spans)
            rules = list(cfg.generate_rules())

            for span, lexicon in zip(spans, lexicons):
                lexicon.span = span

            count = 1
            last_new_edge = None

            for rule in rules:
                new_span = (rule.child[0].span[0], rule.child[-1].span[1])
                rule.span = new_span

                result = detect_func(hg, rule)
                if result is None:
                    rule.has_semantics = False
                    continue
                else:
                    rule.has_semantics = True
                    all_edges, internal_nodes, external_nodes = result

                new_edge = HyperEdge(external_nodes, rule.tag, False, new_span)

                new_nodes = hg.nodes - internal_nodes
                new_edges = (hg.edges - all_edges) | {new_edge}

                hg_new = HyperGraph(new_nodes, new_edges)
                node_rename_map, hrg_rule = HRGRule.extract(
                    all_edges, internal_nodes, external_nodes, rule.tag)

                if draw:
                    pic_path = "/tmp/a3/{}/{}".format(sent_id, count)
                    pics.append(cls.draw(hg, pic_path, all_edges,
                                         internal_nodes, external_nodes, last_new_edge,
                                         draw_format=draw_format))

                hg = hg_new
                last_new_edge = new_edge
                count += 1
                hrg_rule.cfg = cls.convert_cfg_node(rule)
                yield node_rename_map, hrg_rule

            if draw:
                pic_path = "/tmp/a3/{}/{}".format(sent_id, count)
                pics.append(cls.draw(hg, pic_path, last_new_edge=last_new_edge,
                                     draw_format=draw_format))

        if draw_format == "source":
            return cls(generate_derivation(hg)), pics
        else:
            return cls(generate_derivation(hg))

    @classmethod
    def draw(cls, hg, path,
             all_edges=(),
             internal_nodes=(),
             external_nodes=(),
             last_new_edge=None,
             draw_format="png"
             ):
        attrs = {}
        for edge in all_edges:
            attrs[edge] = {"color": "red"}
        for node in internal_nodes:
            attrs[node] = {"color": "red"}
        for node in external_nodes:
            attrs[node] = {"style": "filled", "color": "red"}

        # draw old pic
        if last_new_edge:
            attrs[last_new_edge] = {"color": "blue"} \
                if last_new_edge not in all_edges else {"color": "violet"}

        return hg.draw(path, draw_format, attrs)


class CFGRule(NamedTuple("CFGRule", [
    ("lhs", str),  # CFG LHS
    ("rhs", Sequence[Union[Tuple[str, Optional[HyperEdge]], Tuple[Lexicon, None]]]),
    # rhs = List[(cfg_rhs_1, corresponding hrg edge1), ...]
    ("hrg", Optional[HRGRule])
    # hrg: corresponding hrg rule
])):
    """
    :type self.rhs: Tuple[Union[Tuple[str, Optional[HyperEdge]], Tuple[Lexicon, None]]]
    :type self.hrg: Optional[HRGRule]
    :a
    """

    @classmethod
    def extract(cls,
                hg,  # type: HyperGraph
                cfg,  # type: ConstTree
                draw=False,
                sent_id=None,
                draw_format="png",
                detect_func=None,
                lexicalize_null_semantic=False
                ):
        """ :rtype: List[CFGRule]"""
        # TODO: remove draw
        if detect_func is None:
            detect_func = HRGDerivation.detect_large
        pics = []

        def generate_derivation(hg  # type: HyperGraph
                                ):
            rules = list(cfg.generate_rules())  # root last

            count = 1
            last_new_edge = None

            for rule in rules:
                new_span = (rule.child[0].span[0], rule.child[-1].span[1])
                rule.span = new_span

                result = detect_func(hg, rule)

                # null semantic node
                if result is None:
                    rule.has_semantics = False
                    if lexicalize_null_semantic:
                        cfg_rhs = tuple((j, None) for j in rule.generate_words())  # type: Tuple[Tuple[Lexicon, None]]
                    else:
                        cfg_rhs = tuple((i if isinstance(i, Lexicon) else i.tag, None)
                                        for i in rule.child)
                    yield CFGRule(rule.tag, cfg_rhs, None)
                    continue
                else:
                    rule.has_semantics = True
                    all_edges, internal_nodes, external_nodes = result

                new_edge = HyperEdge(external_nodes, rule.tag, False, new_span)

                new_nodes = hg.nodes - internal_nodes
                new_edges = (hg.edges - all_edges) | {new_edge}

                hg_new = HyperGraph(new_nodes, new_edges)
                node_rename_map, hrg_rule = HRGRule.extract(
                    all_edges, internal_nodes, external_nodes, rule.tag, rule)

                if draw:
                    pic_path = "/tmp/a3/{}/{}".format(sent_id, count)
                    pics.append(HRGDerivation.draw(hg, pic_path, all_edges,
                                                   internal_nodes, external_nodes, last_new_edge,
                                                   draw_format=draw_format))

                hg = hg_new
                last_new_edge = new_edge
                count += 1

                if isinstance(rule.child[0], Lexicon):
                    # leaf node
                    assert len(rule.child) == 1
                    cfg_rhs = ((rule.child[0], None),)
                else:
                    # internal node
                    assert all(isinstance(i, ConstTree) for i in rule.child)
                    cfg_rhs = []
                    for i in rule.child:
                        if not i.has_semantics:
                            if lexicalize_null_semantic:
                                cfg_rhs.extend((j, None) for j in i.generate_words())
                            else:
                                cfg_rhs.append((i.tag, None))
                        else:
                            # find corresponding hyperedge in hrg rule for this tree node
                            target_edges = [j for j in all_edges if j.span == i.span]
                            assert len(target_edges) == 1
                            if target_edges[0].label != i.tag:
                                print("Non-consistent CFG and HRG: ",
                                      " ".join(j.string for j in rule.generate_words()),
                                      file=sys.stderr)
                                cfg_rhs = None
                                break
                            target_edges_r = HyperEdge((node_rename_map[node] for node in target_edges[0].nodes),
                                                       target_edges[0].label, target_edges[0].is_terminal)
                            cfg_rhs.append((i.tag, target_edges_r))

                if cfg_rhs is not None:
                    yield CFGRule(rule.tag, tuple(cfg_rhs), hrg_rule)
                else:
                    yield CFGRule(rule.tag, cfg_rhs, None)

            if draw:
                pic_path = "/tmp/a3/{}/{}".format(sent_id, count)
                pics.append(HRGDerivation.draw(hg, pic_path, last_new_edge=last_new_edge,
                                               draw_format=draw_format))

        if draw_format == "source":
            return list(generate_derivation(hg)), pics
        else:
            return list(generate_derivation(hg))
