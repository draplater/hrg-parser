from typing import List

import attr

from hrgguru.hyper_graph import HyperGraph, GraphNode, HyperEdge


@attr.s(slots=True)
class SubGraph(object):
    """
    Subgraph is a hypergraph corresponds to a span in the string,
    Subgraph doesn't contain any non-terminal edges.
    """
    graph = attr.ib(type=HyperGraph)
    external_nodes = attr.ib(type=List[GraphNode])

    @staticmethod
    def transform_edge(mapping, edge):
        """ transform the edge in the rule into edge in concrete graph."""
        return HyperEdge((mapping[i] for i in edge.nodes),
                         edge.label,
                         edge.is_terminal,
                         None)

    @staticmethod
    def transform_edge_2(mapping, edge):
        """ transform the edge in the rule into edge in concrete graph."""
        return HyperEdge(((mapping.get(i) or i) for i in edge.nodes),
                         edge.label,
                         edge.is_terminal,
                         edge.span)

    @classmethod
    def merge(cls, cfg_node, sync_rule,
              left_sub_graph,  # type: SubGraph
              right_sub_graph,  # type: SubGraph
              ):
        """ :rtype: SubGraph """
        # create concrete node and unify with external nodes of subgraphs
        nodes_mapping = {i: GraphNode() for i in sync_rule.hrg.rhs.nodes}
        external_nodes_map_left = {}
        external_nodes_map_right = {}
        left_name, left_edge = sync_rule.rhs[0]
        if left_edge is not None:
            assert len(left_sub_graph.external_nodes) == len(left_edge.nodes)
            external_nodes_map_left.update(
                {abstract_node: concrete_node
                 for abstract_node, concrete_node
                 in zip(left_edge.nodes, left_sub_graph.external_nodes)})

        right_name, right_edge = sync_rule.rhs[1]
        if right_edge is not None:
            assert len(right_sub_graph.external_nodes) == len(right_edge.nodes)
            external_nodes_map_right.update(
                {abstract_node: concrete_node
                 for abstract_node, concrete_node
                 in zip(right_edge.nodes, right_sub_graph.external_nodes)})
        nodes_mapping.update(external_nodes_map_left)
        nodes_mapping.update(external_nodes_map_right)
        common_mapping = {}
        if left_edge is not None and right_edge is not None:
            for abstract_node in external_nodes_map_left.keys() & external_nodes_map_right.keys():
                common_mapping[external_nodes_map_left[abstract_node]] = external_nodes_map_right[abstract_node]

        # build new graph
        edges = frozenset(cls.transform_edge(nodes_mapping, edge)
                          for edge in sync_rule.hrg.rhs.edges
                          if edge != left_edge and edge != right_edge)

        non_terminals = [i for i in edges if not i.is_terminal]
        if non_terminals:
            raise Exception("Non-terminals {} found by rule {} in node {}".format(
                non_terminals, sync_rule, cfg_node))
        for new_edge in edges:
            if len(new_edge.nodes) == 1 and new_edge.span is None:
                new_edge.span = cfg_node.extra["DelphinSpan"]
        nodes = frozenset(nodes_mapping.values())
        if right_edge is not None:
            edges |= right_sub_graph.graph.edges
            nodes |= right_sub_graph.graph.nodes
        if left_edge is not None:
            edges |= frozenset(cls.transform_edge_2(common_mapping, edge) for edge in left_sub_graph.graph.edges)
            # edges |= left_sub_graph.graph.edges
            nodes |= (left_sub_graph.graph.nodes - common_mapping.keys())
        external_nodes = tuple(nodes_mapping[node]
                               for node in sync_rule.hrg.lhs.nodes)
        sub_graph = HyperGraph(nodes, edges)
        return SubGraph(sub_graph, external_nodes)

    @classmethod
    def create_leaf_graph(cls, cfg_node, sync_rule):
        """ :rtype: SubGraph """
        nodes_mapping = {i: GraphNode() for i in sync_rule.hrg.rhs.nodes}
        edges = frozenset(cls.transform_edge(nodes_mapping, edge)
                          for edge in sync_rule.hrg.rhs.edges)
        for new_edge in edges:
            if len(new_edge.nodes) == 1:
                new_edge.span = cfg_node.extra["DelphinSpan"]
        sub_graph = HyperGraph(frozenset(nodes_mapping.values()),
                               edges)
        external_nodes = tuple(nodes_mapping[node]
                               for node in sync_rule.hrg.lhs.nodes)
        return SubGraph(sub_graph, external_nodes)
