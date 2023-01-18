from .data import LookUpTable
from networkx import DiGraph


class IntersectionGraph(DiGraph):
    def __init__(self, lookup_table: LookUpTable):
        super().__init__()
        for src in lookup_table.get_sources():
            for dst in lookup_table.get_detectors_from(src)['Ending Intersection'].unique():
                self.add_edge(src, dst)
        for dst in lookup_table.get_destinations():
            for src in lookup_table.get_detectors_to(dst)['Starting Intersection'].unique():
                self.add_edge(src, dst)
