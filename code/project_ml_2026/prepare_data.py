from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

def local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[1]
    return s.rsplit("/", 1)[1]

def find_iri_by_localname(g: Graph, name: str) -> URIRef:
    for s in g.all_nodes():
        if isinstance(s, URIRef) and local_name(s) == name:
            return s
    raise KeyError(f"Could not find IRI with local name: {name}")

def get_transitive_subclasses(g: Graph, root: URIRef) -> Set[URIRef]:
  
    children = {}
    for c, _, p in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(c, URIRef) and isinstance(p, URIRef):
            children.setdefault(p, set()).add(c)

    out = set()
    stack = list(children.get(root, []))
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(list(children.get(x, [])))
    return out

def get_leaf_subclasses(g: Graph, root: URIRef) -> Set[URIRef]:
    subs = get_transitive_subclasses(g, root)
    has_child = set()
    for c, _, p in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(c, URIRef) and isinstance(p, URIRef) and p in subs:
            has_child.add(p)
    return {c for c in subs if c not in has_child}


@dataclass
class FeatureSpec:
    cat_feature_names: List[str]               
    action_leaf_classes: List[URIRef]          
    file_feature_classes: List[URIRef]         
    exe_class: URIRef
    dll_class: URIRef
    has_action: URIRef
    has_file_feature: URIRef
    numeric_props: List[URIRef]                

class PEMalwareOntologyTabular(Dataset):

    def __init__(
        self,
        owl_path: str,
        examples_json_path: str,
        normalize_numeric: bool = True,
        log1p_numeric: bool = True,
    ):
        super().__init__()
        self.g = Graph()
        self.g.parse(owl_path)

        self.exe_class = find_iri_by_localname(self.g, "ExecutableFile")
        self.dll_class = find_iri_by_localname(self.g, "DynamicLinkLibrary")
        self.action_class = find_iri_by_localname(self.g, "Action")
        self.has_action = find_iri_by_localname(self.g, "has_action")
        self.has_file_feature = find_iri_by_localname(self.g, "has_file_feature")

        num_names = [
            "exports_count", "imports_count", "mz_count", "symbols_count",
            "path_strings_count", "registry_strings_count", "url_strings_count"
        ]
        self.numeric_props = [find_iri_by_localname(self.g, n) for n in num_names]
        ff_names = ["Debug", "Relocations", "Resources", "Signature", "TLS", "CLR", "NonexecutableEntryPoint"]
        self.file_feature_classes = [find_iri_by_localname(self.g, n) for n in ff_names]
        self.file_feature_name_by_iri = {iri: n for iri, n in zip(self.file_feature_classes, ff_names)}
        leaf_actions = sorted(list(get_leaf_subclasses(self.g, self.action_class)), key=lambda u: str(u))
        self.action_leaf_classes = leaf_actions
        self.action_name_by_iri = {iri: local_name(iri) for iri in self.action_leaf_classes}
        with open(examples_json_path, "r", encoding="utf-8") as f:
            ex = json.load(f)
 
        pos =  [i.split(":")[1] for i in ex.get("positive")]
        neg = [i.split(":")[1] for i in ex.get("negative")]

        self.ids: List[str] = pos + neg
        self.y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)

        self.file_iri_by_id = {}
        for id_ in self.ids:
            found = None
            for s in self.g.subjects(RDF.type, None):
                if isinstance(s, URIRef) and local_name(s) == id_:
                    found = s
                    break
            if found is None:
                raise KeyError(f"Could not locate individual for sample id {id_} in OWL")
            self.file_iri_by_id[id_] = found
        self.actions_present: Dict[str, Set[URIRef]] = {id_: set() for id_ in self.ids}
        for s, _, a_ind in self.g.triples((None, self.has_action, None)):
            if not isinstance(s, URIRef) or not isinstance(a_ind, URIRef):
                continue
            sid = local_name(s)
            if sid not in self.actions_present:
                continue
            for t in self.g.objects(a_ind, RDF.type):
                if isinstance(t, URIRef) and t in self.action_leaf_classes:
                    self.actions_present[sid].add(t)

        
        self.features_present: Dict[str, Set[URIRef]] = {id_: set() for id_ in self.ids}
        for s, _, f_ind in self.g.triples((None, self.has_file_feature, None)):
            if not isinstance(s, URIRef) or not isinstance(f_ind, URIRef):
                continue
            sid = local_name(s)
            if sid not in self.features_present:
                continue
            for t in self.g.objects(f_ind, RDF.type):
                if isinstance(t, URIRef) and t in self.file_feature_classes:
                    self.features_present[sid].add(t)

        self.is_exe = np.zeros(len(self.ids), dtype=np.int64)
        for i, id_ in enumerate(self.ids):
            s = self.file_iri_by_id[id_]
            types = set(self.g.objects(s, RDF.type))
            self.is_exe[i] = 1 if self.exe_class in types else 0  

       
        self.x_num = np.zeros((len(self.ids), len(self.numeric_props)), dtype=np.float32)
        for i, id_ in enumerate(self.ids):
            s = self.file_iri_by_id[id_]
            for j, p in enumerate(self.numeric_props):
                vals = list(self.g.objects(s, p))
                if len(vals) == 0:
                    self.x_num[i, j] = 0.0
                else:
                    self.x_num[i, j] = float(vals[0].toPython())

        if log1p_numeric:
            self.x_num = np.log1p(self.x_num)

        if normalize_numeric:
            mu = self.x_num.mean(axis=0, keepdims=True)
            sigma = self.x_num.std(axis=0, keepdims=True) + 1e-6
            self.x_num = (self.x_num - mu) / sigma

        
        self.cat_feature_names: List[str] = (
            ["is_exe"] +
            [self.file_feature_name_by_iri[c] for c in self.file_feature_classes] +
            [self.action_name_by_iri[c] for c in self.action_leaf_classes]
        )
        assert len(self.cat_feature_names) == 1 + 7 + len(self.action_leaf_classes)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sid = self.ids[idx]

        
        cat = np.zeros((1 + 7 + len(self.action_leaf_classes),), dtype=np.int64)
        cat[0] = self.is_exe[idx]

        
        for k, cls in enumerate(self.file_feature_classes, start=1):
            cat[k] = 1 if cls in self.features_present[sid] else 0

        base = 1 + 7
        present = self.actions_present[sid]
        for j, cls in enumerate(self.action_leaf_classes):
            cat[base + j] = 1 if cls in present else 0

        x_cat = torch.from_numpy(cat).long()
        x_num = torch.from_numpy(self.x_num[idx]).float()
        y = torch.tensor(self.y[idx]).float()
        return x_cat, x_num, y

