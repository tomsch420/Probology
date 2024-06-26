import itertools
import json
import math
import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from variables.variables import *
from typing_extensions import List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import dfl.dlquery as dl

logger.info("Building cache of the DFL ontology")
dl.buildCache()
logger.info("Finished building cache of the DFL ontology")

def best_entropy(coll, kBest):
    def _B(st, universe):
        return [(e in st) for e in universe]

    def _X(a, b):
        return [x ^ y for x, y in zip(a, b)]

    def _H(v):
        if 0 == len(v):
            return 0
        p = sum([1 for x in v if x]) / len(v)
        if 0 == p:
            return 0
        return math.log2(p) * p

    selected = set()
    for k in sorted(list(coll.keys())):
        eqs = set() # dl.whatsEquivalent(k)
        if 0 == len(selected.intersection(eqs)):
            selected.add(k)
    universe = sorted(list(set().union(*list(coll.values()))))
    collV = {k: _B(v, universe) for k, v in coll.items() if (k in selected)}
    kept = {False: [0] * len(universe)}
    while len(kept) <= kBest:
        newE, newK = \
        sorted([(max([_H(_X(last, v)) for last in kept.values()]), k) for k, v in collV.items() if k not in kept])[0]
        print(newK, newE)
        kept[newK] = collV[newK]
    kept.pop(False)
    return sorted(list(kept.keys()))


obj2ConcsFile = os.path.join("data", "obj2Concs.json")

def prepOnto(obj2ConcsFile, kBest):
    obj2ConcsRaw = json.loads(open(obj2ConcsFile).read())
    obj2Conc = {}
    disp2Objs = {}
    sups2Objs = {}
    for o, cs in obj2ConcsRaw.items():
        obj2Conc[o] = None
        if 1 == len(cs):
            obj2Conc[o] = "dfl:" + cs[0]
            dispositions = dl.whatDispositionsDoesObjectHave(obj2Conc[o])
            superclasses = dl.whatSuperclasses(obj2Conc[o])
            for mapStore, stuff in [(disp2Objs, dispositions), (sups2Objs, superclasses)]:
                for e in stuff:
                    if e not in mapStore:
                        mapStore[e] = set()
                    mapStore[e].add(o)
    obj2Conc = {k: v for k, v in obj2Conc.items() if v is not None}
    bestKDispositions = best_entropy(disp2Objs, kBest)
    bestKSuperclasses = best_entropy(sups2Objs, kBest)
    obj2Disp = {o: set() for o in obj2Conc}
    obj2Sups = {o: set() for o in obj2Conc}
    for fkeys, src, trg in [(bestKDispositions, disp2Objs, obj2Disp), (bestKSuperclasses, sups2Objs, obj2Sups)]:
        for fkey in fkeys:
            for o in src[fkey]:
                if o not in trg:
                    trg[o] = set()
                trg[o].add(fkey)
    return obj2Conc, obj2Disp, obj2Sups, bestKDispositions, bestKSuperclasses


obj2Conc, obj2Disp, obj2Sups, bestKDispositions, bestKSuperclasses = prepOnto(obj2ConcsFile, 5)


def ontoLabels(allProps, subList):
    return [str(e in allProps) for e in subList]


@dataclass
class Annotation:
    annotator_idx: int
    correct: List[str]
    implausible: List[str]
    misplaced: List[str]
    object: str
    room: str

    def __post_init__(self):
        if len(self.correct) > 0:
            self.correct = [v.split("|", 1)[1] for v in self.correct]
        if len(self.misplaced) > 0:
            self.misplaced = [v.split("|", 1)[1] for v in self.misplaced]

    def generate_data(self, obj2Disp, obj2Sups, bestKDispositions, bestKSuperclasses) -> List[List[str]]:
        result = []
        for combination in itertools.product(self.correct, self.misplaced):
            result.append([*combination, *ontoLabels(obj2Disp.get(self.object, []), bestKDispositions),
                           *ontoLabels(obj2Sups.get(self.object, []), bestKSuperclasses), self.object, self.room, ])
        return result


file_location = os.path.join("data", "humanReadableAnnotations.json")


def prepData(filePath):
    data = json.loads(open(filePath).read())
    dataset = []
    for element in data:
        dataset.extend(Annotation(**element).generate_data(obj2Disp, obj2Sups, bestKDispositions, bestKSuperclasses))
    dataset = pd.DataFrame(dataset,
                           columns=["correct", "misplaced"] + bestKDispositions + bestKSuperclasses + ["object",
                                                                                                       "room"])
    return dataset


def main():

    # load the dataset
    dataset = prepData(file_location)

    # infer the variables from the dataset
    jpt_variables = infer_variables_from_dataframe(dataset, scale_continuous_types=False)

    # fit a model
    model = JPT(jpt_variables, min_samples_leaf=0.01)
    model.fit(dataset)

    # update the variables to be the variables from the "variables" folder for usability
    variable_update_map = VariableMap({jpt_variable: [v for v in variables if v.name == jpt_variable.name][0]
                                       for jpt_variable in jpt_variables})
    model.update_variables(variable_update_map)

    # query the model
    living_room_event = SimpleEvent({room: Room.living_room}).as_composite_set()
    query = SimpleEvent({v_object: Object.baseball}).as_composite_set() & living_room_event.complement()

    conditional, p = model.conditional(query)
    print(f"P: {p}")
    mode, mp = conditional.mode()
    # print(f"Mode: {mode}")
    marginal = conditional.marginal((room,))

    fig = go.Figure(marginal.plot(), marginal.plotly_layout())
    fig.show()


if __name__ == "__main__":
    main()
