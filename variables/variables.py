"""
This file contains the variables used in the probabilistic model.
"""
from probabilistic_model.learning.jpt.variables import Symbolic

from .enums import *

correct = Symbolic("correct", Location)

contact = Symbolic("dfl:contact.n.wn.contact..forceful.Patient", Boolean)

break_into = Symbolic("dfl:break_into.v.wn.change.Patient", Boolean)

comestible = Symbolic("dfl:comestible", Boolean)

artifact = Symbolic("dfl:container.n.wn.artifact", Boolean)

instrumentality = Symbolic("dfl:device.n.wn.artifact..instrumentality", Boolean)

graspable = Symbolic("dfl:Graspability", Boolean)

separate = Symbolic("dfl:separate.v.wn.contact..forceful.Patient", Boolean)

concrete = Symbolic("dfl:serve.v.wn.consumption..concrete.Theme", Boolean)

solid = Symbolic("dfl:solid", Boolean)

transport = Symbolic("dfl:transport.v.wn.contact.Theme", Boolean)

physical_body = Symbolic("dul:PhysicalBody", Boolean)

misplaced = Symbolic("misplaced", Location)

v_object = Symbolic("object", Object)

room = Symbolic("room", Room)

variables = [correct, contact, break_into, comestible, artifact, instrumentality, graspable, separate, concrete, solid,
             transport, physical_body, misplaced, v_object, room]
