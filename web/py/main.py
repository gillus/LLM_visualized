# PyScript entry point for neural-net.html (runs on MicroPython in the page).
#
# Loads the house data, builds the Lab from nnlab.py, and hands a small API to
# the page's JavaScript as window.NNPY. Each call returns a JSON string: the
# JS side only draws what Python computed.
import json

from pyscript import window
from pyscript.ffi import create_proxy

import nnlab

with open("house_prices.json") as f:
    LAB = nnlab.Lab(json.load(f))


def line_state():
    return json.dumps(LAB.line_state())


def line_reset():
    LAB.line_reset()
    return line_state()


def line_set(slope, base):
    LAB.line_set(float(slope), float(base))
    return line_state()


def line_step(n):
    LAB.line_step(int(n))
    return line_state()


def nn_state(direction):
    return json.dumps(LAB.nn_state(int(direction)))


def nn_reset(arch, direction):
    LAB.nn_reset(str(arch))
    return nn_state(direction)


def nn_step(n, direction):
    LAB.nn_step(int(n))
    return nn_state(direction)


api = window.Object.new()
for name, fn in [("line_state", line_state), ("line_reset", line_reset),
                 ("line_set", line_set), ("line_step", line_step),
                 ("nn_state", nn_state), ("nn_reset", nn_reset),
                 ("nn_step", nn_step)]:
    setattr(api, name, create_proxy(fn))
window.NNPY = api
window.dispatchEvent(window.CustomEvent.new("nnpy-ready"))
