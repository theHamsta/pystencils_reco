# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""
Extra AST-nodes for pystencils_reco
"""

import jinja2

import pystencils
from pystencils.astnodes import Node, SympyAssignment
from pystencils.backends.cbackend import CBackend
from pystencils.data_types import create_type


class ForLoop(Node):
    def __init__(self, body, counter_symbol, min_val, max_val, step_val):
        self.min_val = min_val
        self.max_val = max_val
        self.step_val = step_val
        self._body = body
        self.counter_symbol = counter_symbol

    @property
    def symbols_defined(self):
        return {self.counter_symbol}

    @property
    def undefined_symbols(self):
        return self.body.undefined_symbols - self.symbols_defined

    @property
    def body(self):
        return self._body

    @property
    def args(self):
        return [self._body, self.min_val, self.max_val, self.step_val]


class ForEach(Node):
    def __init__(self, body, elements, iterable, counter_symbol):
        self.num_iterations = len(iterable)
        self.elements = elements
        self.iterable = iterable
        self._body = body
        self.counter_symbol = counter_symbol

    @property
    def symbols_defined(self):
        return set([*self.elements, self.counter_symbol])

    @property
    def undefined_symbols(self):
        return self.body.undefined_symbols - self.symbols_defined

    @property
    def body(self):
        return self._body

    @property
    def args(self):
        return [self._body]


class Select(Node):
    def __init__(self,
                 body,
                 what,
                 from_iterable,
                 predicate,
                 counter_symbol,
                 hit_counter_symbol,
                 max_selected,
                 compilation_target):
        self.num_iterations = len(from_iterable)
        self.elements = what
        self.iterable = from_iterable
        self._body = body
        self.counter_symbol = counter_symbol
        self.hit_counter_symbol = hit_counter_symbol
        self.predicate = predicate
        self.max_selected = max_selected
        self.predicate_symbol = pystencils.TypedSymbol('predicate', create_type('bool'))
        predicate_ast = pystencils.create_kernel(pystencils.Assignment(
            self.predicate_symbol, predicate), target=compilation_target)
        predicate_assignments = []
        stack = [predicate_ast]
        while stack:
            top = stack.pop()
            if isinstance(top, SympyAssignment):
                predicate_assignments.insert(0, top)
            stack.extend(top.args)

        self.predicate_assignments = predicate_assignments

    @property
    def symbols_defined(self):
        rtn = set()
        for p in self.predicate_assignments:
            rtn |= p.symbols_defined
        return set([*self.elements, self.counter_symbol, self.hit_counter_symbol]) | rtn

    @property
    def undefined_symbols(self):
        rtn = set()
        for p in self.predicate_assignments:
            rtn |= p.undefined_symbols
        return (self.body.undefined_symbols | rtn) - self.symbols_defined

    @property
    def body(self):
        return self._body

    @property
    def args(self):
        return [self._body, *self.predicate_assignments]


def _print_ForEach(self, node):
    template = jinja2.Template("""
{
{% for e in elements %}
{%- set last = loop.last -%}
{{ e.dtype }} {{ e.name }}_array[] = { {% for i in iterable %} {{ i[last] }} {{- "," if not loop.last }} {% endfor %} };
{% endfor %}
for ( {{ counter.dtype }} {{ counter.name }} = 0; {{ counter.name }} < {{ num_iterations }}; {{ counter.name }}++ ) {
    
    {% for e in elements -%}
    {{ e.dtype }} {{ e.name }} = {{ e.name }}_array[{{ counter.name }}];
    {% endfor %}

    {{ body | indent(8) }}

}
}
""")  # noqa

    rtn = template.render(
        elements=node.elements,
        iterable=node.iterable,
        num_iterations=node.num_iterations,
        body=self._print(node.body),
        counter=node.counter_symbol
    )
    return rtn


def _print_Select(self, node):
    template = jinja2.Template("""
{
{% for e in elements %}
{%- set last = loop.last -%}
{{ e.dtype }} {{ e.name }}_array[] = { {% for i in iterable %} {{ i[last] }} {{- "," if not loop.last }} {% endfor %} };
{% endfor %}
{{  hit_counter_symbol.dtype  }} {{  hit_counter_symbol.name  }} = 0;
for ( {{ counter.dtype }} {{ counter.name }} = 0; {{ counter.name }} < {{ num_iterations }}; {{ counter.name }}++ ) {
    if ( {{  hit_counter_symbol.name  }} == {{  max_selected  }} ) {
        break;
    }

    {% for p in predicate_assignments -%}
    {{ p }}
    {% endfor %}

    {% for e in elements -%}
    {{ e.dtype }} {{ e.name }} = {{ e.name }}_array[{{ counter.name }}];
    {% endfor %}

    if ( {{  predicate  }} ) {
        {{  hit_counter_symbol.name  }}++;
        {{ body | indent(12) }}
    }
}
}
""")  # noqa

    rtn = template.render(
        elements=node.elements,
        iterable=node.iterable,
        num_iterations=node.num_iterations,
        body=self._print(node.body),
        counter=node.counter_symbol,
        hit_counter_symbol=node.hit_counter_symbol,
        max_selected=node.max_selected,
        predicate_assignments=(self._print(a) for a in node.predicate_assignments),  # TODO
        predicate=node.predicate_symbol.name
    )
    return rtn


def _print_ForLoop(self, node):
    template = jinja2.Template("""
{
for ( {{ counter.dtype }} {{ counter.name }} = {{ min_val }}; {{ counter.name }} < {{ max_val }}; {{ counter.name }} += {{ step_val }} )
    {{ body | indent(8) }}
}
""")  # noqa
    rtn = template.render(
        min_val=node.min_val,
        max_val=node.max_val,
        step_val=node.step_val,
        body=self._print(node.body),
        counter=node.counter_symbol
    )
    return rtn


CBackend._print_Select = _print_Select
CBackend._print_ForLoop = _print_ForLoop
CBackend._print_ForEach = _print_ForEach
