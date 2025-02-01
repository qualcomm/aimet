#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
#  =============================================================================

"""For meta-data collected on a computational graph for e.g. TensorFlow or PyTorch."""

from collections import OrderedDict
from aimet_common.utils import AimetLogger


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class GraphMeta:
    """ Collects the meta data for  a model. """

    def __init__(self, model_name=''):
        self._model_name = model_name
        self._nodes_by_name = OrderedDict()
        self._invocations = []

    def add_node(self, full_name, layer, name, node_type, parms):
        """Register an operation node in the graph; any node can be added only once.
        Nodes are stored in the order they are registered."""
        if full_name in self._nodes_by_name:
            raise AssertionError('%s is an existing node' % full_name)
        self._nodes_by_name[full_name] = {
            'layer': layer,
            'name': name,
            'type': node_type,
            'parms': parms,
            'uses': 0,
            'macs': 0,
        }

    def add_invocation(self, node_name, macs):
        """Register an invocation of a registered operation node and store the
        specified number of MAC's. Nodes may be invoked multiple times.
        Invocations are stored in order."""
        node = self._nodes_by_name[node_name]
        node['uses'] += 1
        node['macs'] += macs  # we accumulate mac's across multiple invocations
        self._invocations.append(node_name)

    def add_winnow_info(self, node_name, inputs_to_ignore, outputs_to_ignore):
        """ Adds the winnow information to an individual node"""
        node = self._nodes_by_name[node_name]
        node['void-inputs'] = inputs_to_ignore    # indices of input channels
        node['void-outputs'] = outputs_to_ignore  # indices of output channels


    @property
    def model_name(self):
        """ Returns the model name. """
        return self._model_name

    @property
    def num_nodes(self):
        """ Returns the number of nodes in the model."""
        return len(self._nodes_by_name)

    @property
    def num_parameters(self):
        """ Returns the number of parameters in the model. """
        return sum(node['parms'] for node in self._nodes_by_name.values())

    @property
    def num_invocations(self):
        """ Returns the number of invocations """
        return len(self._invocations)

    @property
    def num_macs(self):
        """ Returns the MACs for the model. """
        return sum(node['macs'] for node in self._nodes_by_name.values())

    def yield_node_names(self):
        """Return a generator on node names in the order they were registered."""
        for name in self._nodes_by_name.keys():
            yield name

    def yield_node_names_invoked(self, no_dups=False):
        """Return a generator on node names in the order nodes were invoked;
        nodes may occur multiple times unless 'no_dups' is True in which case
        only every first invocation is returned."""
        yielded = set()
        for name in self._invocations:
            if no_dups:
                if name in yielded:
                    continue
                yielded.add(name)
            yield name

    def get_stats_by_type(self):
        """Return statistics by node type: number of nodes in the graph,
        number of parameters, number of invocations, number of MAC's.
        Note that number of invocations is already included in number of MAC's """
        types = {}
        for node_dict in self._nodes_by_name.values():
            typ = node_dict['type']
            if typ not in types:
                types[typ] = {'nodes': 0, 'parms': 0, 'uses': 0, 'macs': 0}
            type_dict = types[typ]
            type_dict['nodes'] += 1
            type_dict['parms'] += node_dict['parms']
            type_dict['uses'] += node_dict['uses']
            type_dict['macs'] += node_dict['macs']
        return types

    def get_node_stats(self, node_name):
        """Return number of parameters, invocations and MAC's for the specified node."""
        return self._nodes_by_name[node_name]

    def dump(self):
        """ Dumps all the meta data associated with the model."""
        num_parms, num_invocations, num_macs, type_lines = self._gather_data_to_dump()

        print()
        print("Graph consists of %d nodes and %d individual parameter values." %
              (self.num_nodes, num_parms))
        print("One forward run results in %d invocations and %d MAC's.\n" %
              (num_invocations, num_macs))

        print("Instances, parameters, invocations and MAC's by node type:\n")
        for line in type_lines:
            print(line)
        print('               --- -------- -- ----------')
        print('       total : %3d %8d %2d %10d' %
              (self.num_nodes, num_parms, num_invocations, num_macs))
        print()

        print("Name, type, parameters, invocations and MACs by node invoked:\n")
        for name in self.yield_node_names_invoked(no_dups=True):
            stats = self.get_node_stats(name)
            print("%32s : %12s %8d %2d %10d" %
                  (name, stats['type'], stats['parms'], stats['uses'], stats['macs']))
        print('                                   ------------ -------- -- ----------')
        print('                           total : %12d %8d %2d %10d' %
              (self.num_nodes, num_parms, num_invocations, num_macs))
        print()

    def _gather_data_to_dump(self):
        """ gathers meta data associated with the model. """
        num_parms = 0
        num_invocations = 0
        num_macs = 0
        type_lines = []
        for typ, stats in self.get_stats_by_type().items():
            num_parms += stats['parms']
            num_invocations += stats['uses']
            num_macs += stats['macs']
            type_lines.append("%12s : %3d %8d %2d %10d" %
                              (typ, stats['nodes'], stats['parms'], stats['uses'], stats['macs']))
        return num_parms, num_invocations, num_macs, type_lines
