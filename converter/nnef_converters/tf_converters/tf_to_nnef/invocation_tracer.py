# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function


import inspect
import sys
import traceback

from ...common import utils


class Invocation(object):
    def __init__(self, func, args, results, stack, nesting_level):
        self.func = func
        self.args = args
        self.results = results
        self.stack = stack
        self.nesting_level = nesting_level
        self.extra = {}


class InvocationTracer(object):

    def __init__(self, functions, invocation_cls=Invocation, functions_for_statistics_only=None):
        if functions_for_statistics_only is None:
            functions_for_statistics_only = []

        self.invocation_cls = invocation_cls
        self.invocations = list()
        self.invocation_stack = list()
        self.function_by_qualified_name = {}
        self.function_by_qualified_name_for_statistics = {}
        self.supported_func_names = set()
        self.call_count = {}

        for func in functions:
            undecorated = utils.get_function_without_decorators(func)
            self.function_by_qualified_name[undecorated.__module__ + '.' + undecorated.__name__] = func
            self.function_by_qualified_name_for_statistics[undecorated.__module__ + '.' + undecorated.__name__] = func
            self.supported_func_names.add(undecorated.__name__)
            self.call_count[func] = 0

        for func in functions_for_statistics_only:
            undecorated = utils.get_function_without_decorators(func)
            self.function_by_qualified_name_for_statistics[undecorated.__module__ + '.' + undecorated.__name__] = func
            self.supported_func_names.add(undecorated.__name__)
            self.call_count[func] = 0

    def __call__(self, frame, event, result):
        func_name = frame.f_code.co_name
        if func_name == '__init__':
            result = frame.f_locals.get('self')
            if result is not None:
                func_name = result.__class__.__name__

        if func_name not in self.supported_func_names:
            return None

        mod = inspect.getmodule(frame)
        if mod is None:
            return None

        func_name = mod.__name__ + '.' + func_name

        if event == 'call' and func_name in self.function_by_qualified_name_for_statistics:
            self.call_count[self.function_by_qualified_name_for_statistics[func_name]] += 1

        if func_name not in self.function_by_qualified_name:
            return None

        if event == 'call':
            func = self.function_by_qualified_name.get(func_name)
            if func is not None:
                arg_values = inspect.getargvalues(frame)
                invocation = self.invocation_cls(
                    func=func,
                    args={key: value for (key, value) in arg_values.locals.items() if key in arg_values.args},
                    results=None,
                    stack=None,
                    nesting_level=len(self.invocation_stack)
                )
                invocation.extra["frame"] = frame
                self.invocation_stack.append(invocation)
            return self  # This allows us to trace the "return" event of the same function
        elif event == 'return':
            if self.invocation_stack:
                invocation = self.invocation_stack[-1]
                if invocation.extra["frame"] == frame:
                    self.invocation_stack.pop()
                    if result is not None:
                        del invocation.extra["frame"]
                        if isinstance(result, list):
                            result = list(result)
                        if isinstance(result, dict):
                            result = dict(result)
                        invocation.results = result if isinstance(result, tuple) else (result,)
                        invocation.stack = traceback.extract_stack(frame.f_back)
                        self.invocations.append(invocation)
                    else:
                        utils.print_error("{} returned without result.".format(invocation.func))
            return None

    @staticmethod
    def get_result_and_invocations(func, functions, invocation_cls=Invocation):
        sys_trace = sys.gettrace()
        trace = InvocationTracer(functions, invocation_cls)
        sys.settrace(trace)
        result = func()
        sys.settrace(sys_trace)
        return result, trace.invocations

    @staticmethod
    def get_result_and_invocations_and_call_count(func, functions, functions_for_statistics_only,
                                                  invocation_cls=Invocation):
        sys_trace = sys.gettrace()
        trace = InvocationTracer(functions, invocation_cls, functions_for_statistics_only=functions_for_statistics_only)
        sys.settrace(trace)
        result = func()
        sys.settrace(sys_trace)
        return result, trace.invocations, trace.call_count
