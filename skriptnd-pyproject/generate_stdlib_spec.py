# Copyright (c) 2017-2025 The Khronos Group Inc.
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

import argparse


def split_label(line):
    pos = line.index(' ')
    return line[:pos], line[pos+1:].strip()


def generate_module(module, output_file):
    filename = f'skriptnd/stdlib/{module}.sknd'
    with open(filename) as input_file:
        line = input_file.readline()
        while line:
            if line.startswith('#'):
                line = line[1:].lstrip()
                if line.startswith('@title'):
                    line = line[6:].lstrip()
                    label, title = split_label(line)
                    output_file.write(f'[[{label}]]\n')
                    output_file.write(f'== {title} ==\n\n')
                    output_file.write(f'The following operators are defined in the `{module}` module.\n\n')
                elif line.startswith('@section'):
                    line = line[8:].lstrip()
                    label, section = split_label(line)
                    output_file.write(f'[[{label}]]\n')
                    output_file.write(f'=== {section} ===\n\n')
                elif line.startswith('@item'):
                    item = line[5:].strip()
                    output_file.write(f'*{item}*\n\n')
                elif line.startswith('@text'):
                    text = line[5:].strip()
                    output_file.write(f'{text}\n\n')
            elif line.startswith('operator'):
                output_file.write('```\n')
                while not line.startswith('}'):
                    output_file.write(line)
                    line = input_file.readline()
                output_file.write(line)
                output_file.write('```\n\n')

            line = input_file.readline()


def main(args):
    with open(args.output_path, 'w') as output_file:
        output_file.write('[[stdlib]]\n')
        output_file.write('= Standard Library Operators =\n\n')
        output_file.write('The Standard Library defines a set of often used operators from which more complex operators '
                          'or complete graphs can be composed. The operators are grouped into various categories.\n\n')

        generate_module('layout', output_file)
        generate_module('math', output_file)
        generate_module('linalg', output_file)
        generate_module('nn', output_file)
        generate_module('image', output_file)
        generate_module('quant', output_file)
        generate_module('algo', output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str,
                        help='The output path')
    exit(main(parser.parse_args()))
