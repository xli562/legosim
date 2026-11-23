import os
import sys
import re
import subprocess
from pathlib import Path
from graphviz import Graph
from graphviz.encoding import DEFAULT_ENCODING
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger


def replace_cpp_vars(file_path:Path|str,
                     replacements:list[tuple[str,str,int|float]]):
    """ Replaces the initialization values of variables in C++
    
    :param file_path: Path to the C++ source code
    :param replacements: List of prefixes, variable names, and new values.

    Example
    --------
    `replacements = ['static constexpr int', 'X_DIM', '10']`
    initializes `X_DIM` to 10.

    Note
    --------
    Written with LLM, not checked
    """
    path = Path(file_path)
    content = path.read_text()

    for prefix, var_name, new_val in replacements:
        prefix_pattern = r'[^\S\n]+'.join(map(re.escape, prefix.split()))
        pattern = (
            rf'{prefix_pattern}[^\S\n]+{re.escape(var_name)}'
            rf'[^\S\n]*=[^\S\n]*'
        )

        matches = list(re.finditer(pattern, content))

        if not matches:
            logger.warning(f'{var_name} not found in {path}. Skipping its replacement')
            continue

        # Process backwards to preserve match positions.
        for match in reversed(matches):
            start = match.end()
            depth = 0

            for end in range(start, len(content)):
                char = content[end]

                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                elif char == ';' and depth == 0:
                    break
            else:
                logger.warning(f'No terminating semicolon found for {var_name}')
                continue

            # Preserve the C++ declaration's indentation.
            line_start = content.rfind('\n', 0, match.start()) + 1
            indent = re.match(r'[ \t]*', content[line_start:]).group()

            value = str(new_val).replace('\n', '\n' + indent)
            content = content[:start] + value + content[end:]

    path.write_text(content)

class Mesh2DGraph(Graph):
    """ Graph of a 2D mesh network """
    def __init__(self, name = None, comment = None, filename=None, 
                 directory=None, format = None, engine = None, 
                 encoding = DEFAULT_ENCODING, graph_attr=None, 
                 node_attr=None, edge_attr=None, body=None, 
                 strict = False, *, renderer = None, formatter = None):
        super().__init__(name, comment, filename, directory, format, 
                         engine, encoding, graph_attr, node_attr, 
                         edge_attr, body, strict, renderer=renderer, 
                         formatter=formatter)
        self.x_dim = None
        self.y_dim = None

    def build_mesh(self, x_dim, y_dim):
        """ Builds the 2D mesh

        :param x_dim: Number of **columns** of the mesh
        :param y_dim: Number of **rows** of the mesh
        
        Note
        --------
        Written with LLM, not checked.
        """
        # Input check
        if x_dim <= 0 or y_dim <= 0:
            raise ValueError('A 2D mesh cannot have '
                             f'{y_dim} rows and {x_dim} columns')

        self.x_dim = x_dim
        self.y_dim = y_dim
        for r in range(y_dim):
            for c in range(x_dim):
                u = r * x_dim + c
                if c + 1 < x_dim:
                    self.edge(str(u), str(u + 1))
                if r + 1 < y_dim:
                    self.edge(str(u), str(u + x_dim))

class RunScriptBuilder:
    def __init__(self, template_path:Path):
        # Input check
        if not template_path.is_file():
            raise FileNotFoundError('Cannot find run.sh template '
                                    f'at {template_path}')

        self.source:str = template_path.read_text()

    def fill(self, placeholder_name:str, value):
        """ Replaces placeholders in the template with values
        
        :param placeholder_name: Name of the placeholder
        :param value: The replacing value
        """
        self.source = self.source.replace(f'<{placeholder_name}>', str(value))

    def write_file(self, path:Path):
        """ Write run.sh """
        # Check syntax
        result = subprocess.run(['bash', '-n'],
                                input=self.source,
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            logger.debug('This bash script does not compile: \n'
                         f'{self.source} \n'
                         f'{result.stderr}')
            raise SyntaxError('Syntax check failed prior to run.sh write.'
                              'Have all placeholders been assigned values?')

        path.write_text(self.source)
