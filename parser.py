"""
Code Parser - AST-based Python code analysis.
Extracts functions, classes, and main process code from Python source.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FunctionInfo:
    """Information about a parsed function."""
    name: str
    lineno: int
    end_lineno: int
    source: str
    docstring: Optional[str]
    args: List[str]
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a parsed class."""
    name: str
    lineno: int
    end_lineno: int
    source: str
    methods: List[FunctionInfo] = field(default_factory=list)


@dataclass
class ParsedCode:
    """Complete parsed representation of Python source code."""
    source: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    main_statements: List[ast.stmt]
    main_source: str
    imports: List[str]


class CodeParser:
    """Parser for Python source code using AST."""

    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines(keepends=True)

    def parse(self) -> ParsedCode:
        """Parse the source code and extract all components."""
        try:
            tree = ast.parse(self.source)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse code: {e}")

        functions = []
        classes = []
        main_statements = []
        imports = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(self._extract_function(node))
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(self._extract_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(self._get_source_segment(node))
            else:
                main_statements.append(node)

        main_source = self._extract_main_source(main_statements)

        return ParsedCode(
            source=self.source,
            functions=functions,
            classes=classes,
            main_statements=main_statements,
            main_source=main_source,
            imports=imports
        )

    def _extract_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """Extract information from a function definition."""
        source = self._get_source_segment(node)
        docstring = ast.get_docstring(node)
        args = [arg.arg for arg in node.args.args]
        decorators = [self._get_source_segment(d) for d in node.decorator_list]

        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=source,
            docstring=docstring,
            args=args,
            decorators=decorators
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract information from a class definition."""
        source = self._get_source_segment(node)
        methods = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item))

        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=source,
            methods=methods
        )

    def _get_source_segment(self, node: ast.AST) -> str:
        """Get the source code for an AST node."""
        try:
            return ast.get_source_segment(self.source, node) or ""
        except:
            # Fallback: extract by line numbers
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                start = node.lineno - 1
                end = node.end_lineno or node.lineno
                return ''.join(self.lines[start:end])
            return ""

    def _extract_main_source(self, statements: List[ast.stmt]) -> str:
        """Extract source code for main (top-level) statements."""
        if not statements:
            return ""

        segments = []
        for stmt in statements:
            segment = self._get_source_segment(stmt)
            if segment:
                segments.append(segment)

        return '\n'.join(segments)


def parse_file(filepath: str) -> ParsedCode:
    """Parse a Python file and return its structure."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    parser = CodeParser(source)
    return parser.parse()


def parse_source(source: str) -> ParsedCode:
    """Parse Python source code string and return its structure."""
    parser = CodeParser(source)
    return parser.parse()
