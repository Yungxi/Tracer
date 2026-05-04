"""
Code Parser - AST-based Python code analysis.
Extracts functions, classes, and main process code from Python source.
Supports multi-file projects with local import detection.
"""

import ast
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set


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
    source_file: Optional[str] = None  # Track which file this function is from


@dataclass
class ClassInfo:
    """Information about a parsed class."""
    name: str
    lineno: int
    end_lineno: int
    source: str
    methods: List[FunctionInfo] = field(default_factory=list)
    source_file: Optional[str] = None  # Track which file this class is from


@dataclass
class SyntaxErrorInfo:
    """Information about a syntax error in the code."""
    lineno: int
    offset: int
    message: str
    text: str  # The line with the error


@dataclass
class ParsedCode:
    """Complete parsed representation of Python source code."""
    source: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    main_statements: List[ast.stmt]
    main_source: str
    imports: List[str]
    syntax_errors: List[SyntaxErrorInfo] = field(default_factory=list)

    def has_syntax_errors(self) -> bool:
        return len(self.syntax_errors) > 0


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
            # Return ParsedCode with syntax error info instead of crashing
            error_line = self.lines[e.lineno - 1] if e.lineno and e.lineno <= len(self.lines) else ""
            syntax_error = SyntaxErrorInfo(
                lineno=e.lineno or 0,
                offset=e.offset or 0,
                message=e.msg or str(e),
                text=error_line.rstrip()
            )
            return ParsedCode(
                source=self.source,
                functions=[],
                classes=[],
                main_statements=[],
                main_source="",
                imports=[],
                syntax_errors=[syntax_error]
            )

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
    parsed = parser.parse()

    # Tag all functions/classes with their source file
    for func in parsed.functions:
        func.source_file = filepath
    for cls in parsed.classes:
        cls.source_file = filepath

    return parsed


def parse_source(source: str) -> ParsedCode:
    """Parse Python source code string and return its structure."""
    parser = CodeParser(source)
    return parser.parse()


def _detect_local_imports(parsed: ParsedCode, base_dir: str) -> List[str]:
    """
    Detect local imports that can be resolved to files in the project.

    Returns list of absolute file paths for local modules.
    """
    local_files = []

    for import_stmt in parsed.imports:
        # Parse the import statement to extract module names
        try:
            tree = ast.parse(import_stmt)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_path = _resolve_module_path(alias.name, base_dir)
                        if module_path:
                            local_files.append(module_path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_path = _resolve_module_path(node.module, base_dir)
                        if module_path:
                            local_files.append(module_path)
                    # Handle relative imports
                    if node.level > 0:
                        # Relative import - try to resolve from base_dir
                        for alias in node.names:
                            if node.module:
                                full_module = f"{node.module}.{alias.name}"
                            else:
                                full_module = alias.name
                            module_path = _resolve_module_path(full_module, base_dir)
                            if module_path:
                                local_files.append(module_path)
        except SyntaxError:
            continue

    return local_files


def _resolve_module_path(module_name: str, base_dir: str) -> Optional[str]:
    """
    Try to resolve a module name to a local file path.

    Returns the absolute path if found, None otherwise.
    """
    # Convert module.name to module/name.py or module/name/__init__.py
    parts = module_name.split('.')

    # Try as a direct .py file
    file_path = os.path.join(base_dir, *parts) + '.py'
    if os.path.isfile(file_path):
        return os.path.abspath(file_path)

    # Try as a package (__init__.py)
    init_path = os.path.join(base_dir, *parts, '__init__.py')
    if os.path.isfile(init_path):
        return os.path.abspath(init_path)

    return None


def parse_project(filepath: str, recursive: bool = True) -> ParsedCode:
    """
    Parse a Python file and optionally all its local imports recursively.

    Args:
        filepath: Path to the main Python file
        recursive: If True, also parse all local imports

    Returns:
        ParsedCode with functions/classes from all parsed files combined
    """
    filepath = os.path.abspath(filepath)
    base_dir = os.path.dirname(filepath)

    # Parse the main file
    main_parsed = parse_file(filepath)

    if not recursive or main_parsed.has_syntax_errors():
        return main_parsed

    # Track all parsed files to avoid circular imports
    parsed_files: Set[str] = {filepath}
    all_functions: List[FunctionInfo] = list(main_parsed.functions)
    all_classes: List[ClassInfo] = list(main_parsed.classes)
    all_imports: List[str] = list(main_parsed.imports)

    # Queue of files to parse
    to_parse = _detect_local_imports(main_parsed, base_dir)

    while to_parse:
        next_file = to_parse.pop(0)

        if next_file in parsed_files:
            continue

        parsed_files.add(next_file)

        try:
            parsed = parse_file(next_file)

            if parsed.has_syntax_errors():
                continue

            # Add functions and classes from this file
            all_functions.extend(parsed.functions)
            all_classes.extend(parsed.classes)
            all_imports.extend(parsed.imports)

            # Find more local imports
            file_dir = os.path.dirname(next_file)
            new_imports = _detect_local_imports(parsed, file_dir)
            for imp in new_imports:
                if imp not in parsed_files:
                    to_parse.append(imp)

        except (IOError, OSError):
            # File couldn't be read, skip it
            continue

    # Return combined ParsedCode
    # Note: main_statements and main_source only come from the entry file
    return ParsedCode(
        source=main_parsed.source,
        functions=all_functions,
        classes=all_classes,
        main_statements=main_parsed.main_statements,
        main_source=main_parsed.main_source,
        imports=all_imports,
        syntax_errors=main_parsed.syntax_errors
    )
