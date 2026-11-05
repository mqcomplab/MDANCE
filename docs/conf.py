# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Incase the project was not installed
import os
import re
import sys
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
sys.path.insert(0, os.path.abspath('..'))
import mdance

# -- Video tutorials ------------------------------------------------------------

# ``.. youtube:: VIDEO_ID`` (with ``:title:``) embeds a video from the MDANCE playlist
PLAYLIST = 'PLPuFwYwFEee8'


class YouTube(Directive):
    required_arguments = 1
    option_spec = {'title': directives.unchanged}

    def run(self):
        title = self.options.get('title', 'Video tutorial')
        html = (f'<div class="video"><iframe src="https://www.youtube-nocookie.com/embed/{self.arguments[0]}'
                f'?list={PLAYLIST}" title="{title}" loading="lazy" allow="accelerometer; clipboard-write; '
                'encrypted-media; gyroscope; picture-in-picture; web-share" '
                'referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>')
        return [nodes.raw('', html, format='html')]


def setup(app):
    app.add_directive('youtube', YouTube)

# -- Tutorials written in Markdown --------------------------------------------

# The DIVINE, mdBIRCH, SHINE and PRISM pages are made from the tutorials in
# tutorials/*.md at build time, so there is one copy of each tutorial
GITHUB = 'https://github.com/mqcomplab/MDANCE/blob/main/'
MD_TUTORIALS = ['divine', 'mdbirch', 'shine', 'prism']
DOCS = Path(__file__).resolve().parent


def md_tutorial(name):
    text = (DOCS.parent / 'tutorials' / f'{name}.md').read_text(encoding='utf-8')
    # Table of contents: Furo shows one beside the page
    text = re.sub(r'^Table of Contents\n=+\n(?:[ \t]*- .*\n)+', '', text, flags=re.M)
    # Video tutorial: the embedded player instead of the thumbnail and its link
    text = re.sub(r'<a href="https://www\.youtube\.com/watch\?v=([\w-]+)[^"]*"><img [^>]*alt="([^"]*)"[^>]*></a>\n\n▶ [^\n]*\n',
                  r'```{youtube} \1\n:title: \2\n```\n', text)
    # Method icon: a docs image
    text = re.sub(r'<p align="center"><img src="\.\./docs/(img/[^"]+)"[^>]*></p>',
                  r'```{image} ../\1\n:width: 150px\n```', text)
    # Images in docs/: relative to docs/tutorials/
    text = text.replace('](../docs/', '](../')
    # Other repository files: link to GitHub
    text = re.sub(r'\]\(\.\./([^)]+)\)', lambda m: f']({GITHUB}{m.group(1)})', text)
    return text


for name in MD_TUTORIALS:
    (DOCS / 'tutorials' / f'{name}.md').write_text(md_tutorial(name), encoding='utf-8')

# -- Project information -----------------------------------------------------

project = 'MDANCE'
copyright = '2024 mqcomplab'                          # as in LICENSE
author = 'Lexin Chen'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx_gallery.gen_gallery',
    'sphinx_design',
    'sphinx_copybutton',
    'myst_parser',
]

# Markdown pages: link targets for headings, like GitHub (#1-input-preparations)
myst_heading_anchors = 4

# Copy code without prompts (>>>, ..., $)
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'examples',
    'filename_pattern': r'.*\.py'
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'tango'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = 'MDANCE'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'light_logo': 'mdance.png',
    'dark_logo': 'mdance-dark.png',
    'sidebar_hide_name': True,
    # Colours and fonts of the MDANCE website
    'light_css_variables': {
        'font-stack': '"Poppins", system-ui, -apple-system, "Segoe UI", sans-serif',
        'font-stack--monospace': '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
        'color-brand-primary': '#CA2D7B',
        'color-brand-content': '#0F7A90',
        'color-foreground-primary': '#0B1026',
        'color-foreground-secondary': '#454D6B',
        'color-foreground-muted': '#5F6788',
        'color-background-primary': '#FFFFFF',
        'color-background-secondary': '#F5F6FA',
        'color-background-hover': '#ECEEF5',
        'color-background-border': 'rgba(11, 16, 38, 0.10)',
        'color-sidebar-link-text': '#0B1026',
        'color-sidebar-link-text--top-level': '#0B1026',
    },
    'dark_css_variables': {
        'color-brand-primary': '#EC3F96',
        'color-brand-content': '#44E1F9',
        'color-foreground-primary': '#F3F4FB',
        'color-foreground-secondary': '#A9AED6',
        'color-foreground-muted': '#858BB8',
        'color-background-primary': '#0B1026',
        'color-background-secondary': '#0E1430',
        'color-background-hover': '#1E2650',
        'color-background-border': 'rgba(255, 255, 255, 0.08)',
        'color-code-background': '#131A3A',
        'color-sidebar-link-text': '#F3F4FB',
        'color-sidebar-link-text--top-level': '#F3F4FB',
    },
    'source_repository': 'https://github.com/mqcomplab/MDANCE/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'footer_icons': [
        {
            'name': 'Website',
            'url': 'https://mqcomplab.github.io/MDANCE/',
            'html': '<svg fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" viewBox="0 0 24 24">'
                    '<circle cx="12" cy="12" r="9"></circle><path d="M3 12h18"></path>'
                    '<path d="M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"></path></svg>',
            'class': '',
        },
        {
            'name': 'GitHub',
            'url': 'https://github.com/mqcomplab/MDANCE',
            'html': '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                    '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                    '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
                    '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15'
                    '-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82'
                    '.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 '
                    '2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            'class': '',
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = '_static/favicon.ico'
html_extra_path = ['_extra']                            # standalone pages, e.g. nani-in-60-seconds.html
html_permalinks = False                                 # no # link icons next to headings
html_css_files = [
    'https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600'
    '&family=JetBrains+Mono:wght@400;500&display=swap',
    'custom.css',                                       # last, so it overrides the theme and extensions
]
html_js_files = ['install-copy.js']
# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'mdancedoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'MDANCE.tex', 'MDANCE Documentation',
     'MDANCE', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'MDANCE', 'MDANCE Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'MDANCE', 'MDANCE Documentation',
     author, 'MDANCE', 'A flexible n-ary clustering package for all applications.',
     'Miscellaneous'),
]
