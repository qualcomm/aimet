# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file
"""Pygments styles matching the AIMET docs palette (see _static/aimet-lumina.css).

Pygments needs literal hex values, so these mirror the CSS palette tokens by hand:
keywords violet, strings teal, numbers/builtins amber, function and class names blue.
Every token clears 4.5:1 contrast against its theme's --lumina-code-bg. Lumina paints the
code block background itself, so the style's own background only matters outside it.
"""

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    Whitespace,
)


def _styles(text, muted, keyword, string, number, name, decorator, error):
    return {
        Text: text,
        Whitespace: "",
        Error: error,
        Comment: f"italic {muted}",
        Comment.Preproc: f"noitalic {keyword}",
        Keyword: keyword,
        Keyword.Constant: number,
        Keyword.Type: number,
        Operator: text,
        Operator.Word: keyword,
        Punctuation: muted,
        Name: text,
        Name.Builtin: number,
        Name.Builtin.Pseudo: number,
        Name.Function: name,
        Name.Class: f"bold {name}",
        Name.Namespace: text,
        Name.Decorator: decorator,
        Name.Exception: f"bold {name}",
        Name.Tag: keyword,
        Name.Attribute: number,
        Name.Label: keyword,
        String: string,
        String.Doc: f"italic {string}",
        String.Escape: number,
        String.Interpol: number,
        Number: number,
        Generic.Deleted: error,
        Generic.Inserted: string,
        Generic.Heading: f"bold {text}",
        Generic.Subheading: f"bold {muted}",
        Generic.Prompt: f"bold {muted}",
        Generic.Output: muted,
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Generic.Error: error,
    }


class AimetLightStyle(Style):
    """Light-theme code style (background matches --lumina-code-bg in light mode)."""

    background_color = "#f5f6fa"
    highlight_color = "#e4e6f7"
    styles = _styles(
        text="#1e2230",
        muted="#5a6274",
        keyword="#6d28d9",
        string="#0f766e",
        number="#a1460a",
        name="#2323c4",
        decorator="#a3238f",
        error="#b91c1c",
    )


class AimetDarkStyle(Style):
    """Dark-theme code style (background matches --lumina-code-bg in dark mode)."""

    background_color = "#181b25"
    highlight_color = "#272c3d"
    styles = _styles(
        text="#e4e7ef",
        muted="#939bb0",
        keyword="#c4b5fd",
        string="#5eead4",
        number="#fcd34d",
        name="#a5b4fc",
        decorator="#f0abfc",
        error="#fca5a5",
    )
