"""Swappable continual-learning paradigms.

Each paradigm implements detection + novel-class discovery behind a common
interface (see ``base.Paradigm``) so the stream loop in ``main.py`` stays
paradigm-agnostic. Selected via ``config.paradigm``; built by
``factory.build_paradigm``.
"""
