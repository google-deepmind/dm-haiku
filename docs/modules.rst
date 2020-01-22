.. currentmodule:: haiku

Modules
=======

:class:`Module` is the core abstraction provided by Haiku.

Haiku ships with many predefined modules (e.g. :class:`Linear`,
:class:`Conv2D`, :class:`BatchNorm`) and some predefined networks of modules
(e.g. :class:`nets.MLP`). If you can't find what you're looking for then we
encourage you to subclass :class:`Module` and implement your ideas.
