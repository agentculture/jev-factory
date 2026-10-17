"""Wire clients shared by the distillery and the backbones (stdlib only).

Nothing here imports :mod:`jev_factory.distill` or a backbone, so a backbone can speak a wire
protocol without depending on the stages that drive it.
"""
