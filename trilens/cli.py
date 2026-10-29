from __future__ import annotations

import importlib
import os
import runpy
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass
class Chain:
    parts: Sequence["Part"]

    def run(self, argv: Sequence[str]) -> None:
        if argv:
            raise SystemExit("this choice runs several modules and takes no other arguments")
        for part in self.parts:
            part.run([])


@dataclass
class Part:
    module: str
    experiment: Optional[str] = None
    args: Sequence[str] = field(default_factory=tuple)

    def run(self, argv: Sequence[str]) -> None:
        argv = [*self.args, *argv]
        if self.experiment is None:
            sys.argv = [self.module, *argv]
            try:
                runpy.run_module(self.module, run_name="__main__", alter_sys=True)
            except SystemExit as stop:
                if stop.code not in (None, 0):
                    raise
            return
        from trilens.config import load_config, show
        from trilens.utils.seeds import set_global_seed

        cfg = load_config(self.experiment, argv)
        set_global_seed(cfg.seed)
        show(cfg)
        importlib.import_module(self.module).run(cfg)


def module(name: str, *args: str) -> Part:
    return Part(name, None, args)


def configured(experiment: str, name: str) -> Part:
    return Part(name, experiment)


def chain(*parts: Part) -> Chain:
    return Chain(parts)


def _usage(title: str, parts, choose: str, env_options) -> str:
    lines = [title, ""]
    if isinstance(parts, dict):
        lines.append(f"choose {choose} from: {', '.join(parts)}" + (", all" if len(parts) > 1 else ""))
        for key, part in parts.items():
            mods = [p.module for p in part.parts] if isinstance(part, Chain) else [part.module]
            lines.append(f"  {key:28s} {' then '.join(mods)}")
    else:
        mods = [p.module for p in parts.parts] if isinstance(parts, Chain) else [parts.module]
        lines.append(f"runs {' then '.join(mods)}; any other arguments are passed on to it")
    for flag, values in (env_options or {}).items():
        lines.append(f"{flag} {{{','.join(values)}}}")
    return "\n".join(lines)


def _take_option(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        value = argv[i + 1]
        return value, argv[:i] + argv[i + 2 :]
    for i, a in enumerate(argv):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1], argv[:i] + argv[i + 1 :]
    return None, argv


def run_step(
    title: str, parts, choose: str = "--part", env_options: Optional[Dict[str, Dict[str, str]]] = None
) -> None:
    argv = sys.argv[1:]
    if argv in (["-h"], ["--help"]):
        print(_usage(title, parts, choose, env_options))
        return
    for flag, values in (env_options or {}).items():
        value, argv = _take_option(argv, flag)
        if value is not None:
            if value not in values:
                raise SystemExit(f"{flag} must be one of {sorted(values)}")
            os.environ.update(values[value])
    if not isinstance(parts, dict):
        parts.run(argv)
        return
    name, argv = _take_option(argv, choose)
    if name is None:
        raise SystemExit(f"give {choose} with one of: {', '.join(parts)}" + (", all" if len(parts) > 1 else ""))
    if name == "all":
        if argv:
            raise SystemExit(f"{choose} all takes no other arguments")
        for key, part in parts.items():
            print(f"\n=== {key} ===", flush=True)
            part.run([])
        return
    if name not in parts:
        raise SystemExit(f"unknown {choose} {name!r}; choose from: {', '.join(parts)}")
    parts[name].run(argv)
