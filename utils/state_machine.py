from typing import Dict, Optional


class State:
    def __init__(self, name: str) -> None:
        self.name = name

    def on_enter(self, **kwargs) -> None:
        return None

    def on_exit(self, **kwargs) -> None:
        return None

    def update(self, **kwargs) -> Optional[str]:
        return None


class StateMachine:
    def __init__(self, initial_state: Optional[str] = None) -> None:
        self._states: Dict[str, State] = {}
        self._current: Optional[State] = None
        self._initial_state = initial_state

    @property
    def current(self) -> Optional[State]:
        return self._current

    def add_state(self, state: State) -> None:
        self._states[state.name] = state

    def set_state(self, name: str, **kwargs) -> None:
        if name not in self._states:
            raise KeyError(f"State '{name}' not registered")
        if self._current is not None:
            self._current.on_exit(**kwargs)
        self._current = self._states[name]
        self._current.on_enter(**kwargs)

    def start(self, **kwargs) -> None:
        if self._current is None:
            if self._initial_state is None:
                raise ValueError("No initial state set")
            self.set_state(self._initial_state, **kwargs)

    def update(self, **kwargs) -> None:
        if self._current is None:
            self.start(**kwargs)
        if self._current is None:
            return
        next_state = self._current.update(**kwargs)
        if next_state:
            self.set_state(next_state, **kwargs)
