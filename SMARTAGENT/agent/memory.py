# SMARTAGENT/agent/memory.py
import json
import os
from typing import Optional

MEMORY_FILE: str = "agent_memory.json"  # Temporary memory file

class LocalMemory:
    def _init_(self, node_id: str) -> None:
        self.node_id = node_id
        self.local_memory: dict[str, str] = {}
        self._load_from_file()  # Load on initialization

    def store(self, key: str, value: str) -> None:
        self.local_memory[key] = value
        self._save_to_file()

    def retrieve(self, key: str) -> Optional[str]:
        return self.local_memory.get(key)

    def _save_to_file(self) -> None:
        """Saves the node's local memory to a JSON file."""
        try:
            with open(MEMORY_FILE, 'r') as f:
                all_memory = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_memory = {}

        all_memory[self.node_id] = self.local_memory

        with open(MEMORY_FILE, 'w') as f:
            json.dump(all_memory, f, indent=4)

    def _load_from_file(self) -> None:
        """Loads the node's local memory from the JSON file."""
        try:
            with open(MEMORY_FILE, 'r') as f:
                all_memory = json.load(f)
                self.local_memory = all_memory.get(self.node_id, {})
        except (FileNotFoundError, json.JSONDecodeError):
            self.local_memory = {}  # Initialize if file not found or error


class GlobalMemory:  # New GlobalMemory class
    def _init_(self) -> None:
        self.global_context = ""

    def update_context(self, new_context: str) -> None:
        self.global_context = new_context

    def get_context(self) -> str:
        return self.global_context
