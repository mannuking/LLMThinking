# SMARTAGENT/agent/attention_mechanism.py
import streamlit as st
import json
from typing import Callable, Optional
from utils import parse_constraint, handle_retryable_error
from .node import Node  # Import Node class

ConstraintChecker = Callable[[str, "Node"], bool]

class AttentionMechanism:
    def _init_(self) -> None:
        self.dependency_graph: dict[str, list[Optional[str]]] = {}  # dependent -> [source1, source2, ...]
        self.constraints: dict[str, list[str]] = {}
        # self.global_context: str = "This agent decomposes complex tasks into smaller sub-tasks." # Moved to agent
        self._constraint_checkers: dict[str, ConstraintChecker] = {}
        # self.execution_count: int = 0 #Moved to Agent

    def track_dependencies(self, parent_node_id: Optional[str], current_node_id: str) -> None:
        """Tracks dependencies between nodes.  Adds current node as a dependent of the parent."""
        self.add_dependency(current_node_id, parent_node_id)

    def add_dependency(self, dependent_node_id: str, dependency_node_id: Optional[str]) -> None:
        if dependent_node_id not in self.dependency_graph:
            self.dependency_graph[dependent_node_id] = []
        if dependency_node_id and dependency_node_id not in self.dependency_graph[dependent_node_id]:
            self.dependency_graph[dependent_node_id].append(dependency_node_id)


    def get_dependencies(self, node_id: str) -> list[Optional[str]]:
        """Returns a list of nodes that the given node depends on."""
        return self.dependency_graph.get(node_id, [])


    def add_constraint(self, node_id: str, constraint: str) -> None:
        if node_id not in self.constraints:
            self.constraints[node_id] = []
        if constraint not in self.constraints[node_id]:
            self.constraints[node_id].append(constraint)

    def get_constraints(self, node_id: str) -> list[str]:
        return self.constraints.get(node_id, [])

    def update_constraint(self, node_id: str, constraint_index: int, new_constraint: str) -> None:
        if node_id in self.constraints and 0 <= constraint_index < len(self.constraints[node_id]):
            self.constraints[node_id][constraint_index] = new_constraint

    def remove_constraint(self, node_id: str, constraint_index: int) -> None:
        if node_id in self.constraints and 0 <= constraint_index < len(self.constraints[node_id]):
            del self.constraints[node_id][constraint_index]

    def propagate_constraints(self, parent_node_id: str) -> None:
        """Propagates constraints from a parent node to all its children."""
        parent_constraints = self.get_constraints(parent_node_id)
        if parent_node_id in st.session_state.node_lookup:
            for child_id in st.session_state.node_lookup[parent_node_id].child_ids:
                for constraint in parent_constraints:
                    # Avoid adding duplicate constraints
                    if constraint not in self.get_constraints(child_id):
                        self.add_constraint(child_id, constraint)


    def _summarize_global_context(self) -> None:  # Now uses GlobalMemory
        """Summarizes the global context using the LLM."""
        prompt = f"""Summarize the following global context into a concise JSON object with a single field "summary":\n\n{st.session_state.agent.global_memory.get_context()}"""
        try:
            # response = st.session_state.llm.generate_content( #Made changes here
            response = st.session_state.agent.llm.generate_content(
                prompt,
                generation_config=st.session_state.llm_config #Using llm config
            )
            response_content = response.text

            if response_content is not None and isinstance(response_content, str):
                summary_json = json.loads(response_content)
                st.session_state.agent.global_memory.update_context(summary_json.get("summary", "Error: Could not summarize global context."))
            else:
                new_context = "Error: LLM returned None or non-string response for global context summarization."
                st.session_state.agent.global_memory.update_context(new_context)
                st.error(new_context)

        except (json.JSONDecodeError, KeyError, Exception) as e:
            new_context = f"Error during summarization: {e}"
            st.session_state.agent.global_memory.update_context(new_context)
            st.error(new_context)

    def summarize_node(self, node: "Node") -> None: # Added type hint
        """Summarizes a completed node and updates the global context."""
        prompt = f"""Summarize the following task and its result concisely into a JSON object with two fields "task_summary" and "result_summary":

Task: {node.task_description}

Result: {node.output}
"""
        try:
            # response = st.session_state.llm.generate_content(
            response = st.session_state.agent.llm.generate_content(
                prompt,
                generation_config= st.session_state.llm_config
            )
            response_content = response.text

            if response_content is not None and isinstance(response_content, str):
                summary_json = json.loads(response_content)
                task_summary = summary_json.get("task_summary", "Task summary not available.")
                result_summary = summary_json.get("result_summary", "Result summary not available.")
                new_context = f"\n- Node {node.node_id} ({node.status}): Task: {task_summary}, Result: {result_summary}"
                st.session_state.agent.global_memory.update_context(st.session_state.agent.global_memory.get_context() + new_context) #Append
            else:
                new_context = f"\n- Node {node.node_id} ({node.status}): Error: LLM returned None or non-string response."
                st.session_state.agent.global_memory.update_context(st.session_state.agent.global_memory.get_context() + new_context)
                st.error(f"Error during summarization of node {node.node_id}: LLM returned None or non-string.")

        except (json.JSONDecodeError, KeyError, Exception) as e:
            new_context = f"\n- Node {node.node_id} ({node.status}): Error during summarization: {e}"
            st.session_state.agent.global_memory.update_context(st.session_state.agent.global_memory.get_context() + new_context)
            st.error(f"Error during summarization of node {node.node_id}: {e}")

        st.session_state.agent.execution_count += 1
        if st.session_state.agent.execution_count % st.session_state.agent.global_context_summary_interval == 0: #Using agent's variables
            self._summarize_global_context()

    def get_global_context(self) -> str:
        """Retrieves the current global context."""
        return st.session_state.agent.global_memory.get_context()

    def add_constraint_checker(self, constraint_type: str, checker: ConstraintChecker) -> None:
        """Registers a constraint checker function."""
        self._constraint_checkers[constraint_type] = checker

    def _check_json_format(self, constraint_value: str, node: "Node") -> bool:  # Added type hint
        """Constraint checker: Checks if the output is valid JSON."""
        try:
            json.loads(node.output)
            return True
        except json.JSONDecodeError:
            node.status = "failed"
            node.error_message = f"Constraint violated: Output must be in JSON format. Output: {node.output}"
            return False

    def _check_contains_word(self, constraint_value: str, node: "Node") -> bool:  # Added type hint
        """Constraint checker: Checks if the output contains a specific word."""
        if constraint_value in node.output:
            return True
        node.status = "failed"
        node.error_message = f"Constraint violated: Output must contain '{constraint_value}'. Output: {node.output}"
        return False

    def _check_max_length(self, constraint_value: str, node: "Node") -> bool:  # Added type hint
        """Constraint checker: Checks if the output length is within a limit."""
        try:
            max_length = int(constraint_value)
            if len(node.output) <= max_length:
                return True
            node.status = "failed"
            node.error_message = f"Constraint violated: Output must be no more than {max_length} characters. Output: {node.output}"
            return False
        except ValueError:
            node.status = "failed"
            node.error_message = f"Constraint violated: Invalid max length value '{constraint_value}'"
            return False

    def check_constraints(self, node: "Node") -> bool:  # Added type hint
        """Checks all constraints for a given node."""
        for constraint in self.get_constraints(node.node_id):
            constraint_type, constraint_value = parse_constraint(constraint)
            checker = self._constraint_checkers.get(constraint_type)
            if checker:
                if not checker(constraint_value, node):
                    return False  # Constraint failed
        return True  # All constraints passed

    def remove_node(self, node_id: str) -> None:
        """Removes a node and its associated data from the attention mechanism."""
        self.dependency_graph.pop(node_id, None)
        self.constraints.pop(node_id, None)

        # Also remove as a dependency for other nodes
        for dependent, sources in self.dependency_graph.items():
            if node_id in sources:
                sources.remove(node_id)
