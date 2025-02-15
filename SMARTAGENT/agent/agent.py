# SMARTAGENT/agent/agent.py
import streamlit as st
from attention_mechanism import AttentionMechanism
from node import Node
from memory import LocalMemory, GlobalMemory  # Import the memory classes
from typing import Optional
# from .llm import LLM, GeminiLLM # Import when you create llm.py
import json
import os


# SMARTAGENT/agent/agent.py
class Agent:
    def _init_(self, llm, llm_config, global_context: str = "This agent decomposes complex tasks.") -> None:
        self.attention_mechanism = AttentionMechanism()
        self.global_memory = GlobalMemory()
        self.global_memory.update_context(global_context)
        self.llm = llm
        self.llm_config = llm_config  # Store the config
        self.execution_count = 0
        self.max_retries = 3
        self.retry_delay = 2
        self.global_context_summary_interval = 5
        self.max_depth = 5

        
    def run(self, task_description: str, initial_constraints: Optional[list[str]] = None) -> None:
        """Main entry point for running the agent."""
        self.reset_agent()  # Reset state before starting
        root_node = self.create_root_node(task_description, initial_constraints)
        st.session_state.root_node_id = root_node.node_id # Store for display
        # self.scheduler.execute_tree(root_node) # When you have the scheduler
        self.execute_tree(root_node)  # Use the simpler version for now
        st.success("Agent execution complete (or stopped by user)!")

    def reset_agent(self) -> None:
        """Resets the agent's state."""
        st.session_state.clear() # Clear EVERYTHING in session state
        if os.path.exists("agent_memory.json"):  # Use const, or better, a method in memory.py
            os.remove("agent_memory.json")
        self.setup_agent() # Re-initialize things that belong in session state

    def setup_agent(self) -> None:
        """Initializes the agent's components."""
        st.session_state.node_lookup = {} # Initialize here!!
        self.attention_mechanism.add_constraint_checker("format", self.attention_mechanism._check_json_format)
        self.attention_mechanism.add_constraint_checker("contains", self.attention_mechanism._check_contains_word)
        self.attention_mechanism.add_constraint_checker("max_length", self.attention_mechanism._check_max_length)
        st.session_state.attention_mechanism = self.attention_mechanism # Store in session state
        # Store the agent itself in session state!  VERY IMPORTANT
        st.session_state.agent = self
        st.session_state.llm = self.llm # Store LLM
        st.session_state.llm_config = self.llm_config


    def create_root_node(self, task_description: str, initial_constraints: Optional[list[str]] = None) -> Node:
        """Creates the root node of the task tree."""
        new_node = Node(parent_id=None, task_description=task_description, depth=0)
        st.session_state.node_lookup[new_node.node_id] = new_node
        if initial_constraints:
            for constraint in initial_constraints:
                st.session_state.attention_mechanism.add_constraint(new_node.node_id, constraint)
        st.session_state.attention_mechanism.track_dependencies(None, new_node.node_id)  # Corrected dependency
        return new_node

    def create_child_node(self, parent_node: Node, task_description: str, depth: int) -> Node:
        """Creates a child node and links it to the parent."""
        new_node = Node(parent_id=parent_node.node_id, task_description=task_description, depth=depth)
        st.session_state.node_lookup[new_node.node_id] = new_node
        parent_node.add_child(new_node)
        st.session_state.attention_mechanism.track_dependencies(parent_node.node_id, new_node.node_id)  # Corrected dependency
        return new_node

    def execute_tree(self, node: Node) -> None:
        """Executes the task tree recursively, with human-in-the-loop."""

        if node.status not in ("completed", "failed", "overridden"):
            node.execute()

        if node.status == "failed":
            st.error(f"ERROR: Node {node.node_id} failed: {node.error_message} (Parent: {node.parent_id})")
            return

        # UI and Human-in-the-Loop
        st.write(f"## Review Node: {node.node_id}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"*Task:* {node.task_description}")
            st.write(f"*Status:* {node.status}")
            st.write("*Constraints:*")
            for constraint in st.session_state.attention_mechanism.get_constraints(node.node_id):
                st.write(f"- {constraint}")

            # Display dependencies
            st.write("*Dependencies:*")
            for dep_id in st.session_state.attention_mechanism.get_dependencies(node.node_id):
                dep_node = st.session_state.node_lookup.get(dep_id)
                if dep_node:  # Check if the dependent node exists
                    st.write(f"- Depends on: {dep_node.node_id} ({dep_node.task_description})")
                else:
                    st.write(f"- Depends on: Node {dep_id} (NOT FOUND - check dependency graph!)")


        with col2:
            st.write("*Output:*")
            st.write(node.output)
            if node.error_message:
                st.write(f"*Error:* {node.error_message}")
            with st.expander("Local Memory"):
                for key, value in node.local_memory.items():
                    st.write(f"*   *{key}:* {value}")

        with st.form(key=f'review_form_{node.node_id}'):
            st.write("### Override Output (Optional)")
            override_output = st.text_area("Enter overridden output:", value=node.output, height=150)

            st.write("### Modify Constraints (Optional)")
            current_constraints = st.session_state.attention_mechanism.get_constraints(node.node_id)
            new_constraints = []
            for i, constraint in enumerate(current_constraints):
                cols = st.columns([8, 2])
                with cols[0]:
                    updated_constraint = st.text_input(f"Constraint {i + 1}:", value=constraint, key=f"constraint_{node.node_id}_{i}")
                    new_constraints.append(updated_constraint)
                with cols[1]:
                    if st.form_submit_button(label=f"Delete", key=f"delete_constraint_{node.node_id}_{i}"):
                        st.session_state.attention_mechanism.remove_constraint(node.node_id, i)
                        st.rerun()  # Key for immediate update

            cols = st.columns([8, 2])
            with cols[0]:
                new_constraint = st.text_input("Add New Constraint:", key=f"new_constraint_{node.node_id}")
            with cols[1]:
                if st.form_submit_button(label="Add", key=f"add_constraint_{node.node_id}"):
                    if new_constraint.strip():
                        st.session_state.attention_mechanism.add_constraint(node.node_id, new_constraint.strip())
                        st.rerun()  # Key for immediate update

            st.write("### Manual Decomposition (Optional)")
            manual_subtasks = st.text_input("Enter sub-tasks (semicolon-separated):", key=f"manual_decompose_{node.node_id}")

            col1, col2, col3 = st.columns(3)
            with col1:
                approve_continue = st.form_submit_button("Approve & Continue")
            with col2:
                stop_execution = st.form_submit_button("Stop Execution")
            with col3:
                re_execute = st.form_submit_button("Re-execute Node")

        if approve_continue:
            # Handle output override
            if override_output != node.output:
                node.output = override_output
                node.status = "overridden"
                # Clear children if output is overridden
                for child_id in node.child_ids:
                    if child_id in st.session_state.node_lookup:
                        st.session_state.node_lookup[child_id].status = "overridden"
                        st.session_state.attention_mechanism.remove_node(child_id)  # Clean up dependencies
                        #del st.session_state.node_lookup[child_id]  # Remove from lookup after removing from attention
                node.child_ids = []  # Clear child IDs

            # Update constraints
            for i, updated_constraint in enumerate(new_constraints):
                st.session_state.attention_mechanism.update_constraint(node.node_id, i, updated_constraint)

            # Handle manual decomposition
            if manual_subtasks.strip():
                sub_tasks = [task.strip() for task in manual_subtasks.split(';') if task.strip()]
                for task_description in sub_tasks:
                    self.create_child_node(node, task_description, node.depth + 1)
                node.remove_node()

            # If not failed/overridden, mark as completed and summarize (if no children)
            if node.status != "failed" and node.status != "overridden":
                node.status = "completed"
                if not node.child_ids: # Only summarize leaf nodes, or overridden
                    st.session_state.attention_mechanism.summarize_node(node)


            # Continue execution with children (if any)
            if node.status == "completed" and node.child_ids:
                st.session_state.attention_mechanism.propagate_constraints(node.node_id)  # Propagate before execution
                for child_id in node.child_ids:
                    if child_id in st.session_state.node_lookup:
                        child_node = st.session_state.node_lookup[child_id]
                        self.execute_tree(child_node)  # Recursive call

        if stop_execution:
            st.warning("Execution stopped by user.")
            return

        if re_execute:
            node.status = "pending"  # Reset status
            node.output = ""  # Clear output
            node.error_message = ""  # Clear error
            st.rerun()  # Force a rerun to re-execute


# SMARTAGENT/agent/attention_mechanism.py
import streamlit as st
import json
from typing import Callable, Optional
from .utils import parse_constraint, handle_retryable_error

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


# SMARTAGENT/agent/node.py

import streamlit as st
import re
import json
import time
import uuid  # Import uuid
from typing import Optional
from .memory import LocalMemory
from .utils import handle_retryable_error

class Node:
    def _init_(self, parent_id: Optional[str], task_description: str, depth: int) -> None:
        self.node_id: str = str(uuid.uuid4())
        self.parent_id: Optional[str] = parent_id
        self.child_ids: list[str] = []
        self.task_description: str = task_description
        self.status: str = "pending"  # pending, running, completed, failed, overridden
        self.output: str = ""
        self.local_memory = LocalMemory(self.node_id)  # Use LocalMemory class
        self.error_message: str = ""
        self.depth: int = depth

    def store_in_memory(self, key: str, value: str) -> None:
        """Stores a key-value pair in the node's local memory."""
        self.local_memory.store(key, value)

    def retrieve_from_memory(self, key: str) -> Optional[str]:
        """Retrieves a value from the node's local memory by key."""
        return self.local_memory.retrieve(key)

    def add_child(self, child_node: "Node") -> None:
        """Adds a child node ID to the list of children."""
        self.child_ids.append(child_node.node_id)

    def get_parent_node(self) -> Optional["Node"]:
        """Retrieves the parent node object from the session state."""
        return st.session_state.node_lookup.get(self.parent_id)

    def _process_store_command(self, line: str) -> None:
        """Processes a STORE command from the LLM output."""
        match = re.match(r"STORE\s+(.+?)\s+(.+)", line)
        if match:
            key, value = match.groups()
            self.store_in_memory(key, value)

    def _process_retrieve_command(self, line: str) -> Optional[str]:
        """Processes a RETRIEVE command from the LLM output."""
        match = re.match(r"RETRIEVE\s+(.+)", line)
        if match:
            key = match.group(1)
            return self.retrieve_from_memory(key)
        return None

    def _process_query_parent_command(self, line: str) -> Optional[str]:
        """Processes a QUERY_PARENT command."""
        match = re.match(r"QUERY_PARENT\s+(.+)", line)
        if match:
            key = match.group(1)
            parent_node = self.get_parent_node()
            return parent_node.retrieve_from_memory(key) if parent_node else None
        return None

    def _process_decompose_command(self, line: str, llm_output: str) -> bool:
        """Processes a DECOMPOSE command, creating child nodes."""
        match = re.match(r"DECOMPOSE\s+(.+)", line)
        if match:
            sub_tasks_str = match.group(1)
            sub_tasks = [t.strip() for t in sub_tasks_str.split(';') if t.strip()]
            if (self.depth + 1) > st.session_state.agent.max_depth: # Using agent's max depth
                self.status = "failed"
                self.error_message = f"Max depth of {st.session_state.agent.max_depth} reached. Cannot Decompose Further."
                return False
            if not sub_tasks:
                self.status = "failed"
                self.error_message = "DECOMPOSE command used but no subtasks provided."
                return False

            constraints = self.extract_constraints(llm_output)
            for task_description in sub_tasks:
                child_node = st.session_state.agent.create_child_node(self, task_description, self.depth + 1)
                for constraint in constraints:
                    st.session_state.attention_mechanism.add_constraint(child_node.node_id, constraint)

            # Propagate constraints and remove parent node
            st.session_state.attention_mechanism.propagate_constraints(self.node_id)
            self.remove_node() # Remove Parent
            return True  # Decomposition successful
        return False  # No decomposition command found

    def extract_constraints(self, llm_output: str) -> list[str]:
        """Extracts constraints from the LLM output (if present)."""
        constraints = []
        try:
            match = re.search(r"Constraints:\s*json\s*([\s\S]*?)\s*", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                constraints_data = json.loads(json_str)
                for item in constraints_data:
                    if isinstance(item, dict) and "constraint" in item:
                        constraints.append(item["constraint"])
        except json.JSONDecodeError:
            st.error("Error decoding JSON in extract_constraints.  Check LLM output.")
            self.error_message = "Error: Invalid JSON format for constraints."
            self.status = "failed"

        return constraints

    def _extract_inter_node_dependencies(self, llm_output: str) -> None:
        """
        Extracts and stores inter-node dependencies from the LLM output.
        Looks for a specific JSON format indicating dependencies.
        """
        try:
            match = re.search(r"Dependencies:\s*json\s*([\s\S]*?)\s*", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                dependencies_data = json.loads(json_str)
                for dep in dependencies_data:
                    if isinstance(dep, dict) and "dependent_node" in dep and "source_node" in dep:
                        dependent_node_str = dep["dependent_node"]  # This node needs something
                        source_node_str = dep["source_node"]       # This node provides something

                        # Convert node IDs to UUIDs, handle potential errors
                        try:
                            dependent_node = str(uuid.UUID(dependent_node_str))
                            source_node = str(uuid.UUID(source_node_str))
                        except ValueError:
                            st.error(f"Invalid UUID format in dependencies: {dependent_node_str} or {source_node_str}")
                            continue  # Skip this dependency

                        # Crucial: Add dependency to the AttentionMechanism
                        st.session_state.attention_mechanism.add_dependency(dependent_node, source_node)
                        # Also store it in local memory for visibility/debugging
                        self.store_in_memory(f"dependency_{dependent_node}", source_node)

        except json.JSONDecodeError:
            st.error("Error decoding JSON in _extract_inter_node_dependencies. Check LLM output.")
            self.error_message = "Error: Invalid JSON format for dependencies."
            self.status = "failed"
        except Exception as e:  # Catch other potential errors
            st.error(f"Error in _extract_inter_node_dependencies: {e}")
            self.error_message = f"Error processing dependencies: {e}"
            self.status = "failed"

    def process_llm_output(self, llm_output: str) -> None:
        """Processes the raw LLM output, handling commands and storing information."""
        lines = llm_output.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("STORE"):
                self._process_store_command(line)
            elif line.startswith("RETRIEVE"):
                retrieved_value = self._process_retrieve_command(line) # Get value
            elif line.startswith("QUERY_PARENT"):
                parent_value = self._process_query_parent_command(line)  # Get value
            elif line.startswith("DECOMPOSE"):
                if self._process_decompose_command(line, llm_output):
                    return  # Exit if decomposition occurred

        self._extract_inter_node_dependencies(llm_output) # Extract dependencies
        self.store_in_memory("raw_llm_output", llm_output) # Store raw output

    def build_prompt(self) -> str:
        """Constructs the prompt for the LLM."""
        constraints = st.session_state.attention_mechanism.get_constraints(self.node_id)
        constraints_str = "\n".join([f"- {c}" for c in constraints]) if constraints else "None"
        global_context = st.session_state.attention_mechanism.get_global_context()

        # Fetch inter-node dependencies and format them more robustly.
        dependencies = st.session_state.attention_mechanism.get_dependencies(self.node_id)
        dependencies_str = ""
        if dependencies:
            dependencies_str = "\nInter-Node Dependencies:\n"
            for dep_node_id in dependencies:
                dep_node = st.session_state.node_lookup.get(dep_node_id)
                if dep_node:
                    dependencies_str += f"- Requires output from Node {dep_node_id} (Task: {dep_node.task_description})\n"
                else:
                    dependencies_str += f"- Requires output from Node {dep_node_id} (Node not found - check dependency graph!)\n"


        prompt = f"""You are a helpful assistant tasked with solving the following:

Task: {self.task_description}

Constraints:
{constraints_str}

{dependencies_str}

Global Context:
{global_context}

You have access to local memory.  Use the following commands to interact with it:
- STORE <key> <value>: Store information in memory.  Example: STORE tools search_engine
- RETRIEVE <key>: Retrieve information from memory. Example: RETRIEVE tools
- QUERY_PARENT <key>: Retrieve information from the parent's memory. Example: QUERY_PARENT overall_goal

If the task is even slighly big for one llm call to solve directly, you can decompose it into smaller sub-tasks:
- DECOMPOSE <task_1>; <task_2>; ... ; <task_n>: Break down the task into sub-tasks.  Example: DECOMPOSE Find contact info; Email the contact; Log the email
  You may also provide CONSTRAINTS for the child nodes in JSON Format.
  Constraints:
    json
    [
        {{"constraint": "format: json"}},
        {{"constraint": "max_length: 200"}}
    ]
    
    
  You may also provide inter-node dependencies.  Crucially, these dependencies must use the UUIDs of the nodes. Example:
  Dependencies:
    json
        [
            {{"dependent_node": "123e4567-e89b-12d3-a456-426614174000", "source_node": "123e4567-e89b-12d3-a456-426614174001"}}
        ]
    

Previous Memory Operations:
{self.local_memory.retrieve("memory_operations_history", "")}

Provide your solution. If you can solve the task directly, provide the solution in JSON format. If not, use DECOMPOSE. The final output MUST be valid JSON.
"""
        return prompt

    def execute(self) -> None:
        """Executes the node's task using the LLM."""
        # self._load_from_file()  # Now handled by LocalMemory init
        self.status = "running"
        prompt = self.build_prompt()

        for attempt in range(st.session_state.agent.max_retries):  # Use agent's max_retries
            try:
                # response = st.session_state.llm.generate_content( # Made changes here
                response = st.session_state.agent.llm.generate_content(
                    prompt,
                    #generation_config= st.session_state.agent.llm_config # OLD - WRONG
                    generation_config= st.session_state.llm_config  # NEW - CORRECT (for option 2)
                )
                llm_output = response.text

                try:
                    json.loads(llm_output)  # Basic JSON validation
                except json.JSONDecodeError:
                    raise ValueError(f"LLM output is not valid JSON: {llm_output}")

                self.output = llm_output
                self.process_llm_output(llm_output)

                if not st.session_state.attention_mechanism.check_constraints(self):
                    return  # Constraint check failed
                break  # Success! Exit the retry loop

            except Exception as e:
                if handle_retryable_error(self, attempt, e):
                    return  # Max retries exceeded or non-retryable error

        if self.status == "running":  # If we exited the loop without failing
            self.status = "completed"
            if not self.child_ids:
                st.session_state.attention_mechanism.summarize_node(self)
            #self.remove_node() # Remove Parent # removing this as we are now removing node when its completed or overridden

    def remove_node(self) -> None:
        """Removes the node from the system (lookup and attention mechanism)."""
        st.session_state.node_lookup.pop(self.node_id, None)
        st.session_state.attention_mechanism.remove_node(self.node_id)

# SMARTAGENT/agent/utils.py
import time
import re
import streamlit as st
import graphviz

def parse_constraint(constraint_string: str) -> tuple[str, str]:
    """Parses a constraint string into its type and value."""
    try:
        constraint_type, constraint_value = constraint_string.split(":", 1)
        return constraint_type.strip(), constraint_value.strip()
    except ValueError:
        return "unknown", constraint_string  # Default type if parsing fails

def handle_retryable_error(node: "Node", attempt: int, error: Exception) -> bool:  # Added type hint
    """Handles retryable LLM API errors (rate limits, timeouts)."""
    if "429" in str(error) or "rate limit" in str(error).lower() or "timeout" in str(error).lower():
        if attempt < st.session_state.agent.max_retries - 1:  # Use agent's max_retries
            delay = st.session_state.agent.retry_delay * (2 ** attempt)  # Exponential backoff, use agent's retry delay
            st.toast(f"{type(error)._name_}. Retrying in {delay} seconds (attempt {attempt + 2}/{st.session_state.agent.max_retries})...")
            time.sleep(delay)
            return False  # Retry
        else:
            node.status = "failed"
            node.error_message = f"LLM API Error: Max retries exceeded: {error}"
            st.error(node.error_message)
            return True  # Stop retrying
    else:
        node.status = "failed"
        node.error_message = f"LLM API Error: {error}"
        st.error(node.error_message)
        return True  # Stop retrying (non-retryable error)

def generate_tree_graph() -> graphviz.Digraph:
    """Generates a Graphviz graph of the task decomposition tree."""
    dot = graphviz.Digraph(comment='Task Decomposition Tree')
    for node_id, node in st.session_state.node_lookup.items():
        node_label = (f"ID: {node.node_id}\nTask: {node.task_description[:20]}"
                      f"{'...' if len(node.task_description) > 20 else ''}\nStatus: {node.status}")
        if node.status == "failed":
            node_label += f"\nError: {node.error_message[:20]}"
            dot.node(node.node_id, label=node_label, color="red")
        elif node.status == "completed":
            dot.node(node.node_id, label=node_label, color="green")
        elif node.status == "overridden":
            dot.node(node.node_id, label=node_label, color="orange")
        else:
            dot.node(node.node_id, label=node_label)

        if node.parent_id:
            dot.edge(node.parent_id, node.node_id)
    return dot

def display_node_textual(node_id: str, level: int = 0) -> None:
    """Displays the node hierarchy in a textual format."""
    if node_id not in st.session_state.node_lookup:
        return
    node = st.session_state.node_lookup[node_id]
    indent = "    " * level
    st.write(f"{indent}- *Node ID:* {node.node_id} (*Status:* {node.status})")
    st.write(f"{indent}  *Task:* {node.task_description}")
    if node.child_ids:
        for child_id in node.child_ids:
            display_node_textual(child_id, level + 1)


# ui/app.py
import streamlit as st
import google.generativeai as genai  # Import here
from agent.agent import Agent  # Import the Agent class
from agent.utils import generate_tree_graph, display_node_textual
import os
from dotenv import load_dotenv

load_dotenv()


# --- Constants ---
MAX_RETRIES: int = 3
RETRY_DELAY: int = 2
LLM_MODEL: str = "gemini-1.5-pro-002"
LLM_TEMPERATURE: float = 0.5
LLM_MAX_TOKENS: int = 1200
GLOBAL_CONTEXT_SUMMARY_INTERVAL: int = 5
MAX_DEPTH: int = 5


st.set_page_config(layout="wide")
st.title("Hierarchical Task Decomposition Agent (with Human-in-the-Loop)")

# Configure Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# --- Main App Logic ---

if 'agent' not in st.session_state:
    llm = genai.GenerativeModel(model_name=LLM_MODEL)
    llm_config = genai.GenerationConfig(
                        temperature=LLM_TEMPERATURE,
                        max_output_tokens=LLM_MAX_TOKENS
                    )
    st.session_state.llm_config = llm_config #Keep this
    st.session_state.agent = Agent(llm=llm, llm_config=llm_config, global_context="This agent decomposes complex tasks into sub-tasks")


with st.form("task_input_form"):
    task_description = st.text_input("Enter the initial task:",
                                     "MAKE A LUDO GAME THAT I CAN ENJOY WITH MY FRIENDS",
                                     max_chars=250)
    constraints_input = st.text_input("Enter any initial constraints (comma-separated):")
    submitted = st.form_submit_button("Start Agent")

if submitted:
    # st.session_state.agent.reset_agent() # Reset using agent's method
    initial_constraints = [c.strip() for c in constraints_input.split(",") if c.strip()]
    st.session_state.agent.run(task_description, initial_constraints)

# Display in text format (Moved to utils)
st.write("## Node Hierarchy (Textual):")
if st.session_state.get("root_node_id"):
     display_node_textual(st.session_state.root_node_id)
# Display Tree (Moved to utils)
st.write("## Node Hierarchy (Graphical):")
try:
    graph = generate_tree_graph()
    st.graphviz_chart(graph)
except Exception as e:
    st.error(f"Error generating graph: {e}")

# Display Global Context
st.write("## Global Context:")
if st.session_state.agent:
    st.write(st.session_state.agent.global_memory.get_context())

if st.button("Reset Agent"):
    st.session_state.agent.reset_agent() # Reset Agent
    st.rerun()
