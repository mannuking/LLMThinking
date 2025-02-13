import streamlit as st
import openai
import uuid
import re
import json
import time
import graphviz
from typing import Callable, Any, Optional

# --- Constants ---

MAX_RETRIES: int = 3
RETRY_DELAY: int = 2  # Seconds
MAX_TASK_LENGTH: int = 250
LLM_MODEL: str = "gpt-3.5-turbo-instruct"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_TOKENS: int = 500
GLOBAL_CONTEXT_SUMMARY_INTERVAL: int = 5  # Summarize global context every 5 node executions


# --- Constraint Checker Type ---
ConstraintChecker = Callable[[str, "Node"], bool]


# --- Node Class Definition ---

class Node:
    def __init__(self, parent_id: Optional[str], task_description: str, llm_instance: openai.OpenAI) -> None:
        self.node_id: str = str(uuid.uuid4())
        self.parent_id: Optional[str] = parent_id
        self.child_ids: list[str] = []
        self.task_description: str = task_description
        self.status: str = "pending"  # pending, running, completed, failed, overridden
        self.output: str = ""
        self.llm_instance: openai.OpenAI = llm_instance
        self.local_memory: dict[str, str] = {}
        self.error_message: str = ""

    def store_in_memory(self, key: str, value: str) -> None:
        """Stores a key-value pair in the node's local memory."""
        self.local_memory[key] = value

    def retrieve_from_memory(self, key: str) -> Optional[str]:
        """Retrieves a value from the node's local memory by key. Returns None if key not found."""
        return self.local_memory.get(key)

    def add_child(self, child_node: "Node") -> None:
        """Adds a child node's ID to the list of child IDs."""
        self.child_ids.append(child_node.node_id)

    def get_parent_node(self) -> Optional["Node"]:
        """Retrieves the parent node object from the global node lookup."""
        return st.session_state.node_lookup.get(self.parent_id)

    def _process_store_command(self, line: str, memory_operations_history: str) -> str:
        """Helper function to process STORE commands."""
        store_match = re.match(r"STORE\s+(.+?)\s+(.+)", line)
        if store_match:
            key, value = store_match.groups()
            self.store_in_memory(key, value)
            memory_operations_history += f"STORE {key} {value}\\n"
        return memory_operations_history

    def _process_retrieve_command(self, line: str, memory_operations_history: str) -> str:
        """Helper function to process RETRIEVE commands."""
        retrieve_match = re.match(r"RETRIEVE\s+(.+)", line)
        if retrieve_match:
            key = retrieve_match.group(1)
            value = self.retrieve_from_memory(key)
            memory_operations_history += f"RETRIEVE {key} -> {value}\\n"
        return memory_operations_history

    def _process_query_parent_command(self, line: str, parent_query_results: str) -> str:
        """Helper function to process QUERY_PARENT commands."""
        query_parent_match = re.match(r"QUERY_PARENT\s+(.+)", line)
        if query_parent_match:
            key = query_parent_match.group(1)
            if self.parent_id is not None:
                parent_node = self.get_parent_node()
                if parent_node:
                    value = parent_node.retrieve_from_memory(key)
                    parent_query_results += f"QUERY_PARENT {key} -> {value}\\n"
                else:
                    parent_query_results += f"QUERY_PARENT {key} -> ERROR: Could not access parent node.\\n"
            else:
                parent_query_results += f"QUERY_PARENT {key} -> ERROR: No parent node.\\n"
        return parent_query_results

    def _process_decompose_command(self, line: str, llm_output:str) -> bool:
        """Helper function to process DECOMPOSE commands.  Returns True if decomposed."""
        decompose_match = re.match(r"DECOMPOSE\s+(.+)", line)
        if decompose_match:
            sub_tasks_str = decompose_match.group(1)
            sub_tasks = [task.strip() for task in sub_tasks_str.split(';')]

            # Get constraints from the LLM's output (if any)
            constraints = self.extract_constraints(llm_output)

            for task_description in sub_tasks:
                child_node = create_child_node(self, task_description, self.llm_instance)
                # Apply dynamically generated constraints
                for constraint in constraints:
                    st.session_state.attention_mechanism.add_constraint(child_node.node_id, constraint)

            self.remove_node()
            return True  # Indicate decomposition occurred
        return False

    def extract_constraints(self, llm_output: str) -> list[str]:
        """
        Extracts constraints from the LLM output, now supporting JSON format.
        """
        constraints = []
        try:
            # Search for constraints provided in JSON format
            match = re.search(r"Constraints:\s*```json\s*([\\s\\S]*?)\s*```", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                constraints_data = json.loads(json_str)
                # Assuming each item in the list is a dictionary with a "constraint" key
                for item in constraints_data:
                    if isinstance(item, dict) and "constraint" in item:
                        constraints.append(item["constraint"])
            else: #If no constraints are provided in JSON format
                for line in llm_output.splitlines():
                    line = line.strip()
                    if line.startswith("CONSTRAINT:"):
                        constraint = line[len("CONSTRAINT:"):].strip()
                        constraints.append(constraint)
        except json.JSONDecodeError:
            st.error("Error decoding JSON in extract_constraints.")
            # Optionally, handle the malformed JSON more gracefully
        return constraints

    def process_llm_output(self, llm_output: str, parent_query_results: str) -> None:
        """Processes the LLM output, handling commands and updating memory."""
        memory_operations_history = self.local_memory.get("memory_operations_history", "")

        lines = llm_output.splitlines()
        for line in lines:
            line = line.strip()
            try:
                memory_operations_history = self._process_store_command(line, memory_operations_history)
                memory_operations_history = self._process_retrieve_command(line, memory_operations_history)
                parent_query_results = self._process_query_parent_command(line, parent_query_results)
                if self._process_decompose_command(line, llm_output):
                    return  # Exit if decomposition occurred
            except Exception as e:
                self.status = "failed"
                self.error_message = f"Error processing LLM output line '{line}': {e}"
                st.error(self.error_message)
                return

        self.store_in_memory("raw_llm_output", llm_output)
        self.store_in_memory("memory_operations_history", memory_operations_history)
        self.store_in_memory("parent_query_results", parent_query_results)

    def build_prompt(self) -> str:
        """Constructs the prompt for the LLM, including task, constraints, context, and memory."""
        constraints: list[str] = st.session_state.attention_mechanism.get_constraints(self.node_id)
        constraints_str: str = "\\n".join([f"- {c}" for c in constraints]) if constraints else "None"
        global_context: str = st.session_state.attention_mechanism.get_global_context()
        memory_operations_history: str = self.local_memory.get("memory_operations_history", "")
        parent_query_results: str = self.local_memory.get("parent_query_results", "")

        return '\\n'.join([
            "You are a helpful assistant tasked with solving the following:",
            "",
            f"Task: {self.task_description}",
            "",
            "Constraints:",
            f"{constraints_str}",
            "",
            "Global Context:",
            f"{global_context}",
            "",
            "You have access to local memory.  Use the following commands to interact with it:",
            "- STORE <key> <value>: Store information in memory.",
            "- RETRIEVE <key>: Retrieve information from memory.",
            "",
            "You can also query your parent node's memory:",
            "- QUERY_PARENT <key>: Retrieve information from the parent's memory.",
            "",
            "If the task is too complex to solve directly, you can decompose it into smaller sub-tasks:",
            "- DECOMPOSE <task_1>; <task_2>; ... ; <task_n>: Break down the task into sub-tasks.",
            "  You may also provide CONSTRAINTS for the child nodes in JSON Format.",
            "  Constraints:",
            "    ```json",
            "    [",
            "        {\"constraint\": \"format: json\"},",
            "        {\"constraint\": \"max_length\": \"200\"}",
            "    ]",
            "    ```",
            "",
            "Previous Memory Operations and Results:",
            f"{memory_operations_history}",
            "",
            "Parent Memory Query Results:",
            f"{parent_query_results}",
            "",
            "Provide your solution. If you can solve the task directly, provide the solution.",
            "If the task is too complex, use the DECOMPOSE command to break it down."
        ])

    def execute(self) -> None:
        """Executes the node's task, interacting with the LLM and handling retries."""

        self.status = "running"
        self.store_in_memory("initial_task", self.task_description)
        prompt = self.build_prompt()

        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm_instance.completions.create(
                    model=LLM_MODEL,
                    prompt=prompt,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE
                )
                self.output = response.choices[0].text
                self.process_llm_output(self.output, self.local_memory.get("parent_query_results", ""))
                break

            except openai.OpenAIError as e:
                if isinstance(e, openai.RateLimitError):
                    if handle_retryable_error(self, attempt, e):
                        return
                elif isinstance(e, openai.Timeout):
                    if handle_retryable_error(self, attempt, e):
                        return
                elif isinstance(e, openai.APIConnectionError):
                    if handle_retryable_error(self, attempt, e):
                        return
                else:
                    self.status = "failed"
                    self.error_message = f"LLM API Error: {e}"
                    return

            except Exception as e:
                self.status = "failed"
                self.error_message = f"Unexpected error during execution: {e}"
                st.error(self.error_message)
                return

        if self.status != "failed":
            self.status = "completed"

    def remove_node(self) -> None:
        """Removes the node from the global node lookup and attention mechanism."""
        st.session_state.node_lookup.pop(self.node_id, None)
        st.session_state.attention_mechanism.remove_node(self.node_id)


    

class AttentionMechanism:
    def __init__(self) -> None:
        self.dependency_graph: dict[str, list[Optional[str]]] = {}
        self.constraints: dict[str, list[str]] = {}
        self.global_context: str = "This agent decomposes complex tasks into smaller sub-tasks."
        self._constraint_checkers: dict[str, ConstraintChecker] = {}
        self.execution_count: int = 0  # To trigger global context summarization

    def add_dependency(self, dependent_node_id: str, dependency_node_id: Optional[str]) -> None:
        """Adds a dependency to the dependency graph."""
        if dependent_node_id not in self.dependency_graph:
            self.dependency_graph[dependent_node_id] = []
        self.dependency_graph[dependent_node_id].append(dependency_node_id)

    def add_constraint(self, node_id: str, constraint: str) -> None:
        """Adds a constraint for a given node."""
        if node_id not in self.constraints:
            self.constraints[node_id] = []
        self.constraints[node_id].append(constraint)

    def get_constraints(self, node_id: str) -> list[str]:
        """Retrieves the constraints for a given node."""
        return self.constraints.get(node_id, [])

    def update_constraint(self, node_id: str, constraint_index: int, new_constraint: str) -> None:
        """Updates a specific constraint for a node."""
        if node_id in self.constraints and 0 <= constraint_index < len(self.constraints[node_id]):
            self.constraints[node_id][constraint_index] = new_constraint

    def remove_constraint(self, node_id: str, constraint_index: int) -> None:
        """Removes a specific constraint for a node."""
        if node_id in self.constraints and 0 <= constraint_index < len(self.constraints[node_id]):
            del self.constraints[node_id][constraint_index]

    def propagate_constraints(self) -> None:
        """Propagates constraints from parent nodes to child nodes (depth-first)."""
        def dfs(node_id: str) -> None:
            if node_id in st.session_state.node_lookup:
                node = st.session_state.node_lookup[node_id]
                parent_constraints = self.get_constraints(node.parent_id) if node.parent_id else []
                for constraint in parent_constraints:
                    self.add_constraint(node_id, constraint)  # Add to *current* node
                for child_id in node.child_ids:
                    dfs(child_id)

        if st.session_state.root_node_id:
            dfs(st.session_state.root_node_id)

    def _summarize_global_context(self) -> None:
        """Summarizes the global context itself using the LLM."""
        prompt = f"""Summarize the following global context:

{self.global_context}

Summary:
"""
        response = openai.completions.create( #Using openai directly
            model=LLM_MODEL,
            prompt=prompt,
            max_tokens=150,  # Adjust as needed
            temperature=0.5
        )
        self.global_context = response.choices[0].text.strip()

    def summarize_node(self, node: Node) -> None:
        """Summarizes a completed node and appends it to the global context."""
        prompt = f"""Summarize the following task and its result concisely:

Task: {node.task_description}

Result: {node.output}

Summary:
"""
        response = node.llm_instance.completions.create(
            model=LLM_MODEL,
            prompt=prompt,
            max_tokens=100,
            temperature=0.5
        )
        summary = response.choices[0].text.strip()
        self.global_context += f"\\n- Node {node.node_id} ({node.status}): {summary}"

        self.execution_count += 1
        if self.execution_count % GLOBAL_CONTEXT_SUMMARY_INTERVAL == 0:
            self._summarize_global_context()

    def get_global_context(self) -> str:
        """Returns the current global context."""
        return self.global_context

    def add_constraint_checker(self, constraint_type: str, checker: ConstraintChecker) -> None:
        """Registers a constraint checker function."""
        self._constraint_checkers[constraint_type] = checker

    def _check_json_format(self, constraint_value: str, node: Node) -> bool:
        """Helper function: Checks if the node output is valid JSON."""
        try:
            json.loads(node.output)
            return True
        except json.JSONDecodeError:
            node.status = "failed"
            node.error_message = f"Constraint violated: Output must be in JSON format. Output: {node.output}"
            return False

    def _check_contains_word(self, constraint_value: str, node: Node) -> bool:
        """Helper function: Checks if the output contains a specific word."""
        if constraint_value in node.output:
            return True
        else:
            node.status = "failed"
            node.error_message = f"Constraint violated: Output must contain '{constraint_value}'. Output: {node.output}"
            return False

    def _check_max_length(self, constraint_value: str, node: Node) -> bool:
        """Helper function: Checks if the output length is within the limit."""
        try:
            max_length = int(constraint_value)
            if len(node.output) <= max_length:
                return True
            else:
                node.status = "failed"
                node.error_message = f"Constraint violated: Output must be no more than {max_length} characters. Output: {node.output}"
                return False
        except ValueError:
            node.status = "failed"
            node.error_message = f"Constraint violated: Invalid max length value '{constraint_value}'"
            return False

    def check_constraints(self, node: Node) -> bool:
        """Checks if a node's output violates any constraints using registered checkers."""
        constraints = self.get_constraints(node.node_id)
        if not constraints:
            return True

        for constraint in constraints:
            constraint_type, constraint_value = parse_constraint(constraint)
            checker = self._constraint_checkers.get(constraint_type)
            if checker:
                if not checker(constraint_value, node):
                    return False
            else:
                st.warning(f"No checker found for constraint type: {constraint_type}")  # Should not happen
        return True

    def track_dependencies(self, parent_node: Optional[Node], child_node: Node) -> None:
        """Tracks the dependencies between parent and child nodes."""
        if parent_node is None:
            self.add_dependency(child_node.node_id, None)
        else:
            self.add_dependency(child_node.node_id, parent_node.node_id)

    def remove_node(self, node_id: str) -> None:
        """Removes the node from the global node lookup and attention mechanism."""
        self.dependency_graph.pop(node_id, None)
        self.constraints.pop(node_id, None)


# --- Helper Functions ---

def parse_constraint(constraint_string: str) -> tuple[str, str]:
    """Parses a constraint string into its type and value."""
    try:
        constraint_type, constraint_value = constraint_string.split(":", 1)
        return constraint_type.strip(), constraint_value.strip()
    except ValueError:
        return "unknown", constraint_string  # Fallback for malformed

def handle_retryable_error(node: Node, attempt: int, error: Exception) -> bool:
    """Handles retryable errors during LLM interaction. Returns True if should exit."""
    if attempt == MAX_RETRIES - 1:
        node.status = "failed"
        node.error_message = f"LLM API Error: Max retries exceeded: {error}"
        return True
    else:
        st.toast(f"{type(error).__name__}. Retrying in {RETRY_DELAY} seconds (attempt {attempt + 2}/{MAX_RETRIES})...")
        time.sleep(RETRY_DELAY)
        return False

def setup_agent() -> None:
    """Initializes the AttentionMechanism and registers constraint checkers."""
    attention_mechanism = AttentionMechanism()
    # Register constraint checkers
    
    attention_mechanism.add_constraint_checker("contains", attention_mechanism._check_contains_word)
    attention_mechanism.add_constraint_checker("max_length", attention_mechanism._check_max_length)
    st.session_state.attention_mechanism = attention_mechanism
    st.session_state.node_lookup = {}
    st.session_state.root_node_id = None

def reset_agent() -> None:
    """Resets the agent to its initial state."""
    st.session_state.clear()
    setup_agent()

def generate_tree_graph() -> graphviz.Digraph:
    """Generates a Graphviz graph of the node tree."""
    dot = graphviz.Digraph(comment='Task Decomposition Tree')
    for node_id, node in st.session_state.node_lookup.items():
        node_label = f"ID: {node.node_id}\\nTask: {node.task_description[:20]}{'...'}...\\nStatus: {node.status}"
        if node.status == "failed":
            node_label += f"\\nError: {node.error_message[:20]}"  # Show error on graph
            dot.node(node.node_id, label=node_label, color="red")
        elif node.status == "completed":
            dot.node(node.node_id, label=node_label, color="green")
        elif node.status == "overridden":
            dot.node(node.node_id, label=node_label, color="blue")
        else:
            dot.node(node.node_id, label=node_label)

        if node.parent_id:
            dot.edge(node.parent_id, node.node_id)
    return dot

# --- Node Management Functions ---

def create_root_node(task_description: str, llm_instance: openai.OpenAI, initial_constraints: Optional[list[str]] = None) -> Node:
    """Creates the root node of the task tree."""
    new_node = Node(parent_id=None, task_description=task_description, llm_instance=llm_instance)
    st.session_state.node_lookup[new_node.node_id] = new_node
    if initial_constraints:
        for constraint in initial_constraints:
            st.session_state.attention_mechanism.add_constraint(new_node.node_id, constraint)
    st.session_state.attention_mechanism.track_dependencies(None, new_node)
    return new_node


def create_child_node(parent_node: Node, task_description: str, llm_instance: openai.OpenAI) -> Node:
    """Creates a child node and links it to its parent."""
    new_node = Node(parent_id=parent_node.node_id, task_description=task_description, llm_instance=llm_instance)
    st.session_state.node_lookup[new_node.node_id] = new_node
    parent_node.add_child(new_node)
    st.session_state.attention_mechanism.track_dependencies(parent_node, new_node)
    return new_node


def _execute_tree_recursive(node: Node) -> None:
    """Recursive helper function for executing the task tree (without UI interaction)."""
    node.execute()
    if node.status == "failed":
        st.error(f"ERROR: Node {node.node_id} failed: {node.error_message} (Parent: {node.parent_id})")
        return

    if not st.session_state.attention_mechanism.check_constraints(node):
        return

    if node.child_ids:
        st.session_state.attention_mechanism.propagate_constraints()
        for child_id in node.child_ids:
            child_node = st.session_state.node_lookup[child_id]
            _execute_tree_recursive(child_node)

    if node.status in ("completed", "overridden"):
        st.session_state.attention_mechanism.summarize_node(node)


def execute_tree(node: Node) -> None:
    """Executes the task tree with human-in-the-loop interaction, leveraging _execute_tree_recursive."""

    st.session_state.current_node_id = node.node_id
    st.session_state.override_output = ""
    st.session_state.review_submitted = False

    while True:
        if node.status not in ("completed", "failed", "overridden"):
            _execute_tree_recursive(node)  # Execute the node (and its children, recursively)

        # If execution stopped due to failure, exit the loop
        if node.status == "failed":
            return

        st.write(f"## Review Node: {node.node_id}")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Task:** {node.task_description}")
            st.write(f"**Status:** {node.status}")
            st.write("**Constraints:**")
            for i, constraint in enumerate(st.session_state.attention_mechanism.get_constraints(node.node_id)):
                st.write(f"- {constraint}")

        with col2:
            st.write("**Output:**")
            st.write(node.output)
            if node.error_message:
                st.write(f"**Error:** {node.error_message}")
            with st.expander("Local Memory"):
                if node.local_memory:
                    for key, value in node.local_memory.items():
                        st.write(f"*   **{key}:** {value}")

        with st.form(key=f'review_form_{node.node_id}'):
            st.write("### Override Output (Optional)")
            override_output = st.text_area("Enter overridden output:", value=st.session_state.override_output if st.session_state.override_output else node.output, height=150)
            if override_output != node.output:
                st.session_state.override_output = override_output

            st.write("### Modify Constraints (Optional)")
            current_constraints = st.session_state.attention_mechanism.get_constraints(node.node_id)
            new_constraints = []

            for i, constraint in enumerate(current_constraints):
                cols = st.columns([8, 2])
                with cols[0]:
                    updated_constraint = st.text_input(f"Constraint {i + 1}:", value=constraint,
                                                        key=f"constraint_{node.node_id}_{i}")
                    new_constraints.append(updated_constraint)
                with cols[1]:
                    if st.form_submit_button(label=f"Delete"):
                        st.session_state.attention_mechanism.remove_constraint(node.node_id, i)
                        st.experimental_rerun() # Force re-render

            cols = st.columns([8, 2])
            with cols[0]:
                new_constraint = st.text_input("Add New Constraint:", key=f"new_constraint_{node.node_id}")
            with cols[1]:
                if st.form_submit_button(label="Add"):
                    if new_constraint.strip():
                        st.session_state.attention_mechanism.add_constraint(node.node_id, new_constraint.strip())
                        st.experimental_rerun()

            st.write("### Manual Decomposition (Optional)")
            manual_subtasks = st.text_input("Enter sub-tasks (semicolon-separated):",
                                                key=f"manual_decompose_{node.node_id}")

            col1, col2, col3 = st.columns(3)
            with col1:
                review_submitted = st.form_submit_button("Approve & Continue")
            with col2:
                stop_execution = st.form_submit_button("Stop Execution")
            with col3:
                force_re_execute = st.form_submit_button("Re-execute Node")



        if review_submitted:
            if st.session_state.override_output != "":
                node.output = st.session_state.override_output
                node.status = "overridden"

            for i, updated_constraint in enumerate(new_constraints):
                st.session_state.attention_mechanism.update_constraint(node.node_id, i, updated_constraint)


            if manual_subtasks.strip():
                sub_tasks = [task.strip() for task in manual_subtasks.split(';') if task.strip()]
                for task_description in sub_tasks:
                    create_child_node(node, task_description, node.llm_instance)
                node.remove_node()  # Remove after manual decomposition
                if node.status != "failed":
                    node.status = "completed"
            if not st.session_state.attention_mechanism.check_constraints(node): #Re-check constraints
                return

            st.session_state.review_submitted = True #To stop the while loop
            break

        if stop_execution:
            st.warning("Execution stopped by user.")
            return

        if force_re_execute:
            node.status = "pending" # Reset status for re-execution
            node.output = "" #Clear output
            node.error_message = "" #Clear error
            st.experimental_rerun()

        st.stop() # Stop execution until user interacts



# --- Streamlit UI Setup ---

st.set_page_config(layout="wide")  # Use wider layout
st.title("Hierarchical Task Decomposition Agent (with Human-in-the-Loop)")

# Initialize LLM API
openai.api_key = st.secrets["OPENAI_API_KEY"]
llm = openai

# Setup agent on first run
if 'attention_mechanism' not in st.session_state:
    setup_agent()

# Use st.form for input grouping
with st.form("task_input_form"):
    task_description = st.text_input("Enter the initial task:",
                                     "Write a short story about a cat that goes on an adventure in Paris.",
                                     max_chars=MAX_TASK_LENGTH)
    constraints_input = st.text_input("Enter any initial constraints (comma-separated):",
                                     "format: json")  # Use constraint format
    submitted = st.form_submit_button("Start Agent")

if submitted:
    reset_agent()  # Reset on each new task
    initial_constraints = [c.strip() for c in constraints_input.split(",") if c.strip()]
    root_node = create_root_node(task_description, llm, initial_constraints)
    st.session_state.root_node_id = root_node.node_id
    execute_tree(root_node)
    st.success("Agent execution complete (or stopped by user)!")

# Display node information (textual representation)
st.write("## Node Hierarchy (Textual):")

def display_node_textual(node_id: str, level: int = 0) -> None:
    """Displays node information recursively (textual version)."""
    if node_id not in st.session_state.node_lookup:
        return
    node = st.session_state.node_lookup[node_id]
    indent = "    " * level
    st.write(f"{indent}- **Node ID:** {node.node_id}  (**Status:** {node.status})")
    st.write(f"{indent}  **Task:** {node.task_description}")
    if node.child_ids:
        for child_id in node.child_ids:
            display_node_textual(child_id, level + 1)


if st.session_state.root_node_id:
    display_node_textual(st.session_state.root_node_id)


# Display node tree (graphical representation)
st.write("## Node Hierarchy (Graphical):")
try:
    graph = generate_tree_graph()
    st.graphviz_chart(graph)
except Exception as e:
    st.error(f"Error generating graph: {e}")

st.write("## Global Context:")
if st.session_state.attention_mechanism:
    st.write(st.session_state.attention_mechanism.get_global_context())

if st.button("Reset Agent"):
    reset_agent()
    st.rerun()
