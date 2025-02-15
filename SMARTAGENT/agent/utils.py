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
