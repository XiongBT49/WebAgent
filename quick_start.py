"""
Quick Start Script for VLLM Web Agent
Simple example demonstrating the agent's capabilities
"""

from main import VLLMWebAgent
import textwrap

# The user's goal for the agent, taken from the project requirements.
user_goal = """
Find the most recent technical report (PDF) about Qwen,
then interpret Figure 1 by describing its purpose and key findings.
"""


def run_quick_start():
    """Runs the quick start example."""
    print("\n" + "╔" + "═" * 80 + "╗")
    print("║" + " " * 21 + "VLLM Web Agent - Project Task" + " " * 22 + "║")
    print("╚" + "═" * 80 + "╝" + "\n")

    print("This example will run the primary task from your project requirements:")
    # Use textwrap to format the goal nicely
    wrapped_goal = textwrap.fill(textwrap.dedent(user_goal).strip(), width=78)
    print("\n" + "-" * 80)
    print("Goal:")
    for line in wrapped_goal.split("\n"):
        print(f"  {line}")
    print("-" * 80 + "\n")

    print("Starting task...\n")

    agent = VLLMWebAgent()
    result = agent.run(user_goal)

    print("\n" + "╔" + "═" * 80 + "╗")
    print("║" + " " * 32 + "Task Complete" + " " * 33 + "║")
    print("╚" + "═" * 80 + "╝" + "\n")

    print(f"Result: {result}")
    print("\nCheck the 'output' directory for:")
    print("- Downloaded PDFs in output/pdfs/")
    print("- Extracted images in output/images/")
    print("- Saved files in output/")
    print("- Execution logs in output/logs/")


if __name__ == "__main__":
    run_quick_start()
