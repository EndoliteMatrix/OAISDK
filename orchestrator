# orchestrator.py
# Multi-agent framework using LangChain and OpenAI API

import os
import re
import subprocess
from subprocess import PIPE, STDOUT
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
dotenv_path = find_dotenv()
print(f"[DEBUG] Loading environment from: {dotenv_path}")
load_dotenv(dotenv_path, override=True)

# Detect API key from common env var names
detected_var = None
for var in ("OPENAI_API_KEY", "OPEN_AI_API_KEY", "OPEN_API_KEY"):
    key = os.getenv(var)
    if key:
        OPENAI_API_KEY = "rm --placeholder $APISDK"
        detected_var = var
        break
if not detected_var:
    raise ValueError(
        "Missing API key. Ensure .env contains one of: OPENAI_API_KEY, OPEN_AI_API_KEY, OPEN_API_KEY"
    )
print(f"[DEBUG] Detected API key in environment variable: {detected_var}")

# Instantiate underlying LLMs for each agent
clarifier_llm = ChatOpenAI(model_name="o4-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
syntax_llm    = ChatOpenAI(model_name="o4-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
executor_llm  = ChatOpenAI(model_name="o3-pro",   temperature=0.1, openai_api_key=OPENAI_API_KEY)
code_llm      = ChatOpenAI(model_name="o4-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)

# Centralized memory for multi-agent context
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# PromptTemplates for cleaner substitution
clarifier_template = PromptTemplate(
    template=(
        "You are a clarifier agent. If the user's request lacks detail, ask a follow-up question. "
        "Otherwise, respond with 'OK'.\n"
        "Context:\n{history}\n"
        "User: {user_input}"
    ),
    input_variables=["history", "user_input"]
)

syntax_template = PromptTemplate(
    template=(
        "You are a syntax validation agent. Review the following code or command snippet. "
        "If syntax or API usage is outdated, provide corrected version only. "
        "If it's correct, reply 'Syntax OK'.\n\n"
        "Snippet:\n{snippet}"
    ),
    input_variables=["snippet"]
)

code_template = PromptTemplate(
    template=(
        "You are a code generation agent. Write a {language} code snippet to accomplish the following task:\n"
        "{task_description}\nProduce only the code block."
    ),
    input_variables=["language", "task_description"]
)

# Instantiate chains with PromptTemplates and memory
clarifier_chain = LLMChain(llm=clarifier_llm, prompt=clarifier_template, memory=memory)
syntax_chain    = LLMChain(llm=syntax_llm,    prompt=syntax_template,    memory=memory)
code_chain      = LLMChain(llm=code_llm,      prompt=code_template,      memory=memory)

# Executor utilities
DEFAULT_TIMEOUT = 60  # seconds

def dry_run(cmd: str) -> str:
    return f"[DRY RUN] $ {cmd}"

def run_with_confirm(cmd: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    print(f"Agent proposes execution: {cmd}")
    if input("Approve? [y/N]: ").strip().lower() != 'y':
        return "⏸️ Aborted"
    try:
        result = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=STDOUT,
                                 timeout=timeout, text=True, check=True)
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"⚠️ Command timed out after {timeout} seconds."
    except subprocess.CalledProcessError as e:
        return e.stdout or str(e)

# Command detection
COMMAND_PATTERNS = [r"^\s*\$ ", r"^\s*(sudo|ls|cat|grep|awk|curl|python|docker|nmap)"]

def is_command(text: str) -> bool:
    return any(re.search(p, text) for p in COMMAND_PATTERNS)

# Orchestrator function
def orchestrate(user_input: str) -> str:
    clarifier_resp = clarifier_chain.run(history=memory.buffer, user_input=user_input)
    if clarifier_resp.strip().lower() != 'ok':
        return clarifier_resp
    if is_command(user_input):
        cmd = re.sub(r"^\s*\$ ", "", user_input)
        syntax_resp = syntax_chain.run(snippet=cmd)
        if syntax_resp.strip().lower() != 'syntax ok':
            return syntax_resp
        print(dry_run(cmd))
        return run_with_confirm(cmd)
    return code_chain.run(language='bash/python', task_description=user_input)

if __name__ == '__main__':
    print("=== Multi-Agent Orchestrator ===")
    while True:
        ui = input("\n>> ")
        if ui.lower() in ('exit', 'quit'):
            break
        output = orchestrate(ui)
        print(f"\n{output}")
