import os
import time
import json
import hashlib
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class AgentRole(Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    ANALYST = "analyst"
    CODER = "coder"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    CUSTOM = "custom"


@dataclass
class AgentCapability:
    name: str
    description: str
    tools_required: List[str] = field(default_factory=list)
    max_concurrent: int = 1
    estimated_time_seconds: float = 30.0
    priority: int = 5


@dataclass
class TaskResult:
    task_id: str
    agent_name: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None


@dataclass
class AgentPerformance:
    agent_name: str
    role: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    success_rate: float = 0.0
    tokens_used: int = 0
    capabilities_used: List[str] = field(default_factory=list)

    def update(self, result: TaskResult):
        if result.status == TaskStatus.COMPLETED:
            self.tasks_completed += 1
            if result.duration:
                self.total_time += result.duration
                self.average_time = self.total_time / self.tasks_completed
        elif result.status == TaskStatus.FAILED:
            self.tasks_failed += 1
        total = self.tasks_completed + self.tasks_failed
        self.success_rate = self.tasks_completed / total if total > 0 else 0.0


@dataclass
class Memory:
    facts: List[Dict[str, Any]] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    max_facts: int = 1000
    max_history: int = 200

    def add_fact(self, fact: str, source: str, confidence: float = 0.8):
        fact_hash = hashlib.md5(fact.encode()).hexdigest()[:8]
        if not any(f["hash"] == fact_hash for f in self.facts):
            self.facts.append({
                "hash": fact_hash,
                "fact": fact,
                "source": source,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "access_count": 0,
            })
            if len(self.facts) > self.max_facts:
                self.facts.sort(key=lambda x: x["access_count"])
                self.facts.pop(0)

    def query_facts(self, keyword: str) -> List[Dict[str, Any]]:
        results = []
        for fact in self.facts:
            if keyword.lower() in fact["fact"].lower():
                fact["access_count"] += 1
                results.append(fact)
        return results

    def add_conversation(self, role: str, content: str, agent_name: str = ""):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "agent": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_context_window(self, last_n: int = 10) -> List[Dict[str, str]]:
        return self.conversation_history[-last_n:]

    def learn_pattern(self, pattern_name: str, pattern_data: Any):
        self.learned_patterns[pattern_name] = {
            "data": pattern_data,
            "learned_at": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
        }

    def apply_pattern(self, pattern_name: str) -> Optional[Any]:
        if pattern_name in self.learned_patterns:
            self.learned_patterns[pattern_name]["usage_count"] += 1
            return self.learned_patterns[pattern_name]["data"]
        return None


@dataclass
class ProgressTracker:
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    started_at: Optional[float] = None
    task_timings: Dict[str, float] = field(default_factory=dict)
    milestones: List[Dict[str, Any]] = field(default_factory=list)

    def start(self):
        self.started_at = time.time()

    def record_task(self, task_id: str, duration: float, success: bool):
        self.task_timings[task_id] = duration
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        progress = (self.completed_tasks + self.failed_tasks) / max(self.total_tasks, 1)
        self.milestones.append({
            "task_id": task_id,
            "progress": round(progress * 100, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def elapsed_time(self) -> float:
        if self.started_at:
            return round(time.time() - self.started_at, 2)
        return 0.0

    @property
    def progress_percentage(self) -> float:
        return round((self.completed_tasks + self.failed_tasks) / max(self.total_tasks, 1) * 100, 1)

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.completed_tasks > 0 and self.started_at:
            avg_time = self.elapsed_time / self.completed_tasks
            remaining = self.total_tasks - self.completed_tasks - self.failed_tasks
            return round(avg_time * remaining, 2)
        return None


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for the given query and return a short summary."""
    return f"Search results for: {query}"


@tool("file_reader")
def file_reader(file_path: str) -> str:
    """Read contents of a file at the given path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:5000] if len(content) > 5000 else content
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool("file_writer")
def file_writer(file_path: str, content: str) -> str:
    """Write content to a file at the given path."""
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    import ast
    import operator
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def safe_eval(node):
        if isinstance(node, ast.Expression):
            return safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](safe_eval(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool("data_analyzer")
def data_analyzer(data: str) -> str:
    """Analyze structured data (JSON or CSV) and return summary statistics."""
    try:
        if data.strip().startswith(("{", "[")):
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return json.dumps({
                    "type": "array",
                    "length": len(parsed),
                    "sample": parsed[:3] if len(parsed) > 3 else parsed,
                }, indent=2)
            elif isinstance(parsed, dict):
                return json.dumps({
                    "type": "object",
                    "keys": list(parsed.keys()),
                    "key_count": len(parsed.keys()),
                }, indent=2)
        else:
            lines = [l.strip() for l in data.strip().split("\n") if l.strip()]
            return json.dumps({
                "type": "text/csv-like",
                "row_count": len(lines),
                "sample": lines[:5],
            }, indent=2)
    except json.JSONDecodeError:
        pass
    return json.dumps({"type": "text", "length": len(data), "word_count": len(data.split())})


@tool("text_transformer")
def text_transformer(text: str, operation: str = "summary") -> str:
    """Transform text with various operations: summary, word_count, char_count, uppercase, lowercase, reverse."""
    operations = {
        "summary": lambda t: t[:200] + "..." if len(t) > 200 else t,
        "word_count": lambda t: f"Word count: {len(t.split())}",
        "char_count": lambda t: f"Character count: {len(t)}",
        "uppercase": lambda t: t.upper(),
        "lowercase": lambda t: t.lower(),
        "reverse": lambda t: t[::-1],
        "title_case": lambda t: t.title(),
    }
    op_func = operations.get(operation, operations["summary"])
    return op_func(text)


TOOLS_MAP = {
    "web_search": web_search,
    "file_reader": file_reader,
    "file_writer": file_writer,
    "calculator": calculator,
    "data_analyzer": data_analyzer,
    "text_transformer": text_transformer,
}


class AgentCrewManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.capabilities: Dict[str, List[AgentCapability]] = {}
        self.memory = Memory()
        self.progress = ProgressTracker()
        self.performance: Dict[str, AgentPerformance] = {}
        self.task_results: List[TaskResult] = []
        self.max_retries = 3
        self.retry_delay = 2.0
        self.parallel_limit = 4

    def register_agent(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[str]] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        allow_delegation: bool = False,
        verbose: bool = True,
    ) -> Agent:
        tool_objects = [TOOLS_MAP[t] for t in (tools or []) if t in TOOLS_MAP]

        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tool_objects,
            verbose=verbose,
            allow_delegation=allow_delegation,
            model=MODEL,
        )

        self.agents[name] = agent
        self.performance[name] = AgentPerformance(agent_name=name, role=role)

        if capabilities:
            self.capabilities[name] = capabilities

        logger.info(f"Registered agent '{name}' with role '{role}'")
        return agent

    def find_agent_for_task(self, task_description: str) -> Optional[str]:
        best_agent = None
        best_score = -1

        task_lower = task_description.lower()

        for name, caps in self.capabilities.items():
            score = 0
            for cap in caps:
                keywords = cap.name.lower().split() + cap.description.lower().split()
                matches = sum(1 for kw in keywords if kw in task_lower)
                score += matches * (6 - cap.priority)
            if score > best_score:
                best_score = score
                best_agent = name

        if best_agent is None and self.agents:
            best_agent = next(iter(self.agents))

        return best_agent

    def create_task(
        self,
        description: str,
        expected_output: str,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        context_tasks: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
    ) -> Task:
        if agent_name is None:
            agent_name = self.find_agent_for_task(description)

        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}")

        agent = self.agents[agent_name]

        if tools:
            tool_objects = [TOOLS_MAP[t] for t in tools if t in TOOLS_MAP]
            agent.tools.extend(tool_objects)

        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )

        return task

    def build_sequential_crew(self, tasks: List[Task]) -> Crew:
        return Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

    def build_hierarchical_crew(self, tasks: List[Task], manager_agent: Optional[str] = None) -> Crew:
        manager = self.agents.get(manager_agent) if manager_agent else None
        return Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.hierarchical,
            manager_agent=manager,
            verbose=True,
        )

    def execute_with_retry(self, crew_func: Callable, task_id: str) -> TaskResult:
        result = TaskResult(
            task_id=task_id,
            agent_name="unknown",
            status=TaskStatus.RUNNING,
            start_time=time.time(),
        )

        for attempt in range(self.max_retries + 1):
            try:
                result.retry_count = attempt
                output = crew_func()
                result.status = TaskStatus.COMPLETED
                result.output = output
                result.end_time = time.time()
                return result
            except Exception as e:
                logger.warning(f"Task '{task_id}' attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries:
                    result.status = TaskStatus.RETRYING
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    result.status = TaskStatus.FAILED
                    result.error = str(e)
                    result.end_time = time.time()

        return result

    def run_parallel_tasks(self, task_groups: List[List[Task]]) -> List[TaskResult]:
        all_results = []

        with ThreadPoolExecutor(max_workers=self.parallel_limit) as executor:
            future_to_group = {}
            for i, group in enumerate(task_groups):
                crew = Crew(
                    agents=list(set(t.agent for t in group)),
                    tasks=group,
                    process=Process.sequential,
                    verbose=False,
                )
                future = executor.submit(crew.kickoff)
                future_to_group[future] = i

            for future in as_completed(future_to_group):
                group_idx = future_to_group[future]
                try:
                    output = future.result()
                    result = TaskResult(
                        task_id=f"parallel_group_{group_idx}",
                        agent_name="parallel",
                        status=TaskStatus.COMPLETED,
                        output=output,
                        start_time=time.time(),
                        end_time=time.time(),
                    )
                except Exception as e:
                    result = TaskResult(
                        task_id=f"parallel_group_{group_idx}",
                        agent_name="parallel",
                        status=TaskStatus.FAILED,
                        error=str(e),
                        start_time=time.time(),
                        end_time=time.time(),
                    )
                all_results.append(result)

        return all_results

    def execute_topic(self, topic: str, mode: str = "sequential") -> Dict[str, Any]:
        self.progress.start()
        start_time = time.time()

        researcher = self.register_agent(
            name="researcher",
            role="Senior Research Analyst",
            goal=f"Research the latest, credible information about {topic}",
            backstory=(
                "You are a meticulous analyst who verifies facts and cites sources. "
                "You synthesize complex topics into clear briefing notes."
            ),
            tools=["web_search"],
            capabilities=[
                AgentCapability(name="research", description="web research and fact gathering", tools_required=["web_search"]),
                AgentCapability(name="analysis", description="data analysis and synthesis"),
            ],
        )

        writer = self.register_agent(
            name="writer",
            role="Technical Writer",
            goal=f"Turn the research on {topic} into a concise, actionable report",
            backstory=(
                "You write for busy executives. You lead with the conclusion, "
                "use bullets, and avoid jargon."
            ),
            tools=["text_transformer"],
            capabilities=[
                AgentCapability(name="writing", description="document writing and formatting"),
                AgentCapability(name="editing", description="text editing and proofreading"),
            ],
        )

        analyst = self.register_agent(
            name="analyst",
            role="Data Analyst",
            goal=f"Analyze data and trends related to {topic}",
            backstory=(
                "You are a data-driven thinker who finds patterns in information "
                "and presents insights backed by numbers."
            ),
            tools=["data_analyzer", "calculator"],
            capabilities=[
                AgentCapability(name="data_analysis", description="quantitative analysis", tools_required=["data_analyzer"]),
                AgentCapability(name="statistics", description="statistical calculations", tools_required=["calculator"]),
            ],
        )

        research_task = self.create_task(
            description=f"Research '{topic}'. Identify 3-5 key findings with brief evidence for each.",
            expected_output="A bulleted list of findings with one-line justification each.",
            agent_name="researcher",
            task_id="research",
        )

        analysis_task = self.create_task(
            description=f"Analyze the research findings on '{topic}'. Identify trends, quantify impacts where possible.",
            expected_output="A structured analysis with metrics and trend identification.",
            agent_name="analyst",
            task_id="analysis",
            context_tasks=["research"],
        )

        write_task = self.create_task(
            description="Write a 250-word executive report combining research and analysis.",
            expected_output="A short markdown report with a heading, summary, and bullet points.",
            agent_name="writer",
            task_id="writing",
            context_tasks=["research", "analysis"],
        )

        self.progress.total_tasks = 3
        self.memory.add_fact(f"Topic being researched: {topic}", source="user_input")

        if mode == "parallel":
            results = self.run_parallel_tasks([[research_task], [analysis_task], [write_task]])
        else:
            crew = self.build_sequential_crew([research_task, analysis_task, write_task])
            try:
                kickoff_result = crew.kickoff()
                results = [TaskResult(
                    task_id="sequential_crew",
                    agent_name="crew",
                    status=TaskStatus.COMPLETED,
                    output=kickoff_result,
                    start_time=start_time,
                    end_time=time.time(),
                )]
            except Exception as e:
                results = [TaskResult(
                    task_id="sequential_crew",
                    agent_name="crew",
                    status=TaskStatus.FAILED,
                    error=str(e),
                    start_time=start_time,
                    end_time=time.time(),
                )]

        self.task_results.extend(results)
        for result in results:
            if result.agent_name in self.performance:
                self.performance[result.agent_name].update(result)
            self.progress.record_task(result.task_id, result.duration or 0, result.status == TaskStatus.COMPLETED)

        final_report = self._generate_report(topic, results)

        return {
            "topic": topic,
            "mode": mode,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "output": str(r.output) if r.output else None,
                    "error": r.error,
                    "duration": r.duration,
                    "retry_count": r.retry_count,
                }
                for r in results
            ],
            "agent_performance": {
                name: {
                    "role": perf.role,
                    "tasks_completed": perf.tasks_completed,
                    "tasks_failed": perf.tasks_failed,
                    "success_rate": round(perf.success_rate * 100, 1),
                    "average_time": round(perf.average_time, 2),
                }
                for name, perf in self.performance.items()
            },
            "final_report": final_report,
            "metrics": {
                "total_time": round(time.time() - start_time, 2),
                "progress": self.progress.progress_percentage,
                "completed": self.progress.completed_tasks,
                "failed": self.progress.failed_tasks,
                "memory_facts": len(self.memory.facts),
                "memory_patterns": len(self.memory.learned_patterns),
            },
        }

    def _generate_report(self, topic: str, results: List[TaskResult]) -> str:
        completed = [r for r in results if r.status == TaskStatus.COMPLETED]
        failed = [r for r in results if r.status == TaskStatus.FAILED]

        report_lines = [
            f"# Research Report: {topic}",
            f"Generated at: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Summary",
            f"- Tasks completed: {len(completed)}/{len(results)}",
            f"- Total time: {self.progress.elapsed_time}s",
            "",
        ]

        if completed:
            report_lines.append("## Results")
            for result in completed:
                report_lines.append(f"### {result.task_id}")
                report_lines.append(str(result.output)[:1000] if result.output else "No output")
                report_lines.append("")

        if failed:
            report_lines.append("## Failed Tasks")
            for result in failed:
                report_lines.append(f"- {result.task_id}: {result.error}")

        report_lines.append("")
        report_lines.append("## Agent Performance")
        for name, perf in self.performance.items():
            report_lines.append(
                f"- {name} ({perf.role}): {perf.tasks_completed} completed, "
                f"{perf.success_rate*100:.0f}% success rate"
            )

        return "\n".join(report_lines)

    def get_memory_context(self) -> str:
        context_parts = []
        facts = self.memory.query_facts("")
        if facts:
            context_parts.append("Known facts:")
            for f in facts[-10:]:
                context_parts.append(f"  - {f['fact']}")

        recent = self.memory.get_context_window(5)
        if recent:
            context_parts.append("\nRecent context:")
            for msg in recent:
                context_parts.append(f"  [{msg['role']}] {msg['content'][:100]}")

        return "\n".join(context_parts)


def run(topic: str = "autonomous AI agents", mode: str = "sequential") -> Dict[str, Any]:
    manager = AgentCrewManager()
    result = manager.execute_topic(topic, mode=mode)
    return result


def run_simple(topic: str = "autonomous AI agents") -> str:
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research the latest, credible information about {topic}",
        backstory=(
            "You are a meticulous analyst who verifies facts and cites sources. "
            "You synthesize complex topics into clear briefing notes."
        ),
        tools=[web_search],
        verbose=True,
        allow_delegation=False,
        model=MODEL,
    )

    writer = Agent(
        role="Technical Writer",
        goal=f"Turn the research on {topic} into a concise, actionable report",
        backstory=(
            "You write for busy executives. You lead with the conclusion, "
            "use bullets, and avoid jargon."
        ),
        verbose=True,
        allow_delegation=False,
        model=MODEL,
    )

    research_task = Task(
        description=f"Research '{topic}'. Identify 3-5 key findings with brief evidence for each.",
        expected_output="A bulleted list of findings with one-line justification each.",
        agent=researcher,
    )

    write_task = Task(
        description="Write a 250-word executive report from the research findings.",
        expected_output="A short markdown report with a heading, summary, and bullet points.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    import sys

    topic = sys.argv[1] if len(sys.argv) > 1 else "autonomous AI agents"
    mode = sys.argv[2] if len(sys.argv) > 2 else "simple"

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run the crew against a real LLM.")
        sys.exit(1)

    if mode == "full":
        result = run(topic, mode="sequential")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(run_simple(topic))
