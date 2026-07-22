"""
fluxrouter.cli — Professional Click & Rich CLI for FluxRouter.

Commands:
  fluxrouter              Print help and exit 0
  fluxrouter help         Detailed interactive help & documentation
  fluxrouter init         Interactive setup → ~/.fluxrouter/config.yaml
  fluxrouter route QUERY  Classify and route a query with markdown formatting
  fluxrouter agent TASK   Decompose multi-part task & route each subtask
  fluxrouter serve        Launch FastAPI web dashboard with auto-port discovery
  fluxrouter config       Inspect or edit configuration (--edit)
"""

from __future__ import annotations

import io
import json
import logging
import os
import socket
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event, Thread
from typing import Any

import click
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# Suppress verbose warnings from third-party libraries during CLI execution
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()
err = Console(stderr=True)

VERSION = "0.4.0"

# ── Pricing constants (Groq on-demand, verified 2026-07) ─────────────────────
_PRICE = {
    "fast":   {"input": 0.05,  "output": 0.08},   # llama-3.1-8b   $/1M tok
    "strong": {"input": 0.39,  "output": 0.49},   # qwen/qwen3.6-27b $/1M tok
}

_FAST_DEFAULT   = "llama-3.1-8b-instant"
_STRONG_DEFAULT = "qwen/qwen3.6-27b"



# ── Quiet loggers helper ──────────────────────────────────────────────────────

def _silence_loggers() -> None:
    for lg_name in ("fluxrouter", "sentence_transformers", "transformers", "httpx", "groq", "urllib3"):
        logging.getLogger(lg_name).setLevel(logging.WARNING)


# ── Cost calculation helpers ──────────────────────────────────────────────────

def _cost(model: str, in_tok: int, out_tok: int) -> float:
    p = _PRICE["strong"] if ("70b" in model or "versatile" in model) else _PRICE["fast"]
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


def _strong_cost(in_tok: int, out_tok: int) -> float:
    p = _PRICE["strong"]
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


# ── Config loading helper ─────────────────────────────────────────────────────

def _load_cfg_or_exit():
    """Load config; exit with a helpful message if missing."""
    from fluxrouter.config import load, ConfigNotFoundError
    try:
        return load()
    except ConfigNotFoundError as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        console.print("\nRun [bold cyan]fluxrouter init[/bold cyan] to set up FluxRouter.")
        sys.exit(1)


def _build_components(cfg):
    """Instantiate GroqAdapter + CascadeController + ComplexityClassifier cleanly."""
    _silence_loggers()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from router.classifier.complexity import ComplexityClassifier
        from router.adapters.groq_adapter import GroqAdapter
        from router.cascade.controller import CascadeController
        from router.orchestrator.agent import AgentOrchestrator

        clf = ComplexityClassifier()
        adapter = GroqAdapter(api_key=cfg.api_key)
        ctrl = CascadeController(
            classifier=clf,
            adapter=adapter,
            fast_model=cfg.fast_model,
            strong_model=cfg.strong_model,
            margin_threshold=cfg.margin_threshold,
            judge_escalation_threshold=cfg.escalation_judge_threshold,
        )
        orch = AgentOrchestrator(
            adapter=adapter,
            cascade_controller=ctrl,
            strong_model=cfg.strong_model,
        )
    return adapter, ctrl, orch


# ── UI Badges & Formatting ────────────────────────────────────────────────────

def _tier_badge(tier: str) -> str:
    t = tier.lower()
    if t == "simple":
        return "[bold black on #3fb950] SIMPLE [/bold black on #3fb950]"
    elif t == "medium":
        return "[bold black on #e3b341] MEDIUM [/bold black on #e3b341]"
    elif t == "complex":
        return "[bold white on #f0883e] COMPLEX [/bold white on #f0883e]"
    return f"[bold white on blue] {tier.upper()} [/bold white on blue]"


def _brand_header(subtitle: str | None = None) -> None:
    header = Text()
    header.append("⚡ FluxRouter", style="bold cyan")
    header.append(f" v{VERSION}", style="dim white")
    if subtitle:
        header.append(f"  •  {subtitle}", style="dim cyan")
    console.print(header)
    console.print()


# ── Main CLI entry point ──────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="Show FluxRouter version and exit.")
@click.pass_context
def main(ctx: click.Context, version: bool) -> None:
    """
    \b
    ⚡ FluxRouter — Complexity-Aware LLM Routing Engine

    Classifies queries by complexity (Simple / Medium / Complex) and dynamically
    routes them to fast vs strong model tiers, achieving cost reduction with zero quality compromise.

    Run 'fluxrouter help' for full documentation and example commands.
    """
    if version:
        console.print(f"FluxRouter v{VERSION}")
        sys.exit(0)

    if ctx.invoked_subcommand is None:
        _print_main_help()


# ── Help Command ──────────────────────────────────────────────────────────────

def _print_main_help() -> None:
    _brand_header("Complexity-Aware LLM Router")

    commands_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    commands_table.add_column("Command", style="bold yellow", width=18)
    commands_table.add_column("Description", style="white")
    commands_table.add_column("Usage Example", style="dim cyan")

    commands_table.add_row(
        "init",
        "Interactive setup wizard (API key, model defaults, thresholds)",
        "fluxrouter init"
    )
    commands_table.add_row(
        "route <query>",
        "Classify query and execute cascade routing pipeline",
        'fluxrouter route "What is 2+2?"'
    )
    commands_table.add_row(
        "agent <task>",
        "Decompose multi-part task, route subtasks & synthesize",
        'fluxrouter agent "Write & test a parser"'
    )
    commands_table.add_row(
        "serve",
        "Launch FastAPI backend & modern web dashboard UI",
        "fluxrouter serve --port 8000"
    )
    commands_table.add_row(
        "config",
        "Inspect or re-configure settings (--edit)",
        "fluxrouter config --edit"
    )
    commands_table.add_row(
        "help [cmd]",
        "Show detailed documentation for any command",
        "fluxrouter help route"
    )

    console.print(commands_table)
    console.print()

    quick_panel = Panel(
        "[bold white]Quick Start Workflow:[/bold white]\n"
        "  1. [cyan]fluxrouter init[/cyan]                    Initialize API key and configuration\n"
        "  2. [cyan]fluxrouter route \"What is 2+2?\"[/cyan]    Route single query with terminal UI\n"
        "  3. [cyan]fluxrouter serve[/cyan]                   Open interactive Web Dashboard",
        title="[bold green]Get Started[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(quick_panel)


@main.command("help")
@click.argument("command_name", required=False)
def help_cmd(command_name: str | None) -> None:
    """Show detailed help and documentation for commands."""
    if not command_name:
        _print_main_help()
        return

    cmd = command_name.lower().strip()
    _brand_header(f"Help: {cmd}")

    if cmd == "route":
        console.print(Panel(
            "[bold white]fluxrouter route QUERY [--json][/bold white]\n\n"
            "Classifies the query into [green]Simple[/green], [yellow]Medium[/yellow], or [orange3]Complex[/orange3] tier using k-NN cosine similarity.\n"
            "Evaluates margin threshold and LLM-judge quality escalation to pick the optimal model.\n\n"
            "[bold cyan]Options:[/bold cyan]\n"
            "  --json    Output raw structured JSON output (ideal for scripts & benchmarks)\n\n"
            "[bold cyan]Examples:[/bold cyan]\n"
            '  fluxrouter route "What is the capital of France?"\n'
            '  fluxrouter route "Design a distributed raft consensus algorithm" --json',
            title="[bold cyan]route Command[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
    elif cmd == "agent":
        console.print(Panel(
            "[bold white]fluxrouter agent TASK [--json][/bold white]\n\n"
            "Decomposes multi-part prompts into independent atomic subtasks, routes each subtask through\n"
            "the cascade controller, and synthesizes a final unified answer.\n\n"
            "[bold cyan]Options:[/bold cyan]\n"
            "  --json    Output full task orchestration JSON structure\n\n"
            "[bold cyan]Examples:[/bold cyan]\n"
            '  fluxrouter agent "Write a Python function to reverse a string, add docstring & unit test"',
            title="[bold cyan]agent Command[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
    elif cmd == "serve":
        console.print(Panel(
            "[bold white]fluxrouter serve [--port 8000] [--no-browser][/bold white]\n\n"
            "Launches the FastAPI proxy backend and serves the single-page Web Dashboard.\n"
            "Automatically detects port conflicts and finds an available port.\n\n"
            "[bold cyan]Options:[/bold cyan]\n"
            "  --port PORT       Specify HTTP port (default: 8000)\n"
            "  --no-browser      Disable automatic browser tab launch",
            title="[bold cyan]serve Command[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
    elif cmd in ("init", "config"):
        console.print(Panel(
            "[bold white]fluxrouter init / config[/bold white]\n\n"
            "Manage API key, model selection (fast vs strong), and classification margin thresholds.\n"
            "Configuration is saved to [cyan]~/.fluxrouter/config.yaml[/cyan].\n\n"
            "[bold cyan]Examples:[/bold cyan]\n"
            "  fluxrouter init\n"
            "  fluxrouter config --edit",
            title="[bold cyan]Configuration Commands[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
    else:
        console.print(f"[yellow]Unknown command '{command_name}'. Showing general help:[/yellow]\n")
        _print_main_help()


# ── init ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--force", is_flag=True, help="Overwrite existing config without prompting.")
def init(force: bool) -> None:
    """Interactive setup — save API key and routing preferences."""
    from fluxrouter.config import CONFIG_PATH, FluxConfig, save

    _brand_header("Interactive Setup Wizard")

    if CONFIG_PATH.exists() and not force:
        console.print(f"[yellow]Config already exists at {CONFIG_PATH}.[/yellow]")
        if not click.confirm("Do you want to overwrite it?", default=False):
            console.print("[dim]Aborted setup.[/dim]")
            return

    console.print(Panel.fit(
        "[bold white]Configuration Parameters[/bold white]\n"
        "[dim]Press Enter to accept default values shown in brackets.[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    api_key = click.prompt(
        "\n  Groq API key",
        hide_input=True,
        prompt_suffix=" > ",
    ).strip()
    if not api_key:
        err.print("[bold red]✗ API key cannot be empty.[/bold red]")
        sys.exit(1)

    # Validate key with smallest possible API request
    with console.status("[dim]Validating API key with Groq…[/dim]", spinner="dots"):
        try:
            from groq import Groq
            Groq(api_key=api_key).chat.completions.create(
                model=_FAST_DEFAULT,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            key_valid = True
        except Exception as exc:
            key_valid = False
            err_msg = str(exc)

    if not key_valid:
        console.print(f"  [bold red]✗ API key validation failed:[/bold red] {err_msg}")
        sys.exit(1)
    
    console.print("  [bold green]✓ Groq API key validated successfully![/bold green]")

    fast_model = click.prompt(
        "\n  Fast Model",
        default=_FAST_DEFAULT,
        prompt_suffix=" > ",
    ).strip()

    strong_model = click.prompt(
        "  Strong Model",
        default=_STRONG_DEFAULT,
        prompt_suffix=" > ",
    ).strip()

    margin = click.prompt(
        "  Margin Threshold (0.0–1.0)",
        default="0.08",
        prompt_suffix=" > ",
    ).strip()
    try:
        margin_f = float(margin)
        if not 0 < margin_f < 1:
            raise ValueError
    except ValueError:
        err.print("[bold red]✗ Margin must be a float between 0 and 1.[/bold red]")
        sys.exit(1)

    cfg = FluxConfig(
        api_key=api_key,
        fast_model=fast_model,
        strong_model=strong_model,
        margin_threshold=margin_f,
    )
    save(cfg)
    console.print()
    console.print(Panel(
        f"[bold green]✓ Configuration Saved![/bold green]\n"
        f"[dim]File:[/dim] {CONFIG_PATH}\n"
        f"[dim]Fast Model:[/dim] {fast_model}\n"
        f"[dim]Strong Model:[/dim] {strong_model}\n"
        f"[dim]Margin Threshold:[/dim] {margin_f}",
        border_style="green",
        box=box.ROUNDED,
    ))


# ── route ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("query")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON (for scripting).")
def route(query: str, as_json: bool) -> None:
    """Route QUERY through the cascade controller and print the result."""
    cfg = _load_cfg_or_exit()

    with console.status("[dim]Initializing Classifier & Routing Engine…[/dim]", spinner="dots"):
        adapter, ctrl, _ = _build_components(cfg)

    with console.status("[dim]Evaluating Query & Routing…[/dim]", spinner="dots"):
        decision = ctrl.route(query)

    resp = decision.final_response
    actual_cost = _cost(resp.model, resp.input_tokens, resp.output_tokens)
    strong_c    = _strong_cost(resp.input_tokens, resp.output_tokens)
    saved       = max(0.0, strong_c - actual_cost)
    pct_saved   = (saved / strong_c * 100) if strong_c > 0 else 0.0

    if as_json:
        out: dict[str, Any] = {
            "query":          decision.query,
            "predicted_tier": decision.predicted_tier.value,
            "margin":         round(decision.margin, 6),
            "escalated":      decision.escalated,
            "judge_score":    decision.judge_score,
            "models_called":  decision.models_called,
            "model_used":     resp.model,
            "input_tokens":   resp.input_tokens,
            "output_tokens":  resp.output_tokens,
            "total_latency_ms": round(decision.total_latency_ms, 2),
            "actual_cost_usd": round(actual_cost, 6),
            "saved_vs_strong_usd": round(saved, 6),
            "response": resp.text,
        }
        click.echo(json.dumps(out, indent=2))
        return

    # ── Revamped Terminal UI Output ───────────────────────────────────────────
    _brand_header("Route Execution")

    tier = decision.predicted_tier.value
    margin = decision.margin
    esc = decision.escalated
    latency = decision.total_latency_ms

    # Meta Table Card
    card = Table(box=box.ROUNDED, show_header=False, expand=True, border_style="cyan")
    card.add_column("Property", style="bold dim white", width=18)
    card.add_column("Value")

    card.add_row("Query Prompt", f"[white]{query}[/white]")
    card.add_row("Complexity Tier", f"{_tier_badge(tier)} [dim](Margin: {margin:.4f})[/dim]")
    card.add_row("Routed Model", f"[bold cyan]{resp.model}[/bold cyan]")
    
    if esc:
        esc_str = "[bold yellow]YES ▲ (Escalated to Strong Tier)[/bold yellow]"
        if decision.judge_score is not None:
            esc_str += f" [dim]• Judge Score: {decision.judge_score}/5[/dim]"
    else:
        esc_str = "[bold green]NO (Fast Tier Satisfied Prompt)[/bold green]"
        if decision.judge_score is not None:
            esc_str += f" [dim]• Judge Passed ({decision.judge_score}/5)[/dim]"
    
    card.add_row("Cascade Escalated", esc_str)
    card.add_row("Execution Time", f"{latency:.0f} ms")

    console.print(Panel(card, title="[bold cyan]Routing Audit Trail[/bold cyan]", border_style="cyan", box=box.ROUNDED))
    console.print()

    # Response Panel with Syntax-Highlighted Markdown
    markdown_content = Markdown(resp.text)
    border_color = "yellow" if esc else "green"
    console.print(Panel(
        markdown_content,
        title=f"[bold white] Response ({resp.model}) [/bold white]",
        border_style=border_color,
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    # Metrics Footer
    footer_text = Text()
    footer_text.append("Cost: ", style="dim white")
    footer_text.append(f"${actual_cost:.6f} ", style="bold white")
    footer_text.append(" • ", style="dim white")
    footer_text.append("Savings vs Strong: ", style="dim white")
    footer_text.append(f"+${saved:.6f} (+{pct_saved:.1f}%)", style="bold green")
    footer_text.append(" • ", style="dim white")
    footer_text.append(f"Tokens: {resp.input_tokens} in / {resp.output_tokens} out", style="dim white")
    
    console.print(footer_text)
    console.print()


# ── agent ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("task")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def agent(task: str, as_json: bool) -> None:
    """Decompose TASK into subtasks, route each, synthesize a final answer."""
    cfg = _load_cfg_or_exit()

    with console.status("[dim]Initializing Agent Orchestrator & Classifier…[/dim]", spinner="dots"):
        adapter, ctrl, orch = _build_components(cfg)

    with console.status("[dim]Orchestrating Multi-Task Pipeline…[/dim]", spinner="dots"):
        result = orch.execute(task)

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    # ── Revamped Terminal UI Output ───────────────────────────────────────────
    _brand_header("Agent Multi-Task Orchestrator")

    console.print(f"[bold white]Original Goal:[/bold white] [cyan]{task}[/cyan]\n")

    if len(result.subtasks) == 1 and result.subtasks[0] == task:
        console.print("  [bold green]↳ Task is atomic — skipped multi-part decomposition[/bold green]\n")
    else:
        console.print(f"  [bold cyan]↳ Task decomposed into {len(result.subtasks)} subtasks:[/bold cyan]\n")

    # Subtasks Table
    sub_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", expand=True)
    sub_table.add_column("#", style="bold yellow", width=4)
    sub_table.add_column("Subtask Query", style="white")
    sub_table.add_column("Tier", width=12)
    sub_table.add_column("Routed Model", style="cyan")
    sub_table.add_column("Judge", width=8)
    sub_table.add_column("Latency", justify="right", width=10)

    for i, d in enumerate(result.routing_decisions, 1):
        judge_str = f"{d.judge_score}/5 ★" if d.judge_score is not None else "—"
        esc_mark = " [yellow]▲[/yellow]" if d.escalated else ""
        sub_table.add_row(
            str(i),
            d.query[:70] + ("…" if len(d.query) > 70 else ""),
            _tier_badge(d.predicted_tier.value),
            d.final_response.model + esc_mark,
            judge_str,
            f"{d.total_latency_ms:.0f} ms"
        )

    console.print(sub_table)
    console.print()

    # Final Synthesized Answer Panel
    markdown_answer = Markdown(result.final_answer)
    console.print(Panel(
        markdown_answer,
        title="[bold white] Synthesized Final Answer [/bold white]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    # Aggregated Costs & Performance
    cb = result.cost_breakdown
    total_actual = sum(
        _cost(d.final_response.model, d.final_response.input_tokens, d.final_response.output_tokens)
        for d in result.routing_decisions
    )
    total_strong_hyp = sum(
        _strong_cost(d.final_response.input_tokens, d.final_response.output_tokens)
        for d in result.routing_decisions
    )
    saved = max(0.0, total_strong_hyp - total_actual)
    pct_saved = (saved / total_strong_hyp * 100) if total_strong_hyp > 0 else 0.0

    summary_text = Text()
    summary_text.append(f"Calls: {cb.fast_model_calls} fast, {cb.strong_model_calls} strong, {cb.judge_calls} judge ({cb.total_calls} total)", style="dim white")
    summary_text.append("  •  ", style="dim white")
    summary_text.append(f"Total Time: {result.total_latency_ms:.0f} ms", style="dim white")
    summary_text.append("  •  ", style="dim white")
    summary_text.append(f"Cost: ${total_actual:.6f}", style="dim white")
    summary_text.append("  •  ", style="dim white")
    summary_text.append(f"Saved: +${saved:.6f} (+{pct_saved:.1f}%)", style="bold green")

    console.print(summary_text)
    console.print()


# ── serve ─────────────────────────────────────────────────────────────────────

def _find_available_port(start_port: int, max_attempts: int = 20) -> int:
    """Find an available TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port


@main.command()
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
@click.option("--no-browser", is_flag=True, help="Do not auto-open the browser.")
def serve(port: int, no_browser: bool) -> None:
    """Launch the FluxRouter web dashboard."""
    _load_cfg_or_exit()

    actual_port = _find_available_port(port)
    if actual_port != port:
        console.print(f"[yellow]Port {port} is currently in use. Automatically switching to port {actual_port}.[/yellow]\n")

    url = f"http://localhost:{actual_port}"

    _brand_header("Dashboard Web Server")

    console.print(Panel(
        f"[bold white]FluxRouter Web Dashboard[/bold white]\n\n"
        f"  • Dashboard URL: [bold cyan]{url}[/bold cyan]\n"
        f"  • API Endpoint:  [bold cyan]http://localhost:{actual_port}/v1/chat/completions[/bold cyan]\n"
        f"  • Health Probe:  [bold cyan]http://localhost:{actual_port}/health[/bold cyan]\n\n"
        f"[dim]Press Ctrl+C to safely shutdown the server.[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    if not no_browser:
        def _open_browser():
            time.sleep(2.0)
            try:
                import webbrowser
                with redirect_stderr(io.StringIO()):
                    webbrowser.open(url)
            except Exception:
                pass
        Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        uvicorn.run("api.main:app", host="0.0.0.0", port=actual_port, reload=False, log_level="warning")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]FluxRouter Dashboard stopped gracefully.[/bold yellow]")


# ── config ────────────────────────────────────────────────────────────────────

@main.command("config")
@click.option("--edit", is_flag=True, help="Re-run interactive setup wizard.")
def config_cmd(edit: bool) -> None:
    """Show current configuration (use --edit to update)."""
    if edit:
        ctx = click.get_current_context()
        ctx.invoke(init, force=False)
        return

    from fluxrouter.config import CONFIG_PATH, load, ConfigNotFoundError
    try:
        cfg = load()
    except ConfigNotFoundError as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        sys.exit(1)

    _brand_header("Configuration Inspector")

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Setting", style="bold white", width=22)
    t.add_column("Configured Value", style="cyan")
    
    t.add_row("Groq API Key", cfg.masked_key())
    t.add_row("Fast Model", cfg.fast_model)
    t.add_row("Strong Model", cfg.strong_model)
    t.add_row("Margin Threshold", f"{cfg.margin_threshold}")
    t.add_row("Configuration Path", str(CONFIG_PATH))

    console.print(Panel(t, title="[bold cyan]Active Settings[/bold cyan]", border_style="cyan", box=box.ROUNDED))
    console.print("  [dim]Tip: Run 'fluxrouter config --edit' to change settings.[/dim]\n")
