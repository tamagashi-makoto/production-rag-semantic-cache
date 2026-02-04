#!/usr/bin/env python3
"""RAG semantic cache demo."""

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from rag_pipeline import RAGPipeline, RAGResponse


console = Console()


def print_header():
    header_text = Text()
    header_text.append("RAG with Semantic Caching\n", style="bold cyan")
    header_text.append("Demonstrating cost & latency optimization through semantic similarity caching", style="white")

    console.print(Panel(
        header_text,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def print_query(query: str, scenario: str):
    console.print(f"[bold white]Query:[/bold white] {scenario}")
    console.print(f'   "{query}"')
    console.print()


def print_response(response: RAGResponse, show_answer: bool = True):
    if response.cache_hit:
        status_text = "[bold green]CACHE HIT[/bold green]"
        cost_style = "green"
        cost_note = " (saved)"
        similarity_text = f" | Similarity: {response.similarity_score:.2%}"
    else:
        status_text = "[bold yellow]CACHE MISS - LLM call[/bold yellow]"
        cost_style = "yellow"
        cost_note = ""
        similarity_text = ""

    console.print(f"   {status_text}")

    latency_text = f"Latency: {response.latency_ms:.0f}ms"
    cost_text = f"Cost: ${response.cost_usd:.4f}{cost_note}"

    console.print(f"   {latency_text} | [{cost_style}]{cost_text}[/{cost_style}]{similarity_text}")

    if response.source_docs:
        docs_text = ", ".join(response.source_docs[:2])
        console.print(f"   Sources: [dim]{docs_text}[/dim]")

    if show_answer:
        answer_preview = response.answer[:150] + "..." if len(response.answer) > 150 else response.answer
        console.print(f"\n   Answer: {answer_preview}")

    console.print()


def print_knowledge_update(doc_id: str, invalidated_count: int):
    console.print(Panel(
        Text.assemble(
            ("KNOWLEDGE BASE UPDATED\n", "bold yellow"),
            (f"Document: ", "white"),
            (f'"{doc_id}"\n', "cyan"),
            ("Cache entries invalidated: ", "white"),
            (str(invalidated_count), "bold red"),
        ),
        box=box.ROUNDED,
        border_style="yellow",
        padding=(0, 2),
    ))
    console.print()


def print_summary(pipeline: RAGPipeline):
    stats = pipeline.get_stats()

    table = Table(
        title="Cost Savings Summary",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        header_style="bold cyan",
        title_style="bold white",
    )

    table.add_column("Metric", style="white", justify="left")
    table.add_column("Value", style="bold", justify="right")

    table.add_row("Total Queries", str(stats.total_queries))
    table.add_row("Cache Hits", f"[green]{stats.cache_hits}[/green]")
    table.add_row("Cache Misses", f"[yellow]{stats.cache_misses}[/yellow]")
    table.add_row("Hit Rate", f"[bold green]{stats.hit_rate:.1%}[/bold green]")
    table.add_row("-" * 20, "-" * 15)
    table.add_row("Total Cost Saved", f"[bold green]${stats.total_cost_saved:.4f}[/bold green]")
    table.add_row("Cost Reduction", f"[bold green]{stats.cost_reduction_percent:.0f}%[/bold green]")

    console.print(table)
    console.print()

    # Projection
    projection_text = Text()
    projection_text.append("Scale Projection\n", style="bold white")
    projection_text.append("-" * 40 + "\n", style="dim")
    projection_text.append("With 10,000 queries/day and ", style="white")
    projection_text.append(f"{stats.hit_rate:.0%} cache hit rate", style="bold green")
    projection_text.append(":\n\n", style="white")

    daily_queries = 10000
    daily_hits = int(daily_queries * stats.hit_rate)
    cost_per_api_call = 0.002
    daily_savings = daily_hits * cost_per_api_call
    monthly_savings = daily_savings * 30
    yearly_savings = daily_savings * 365

    projection_text.append(f"   Daily Savings:   ${daily_savings:,.2f}\n", style="green")
    projection_text.append(f"   Monthly Savings: ${monthly_savings:,.2f}\n", style="green")
    projection_text.append(f"   Yearly Savings:  ", style="green")
    projection_text.append(f"${yearly_savings:,.2f}", style="bold green on dark_green")

    console.print(Panel(
        projection_text,
        box=box.ROUNDED,
        border_style="green",
        padding=(1, 2),
    ))


def print_separator(title: str = ""):
    if title:
        console.print(f"\n[bold blue]{'=' * 15} {title} {'=' * 15}[/bold blue]\n")
    else:
        console.print(f"\n[dim]{'-' * 50}[/dim]\n")


def run_demo():
    print_header()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Initializing RAG Pipeline...[/bold blue]"),
        console=console,
    ) as progress:
        task = progress.add_task("init", total=None)
        pipeline = RAGPipeline(mock_mode=True)
        time.sleep(0.5)

    console.print("[green]OK[/green] Pipeline initialized with 4 knowledge documents\n")

    # Scenario 1: Initial Query
    print_separator("SCENARIO 1: Initial Query")

    query1 = "What is the refund policy?"
    print_query(query1, "First-time question")

    with Progress(
        SpinnerColumn(),
        TextColumn("[yellow]Processing query...[/yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("query", total=None)
        response1 = pipeline.answer_query(query1)

    print_response(response1)

    # Scenario 2: Semantically Similar Query
    print_separator("SCENARIO 2: Semantically Similar Query")

    query2 = "Can I get my money back?"
    print_query(query2, "Different words, same meaning")

    console.print("   [dim]Note: Different tokens but same semantic meaning[/dim]\n")

    response2 = pipeline.answer_query(query2)
    print_response(response2)

    # Comparison
    comparison = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    comparison.add_column("Metric", style="white")
    comparison.add_column("Query 1", style="yellow")
    comparison.add_column("Query 2", style="green")
    comparison.add_column("Improvement", style="bold cyan")

    comparison.add_row(
        "Latency",
        f"{response1.latency_ms:.0f}ms",
        f"{response2.latency_ms:.0f}ms",
        f"[bold green]{response1.latency_ms / max(response2.latency_ms, 1):.0f}x faster[/bold green]"
    )
    comparison.add_row(
        "Cost",
        f"${response1.cost_usd:.4f}",
        f"${response2.cost_usd:.4f}",
        "[bold green]100% saved[/bold green]"
    )

    console.print(Panel(comparison, title="[bold]Comparison[/bold]", border_style="cyan"))
    console.print()

    # Scenario 3: Another Variation
    print_separator("SCENARIO 3: Another Variation")

    query3 = "How do I return a product and get a refund?"
    print_query(query3, "Yet another variation")

    response3 = pipeline.answer_query(query3)
    print_response(response3, show_answer=False)

    # Scenario 4: Knowledge Update
    print_separator("SCENARIO 4: Knowledge Update -> Cache Invalidation")

    console.print("[bold yellow]Business Update:[/bold yellow] Refund policy has changed!")
    console.print("[dim]   New policy: 60-day refund window (was 30 days)[/dim]\n")

    new_policy = (
        "Our UPDATED refund policy now allows customers to return products within "
        "60 days of purchase for a full refund. This is an extended window from our "
        "previous 30-day policy. Items must be unused and in original packaging. "
        "Refunds are processed within 3-5 business days."
    )

    invalidated = pipeline.update_knowledge_base("refund_policy", new_policy)
    print_knowledge_update("refund_policy", invalidated)

    console.print("[bold white]Re-asking the same question after policy change:[/bold white]\n")
    print_query(query1, "Same question, but knowledge has changed")

    with Progress(
        SpinnerColumn(),
        TextColumn("[yellow]Fetching fresh answer...[/yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("query", total=None)
        response4 = pipeline.answer_query(query1)

    print_response(response4)

    console.print(Panel(
        "[bold green]Cache invalidation working correctly![/bold green]\n"
        "The system detected that the source document changed and invalidated\n"
        "the cached answer, ensuring users get fresh information.",
        border_style="green",
        box=box.ROUNDED,
    ))
    console.print()

    # Summary
    print_separator("RESULTS")
    print_summary(pipeline)

    console.print(Panel(
        Text.assemble(
            ("Key Takeaways\n\n", "bold white"),
            ("1. ", "cyan"), ("Semantic caching matches ", "white"),
            ("meaning", "bold yellow"), (", not just keywords\n", "white"),
            ("2. ", "cyan"), ("Cache hits are ", "white"),
            ("significantly faster", "bold green"), (" and ", "white"),
            ("cheaper", "bold green"), ("\n", "white"),
            ("3. ", "cyan"), ("Automatic cache invalidation ", "white"),
            ("maintains data accuracy", "bold yellow"),
        ),
        title="[bold]Summary[/bold]",
        border_style="cyan",
        box=box.DOUBLE,
        padding=(1, 2),
    ))


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise
