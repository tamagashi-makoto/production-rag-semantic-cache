#!/usr/bin/env python3
"""Large-scale RAG demo with semantic caching."""

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from knowledge_base import EXPANDED_KNOWLEDGE_BASE, KNOWLEDGE_BASE_STATS
from rag_pipeline import RAGPipeline

console = Console()


DEMO_QUERIES = [
    {
        "category": "Returns & Refunds",
        "queries": [
            ("What is your refund policy?", "Initial refund query"),
            ("How can I get my money back?", "Semantic variation #1"),
            ("I want to return something, how?", "Semantic variation #2"),
            ("What's the return window for electronics?", "Specific: electronics"),
            ("Can I return clothes after wearing them?", "Specific: clothing"),
        ]
    },
    {
        "category": "Shipping & Delivery",
        "queries": [
            ("How long does shipping take?", "Initial shipping query"),
            ("What are the delivery times?", "Semantic variation #1"),
            ("How fast can I get my order?", "Semantic variation #2"),
            ("Do you offer overnight delivery?", "Specific: overnight"),
            ("How do I track my package?", "Specific: tracking"),
        ]
    },
    {
        "category": "Warranty & Protection",
        "queries": [
            ("What does the warranty cover?", "Initial warranty query"),
            ("Is there a guarantee on products?", "Semantic variation #1"),
            ("How do I file a warranty claim?", "Specific: claims"),
            ("Can I get extended protection?", "Specific: extended warranty"),
        ]
    },
    {
        "category": "Customer Support",
        "queries": [
            ("How do I contact customer service?", "Initial support query"),
            ("I need help with my order", "Semantic variation #1"),
            ("What's your support phone number?", "Specific: phone"),
            ("Do you have live chat support?", "Specific: chat"),
        ]
    },
    {
        "category": "Product Information",
        "queries": [
            ("What size TV should I buy?", "Initial product query"),
            ("Which laptop is best for work?", "Specific: laptop"),
            ("What should I look for in a refrigerator?", "Specific: appliance"),
        ]
    },
]


def print_header():
    header_text = Text()
    header_text.append("Large-Scale RAG Demo\n", style="bold cyan")
    header_text.append("-" * 55 + "\n", style="dim")
    header_text.append("Demonstrating real RAG with ", style="white")
    header_text.append(f"{KNOWLEDGE_BASE_STATS['total_documents']} documents", style="bold yellow")
    header_text.append(" across 6 categories", style="white")

    console.print(Panel(
        header_text,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2),
    ))


def print_knowledge_base_summary():
    table = Table(
        title="Knowledge Base Contents",
        box=box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Category", style="cyan")
    table.add_column("Documents", justify="right", style="green")

    for category, count in KNOWLEDGE_BASE_STATS["categories"].items():
        table.add_row(category.title(), str(count))

    table.add_row("-" * 15, "-" * 5)
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{KNOWLEDGE_BASE_STATS['total_documents']}[/bold]"
    )

    console.print(table)
    console.print(f"\n[dim]Total content: {KNOWLEDGE_BASE_STATS['total_content_chars']:,} characters[/dim]\n")


def print_query_result(query: str, response, note: str):
    if response.cache_hit:
        status = "[bold green]CACHE HIT[/bold green]"
        cost_style = "green"
        similarity = f" | Similarity: {response.similarity_score:.1%}"
    else:
        status = "[bold red]CACHE MISS[/bold red]"
        cost_style = "yellow"
        similarity = ""

    console.print(f"   [dim]{note}[/dim]")
    console.print(f'   "{query}"')
    console.print(f"   {status} | Latency: {response.latency_ms:.0f}ms | [{cost_style}]Cost: ${response.cost_usd:.4f}[/{cost_style}]{similarity}")

    if response.source_docs and not response.cache_hit:
        docs = ", ".join(response.source_docs[:2])
        console.print(f"   [dim]Retrieved: {docs}[/dim]")

    if not response.cache_hit:
        answer_preview = response.answer[:120].replace("\n", " ") + "..."
        console.print(f"   [dim]{answer_preview}[/dim]")

    console.print()


def run_category_demo(pipeline: RAGPipeline, category_data: dict):
    console.print(f"\n[bold blue]{'=' * 10} {category_data['category']} {'=' * 10}[/bold blue]\n")

    for query, note in category_data["queries"]:
        response = pipeline.answer_query(query)
        print_query_result(query, response, note)
        time.sleep(0.1)


def print_final_summary(pipeline: RAGPipeline, total_queries: int, start_time: float):
    stats = pipeline.get_stats()
    elapsed = time.time() - start_time

    table = Table(
        title="Demo Results Summary",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
    )
    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Total Queries", str(total_queries))
    table.add_row("Cache Hits", f"[green]{stats.cache_hits}[/green]")
    table.add_row("Cache Misses", f"[yellow]{stats.cache_misses}[/yellow]")
    table.add_row("Hit Rate", f"[bold cyan]{stats.hit_rate:.1%}[/bold cyan]")
    table.add_row("-" * 20, "-" * 15)
    table.add_row("Cost Without Cache", f"${total_queries * 0.002:.4f}")
    table.add_row("Actual Cost", f"${stats.cache_misses * 0.002:.4f}")
    table.add_row("Total Saved", f"[bold green]${stats.total_cost_saved:.4f}[/bold green]")
    table.add_row("Cost Reduction", f"[bold green]{stats.cost_reduction_percent:.0f}%[/bold green]")
    table.add_row("-" * 20, "-" * 15)
    table.add_row("Demo Duration", f"{elapsed:.1f}s")
    table.add_row("Knowledge Docs", str(KNOWLEDGE_BASE_STATS['total_documents']))

    console.print(table)

    daily_queries = 10000
    monthly_savings = daily_queries * 30 * (stats.hit_rate * 0.002)
    yearly_savings = monthly_savings * 12

    projection = Text()
    projection.append("Scale Projection\n\n", style="bold white")
    projection.append(f"With {daily_queries:,} queries/day and {stats.hit_rate:.0%} cache hit rate:\n\n", style="white")
    projection.append(f"   Monthly Savings:  ", style="white")
    projection.append(f"${monthly_savings:,.0f}\n", style="bold green")
    projection.append(f"   Yearly Savings:   ", style="white")
    projection.append(f"${yearly_savings:,.0f}", style="bold green on dark_green")

    console.print(Panel(projection, border_style="green", padding=(1, 2)))


def run_demo():
    print_header()

    print_knowledge_base_summary()

    console.print("[bold blue]Initializing RAG Pipeline...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading {task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("knowledge base documents", total=100)

        pipeline = RAGPipeline(mock_mode=True)
        progress.update(task, advance=30)

        for i, (doc_id, doc_data) in enumerate(EXPANDED_KNOWLEDGE_BASE.items()):
            embedding = pipeline.embedding_service.embed(doc_data["content"])
            pipeline.knowledge_base.add_document(
                doc_id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"],
                embedding=embedding,
            )
            progress.update(task, advance=70 / len(EXPANDED_KNOWLEDGE_BASE))

    total_docs = pipeline.knowledge_base.index.ntotal
    console.print(f"[green]OK[/green] Pipeline initialized with [bold]{total_docs}[/bold] documents\n")

    total_queries = sum(len(cat["queries"]) for cat in DEMO_QUERIES)
    console.print(f"[bold]Running {total_queries} queries across {len(DEMO_QUERIES)} categories...[/bold]\n")

    start_time = time.time()

    for category_data in DEMO_QUERIES:
        run_category_demo(pipeline, category_data)

    console.print(f"\n[bold yellow]{'=' * 10} Cache Invalidation Demo {'=' * 10}[/bold yellow]\n")

    console.print("[yellow]Updating 'refund_policy_general' document...[/yellow]")
    console.print("[dim]   New policy: 60-day return window (was 30 days)[/dim]\n")

    new_content = """Our NEW refund policy extends the return window to 60 days for all customers.
This is double our previous 30-day policy. Full refund guaranteed with receipt. Items must be
unused and in original packaging. Refunds processed within 3-5 business days."""

    invalidated = pipeline.update_knowledge_base("refund_policy_general", new_content)
    console.print(f"[red]   {invalidated} cache entries invalidated[/red]\n")

    console.print("[bold]Re-querying after knowledge update:[/bold]\n")
    response = pipeline.answer_query("What is your refund policy?")
    print_query_result("What is your refund policy?", response, "After document update")

    console.print()
    print_final_summary(pipeline, total_queries + 1, start_time)

    console.print(Panel(
        Text.assemble(
            ("Key Takeaways\n\n", "bold white"),
            ("1. ", "cyan"), ("Real RAG retrieval from ", "white"),
            (f"{total_docs} documents\n", "bold yellow"),
            ("2. ", "cyan"), ("Semantic similarity enables ", "white"),
            ("cross-phrasing cache hits\n", "bold green"),
            ("3. ", "cyan"), ("Automatic cache invalidation ", "white"),
            ("on document updates\n", "bold yellow"),
            ("4. ", "cyan"), ("Production-ready architecture ", "white"),
            ("for enterprise scale", "bold cyan"),
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
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
