#!/usr/bin/env python3
"""
RAG Demo with Ollama + Local LLM

This script demonstrates real RAG with a local LLM using Ollama.
Semantic caching still works to reduce redundant LLM calls.

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Pull a model: ollama pull gemma3:4b (or any other model)
3. Start Ollama: ollama serve (if not running)

Run: python3 demo_ollama.py
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from knowledge_base import EXPANDED_KNOWLEDGE_BASE, KNOWLEDGE_BASE_STATS
from rag_pipeline import RAGPipeline

console = Console()


# Demo queries for Ollama
DEMO_QUERIES = [
    ("What is your refund policy?", "Initial refund question"),
    ("Can I get my money back?", "Semantic variation → Cache hit"),
    ("How long does shipping take?", "New topic: shipping"),
    ("What are the delivery times?", "Semantic variation → Cache hit"),
    ("What does the warranty cover?", "New topic: warranty"),
    ("Is there a warranty on products?", "Semantic variation → Cache hit"),
]

# Default Ollama model - change to your preferred model
OLLAMA_MODEL = "gemma3:4b"


def print_header():
    """Print demo header."""
    header = Text()
    header.append("🦙 RAG Demo with Ollama + Gemma3:1B\n", style="bold cyan")
    header.append("━" * 45 + "\n", style="dim")
    header.append("Real LLM generation with semantic caching", style="white")
    
    console.print(Panel(header, box=box.DOUBLE, border_style="cyan", padding=(1, 2)))
    console.print()


def print_response(query: str, response, note: str):
    """Display query result."""
    if response.cache_hit:
        status = "[bold green]🟢 CACHE HIT[/bold green]"
        cost_style = "green"
        similarity = f" | Sim: {response.similarity_score:.1%}"
    else:
        status = "[bold yellow]🦙 OLLAMA GENERATION[/bold yellow]"
        cost_style = "cyan"
        similarity = ""
    
    console.print(f"[dim]{note}[/dim]")
    console.print(f'[italic]"{query}"[/italic]')
    console.print(f"{status} | ⏱️ {response.latency_ms:.0f}ms | [{cost_style}]💰 ${response.cost_usd:.4f}[/{cost_style}]{similarity}")
    
    if response.source_docs:
        docs = ", ".join(response.source_docs[:2])
        console.print(f"[dim]📄 Sources: {docs}[/dim]")
    
    # Show answer (truncated)
    answer_lines = response.answer.strip().split('\n')
    answer_preview = ' '.join(answer_lines)[:200]
    if len(response.answer) > 200:
        answer_preview += "..."
    console.print(f"\n[white]💬 {answer_preview}[/white]")
    console.print()


def print_summary(pipeline: RAGPipeline, elapsed: float):
    """Print results summary."""
    stats = pipeline.get_stats()
    
    table = Table(title="📊 Results Summary", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right", style="bold")
    
    table.add_row("Total Queries", str(stats.total_queries))
    table.add_row("Cache Hits", f"[green]{stats.cache_hits}[/green]")
    table.add_row("LLM Generations", f"[yellow]{stats.cache_misses}[/yellow]")
    table.add_row("Hit Rate", f"[bold cyan]{stats.hit_rate:.1%}[/bold cyan]")
    table.add_row("─" * 15, "─" * 10)
    table.add_row("LLM Calls Saved", f"[bold green]{stats.cache_hits}[/bold green]")
    table.add_row("Total Time", f"{elapsed:.1f}s")
    
    console.print(table)


def run_demo():
    """Run the Ollama RAG demo."""
    print_header()
    
    # Check Ollama
    console.print("[bold]Checking Ollama connection...[/bold]")
    
    try:
        import ollama
        models = ollama.list()
        model_names = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
        
        # Check if the specified model is available
        has_model = any(OLLAMA_MODEL.split(':')[0] in name.lower() for name in model_names)
        
        if not has_model:
            console.print(f"[yellow]⚠️  {OLLAMA_MODEL} not found. Pulling model...[/yellow]")
            console.print("[dim]This may take a few minutes on first run.[/dim]\n")
            ollama.pull(OLLAMA_MODEL)
        
        console.print(f"[green]✓[/green] Ollama ready with {OLLAMA_MODEL}\n")
        
    except ImportError:
        console.print("[red]✗ ollama package not installed[/red]")
        console.print("Run: pip install ollama")
        return
    except Exception as e:
        console.print(f"[red]✗ Ollama error: {e}[/red]")
        console.print("Make sure Ollama is running: ollama serve")
        return
    
    # Initialize pipeline with Ollama
    console.print("[bold blue]Initializing RAG Pipeline with Ollama...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading knowledge base...[/bold blue]"),
        console=console,
    ) as progress:
        task = progress.add_task("init", total=None)
        pipeline = RAGPipeline(mock_mode=False, use_ollama=True, ollama_model=OLLAMA_MODEL)
        
        # Add expanded knowledge base
        for doc_id, doc_data in list(EXPANDED_KNOWLEDGE_BASE.items())[:20]:  # Load 20 docs for demo
            embedding = pipeline.embedding_service.embed(doc_data["content"])
            pipeline.knowledge_base.add_document(
                doc_id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"],
                embedding=embedding,
            )
    
    doc_count = pipeline.knowledge_base.index.ntotal
    console.print(f"[green]✓[/green] Pipeline ready with {doc_count} documents\n")
    
    # Run queries
    console.print(f"[bold]Running {len(DEMO_QUERIES)} queries...[/bold]\n")
    console.print("─" * 60 + "\n")
    
    start_time = time.time()
    
    for query, note in DEMO_QUERIES:
        with Progress(
            SpinnerColumn(),
            TextColumn("[yellow]Processing...[/yellow]" if "Cache" not in note else "[green]Checking cache...[/green]"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("query", total=None)
            response = pipeline.answer_query(query)
        
        print_response(query, response, note)
        console.print("─" * 60 + "\n")
    
    elapsed = time.time() - start_time
    
    # Summary
    print_summary(pipeline, elapsed)
    
    # Key takeaways
    console.print(Panel(
        Text.assemble(
            ("🎯 Key Points\n\n", "bold white"),
            ("• ", "cyan"), ("Local LLM", "bold yellow"), (" - No API costs, full privacy\n", "white"),
            ("• ", "cyan"), ("Semantic Cache", "bold green"), (" - Similar queries skip LLM entirely\n", "white"),
            ("• ", "cyan"), ("Real RAG", "bold cyan"), (" - Answers based on actual documents", "white"),
        ),
        border_style="cyan",
        box=box.ROUNDED,
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
