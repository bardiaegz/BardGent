"""Tiny standalone UI helpers that don't fit naturally elsewhere."""

from bardgent.config import console


def print_welcome():
    console.print("[bold italic magenta]Welcome to BardGent ☻ [/bold italic magenta]!")
    console.print("Type 'exit' or 'quit' to leave. Type /help for slash commands.")
    console.print("[dim]Shift+Tab cycles mode (normal -> auto -> plan), or use /normal, /auto, /plan.[/dim]")
    console.print("[dim]Type /skills to see installed skills; Tab-complete any slash command.[/dim]")
    console.print()
