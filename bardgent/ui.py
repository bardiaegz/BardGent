"""Tiny standalone UI helpers that don't fit naturally elsewhere."""

from bardgent.config import console


def print_welcome():
    console.print("[bold italic magenta]Welcome to BardGent ☻ [/bold italic magenta]!")
    console.print("Type 'exit' or 'quit' to leave.")
    console.print("[dim]Shift+Tab cycles mode (normal -> auto -> plan), or use /normal, /auto, /plan.[/dim]")
    console.print("[dim]Type /skills to see what's auto-detected, or /help style commands via Tab-completion.[/dim]")
    console.print()
