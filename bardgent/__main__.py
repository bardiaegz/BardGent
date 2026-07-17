"""Allow `python -m bardgent` (and the scheduler daemon spawn) to work."""
from bardgent.main import main

if __name__ == '__main__':
    main()
