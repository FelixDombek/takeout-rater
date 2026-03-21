"""Allow ``python -m takeout_rater`` to invoke the CLI."""

import sys

from takeout_rater.cli import main

sys.exit(main())
