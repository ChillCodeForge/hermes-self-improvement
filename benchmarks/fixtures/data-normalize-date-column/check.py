import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = """order_id,customer,order_date,total
1001,ACME GmbH,2024-03-05,199.00
1002,Globex,2024-03-05,45.50
1003,Initech,2024-03-07,12.00
1004,Umbrella,2024-03-09,300.10
1005,Hooli,2023-12-31,80.00
"""


def body(checks):
    checks.check("clean_file_exact", read("orders_clean.csv") == EXPECTED)
    checks.check("source_unchanged", '"March 7, 2024"' in read("orders.csv"))


run(body)
