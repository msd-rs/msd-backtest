from mbt.run.report import Report
import orjson
import sys
import argparse
import polars as pl



def parse_args():
  parser = argparse.ArgumentParser(description="Review a backtest report")
  parser.add_argument("report", type=str, help="Path to the report file")

  sub_commands = parser.add_subparsers(dest="command", required=True)

  stat_parser = sub_commands.add_parser("stat", help="Show statistics")
  stat_parser.set_defaults(func=do_stat)

  sort_parser = sub_commands.add_parser("sort", help="Sort by a metric")
  sort_parser.add_argument("-m", "--metric", type=str, help="Metric to sort by")
  sort_parser.add_argument("-o", "--order", type=str, help="Order, desc or asc", default="desc")
  sort_parser.add_argument("-l", "--limit", type=int, help="Limit number of symbols")
  sort_parser.set_defaults(func=do_sort)

  detail_parser = sub_commands.add_parser("detail", help="Show detail for a symbol")
  detail_parser.add_argument("-s", "--symbol", type=str, help="Symbol to show")
  detail_parser.set_defaults(func=do_detail)
  
  return parser.parse_args()

def do_stat(args):
  reports = load_reports(args.report)
  df = pl.DataFrame([r.metrics(True) for r in reports.values()])
  if "symbol" in df.columns:
    df = df.drop("symbol")
  stat = df.describe()
  stat_col = stat.columns[0]
  stats = stat[stat_col].to_list()
  res = {}
  for col in stat.columns[1:]:
    vals = stat[col].to_list()
    res[col] = {s: v for s, v in zip(stats, vals) if s != "null_count" and v is not None}
  print(orjson.dumps(res).decode())

def do_detail(args):
  reports = load_reports(args.report)
  symbol_report = reports[args.symbol]
  d = {}
  d.update(symbol_report.detailed(False))
  d.update(symbol_report.metrics(True))
  print(orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY))

def do_sort(args):
  reports = load_reports(args.report)
  records = [{"symbol": r.symbol, "val": getattr(r, args.metric, 0)} for r in reports.values()]
  records.sort(key=lambda x: x["val"], reverse=args.order != "asc")
  if args.limit:
    records = records[:args.limit]
  res = {args.metric: {r["symbol"]: r["val"] for r in records}}
  print(orjson.dumps(res).decode())
  
def load_reports(report_path: str) -> dict[str, Report]:
  with open(report_path, "r") as f:
    data = orjson.loads(f.read())
  return {symbol: Report.from_dict(symbol, data[symbol]) for symbol in data}


def main():
  args = parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
  