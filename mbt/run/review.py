# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from mbt.run.report import Report, build_markdown_summary, compute_stock_attribution
import orjson
import sys
import argparse
import polars as pl



def parse_args():
  parser = argparse.ArgumentParser(description="Review a backtest report")
  parser.add_argument("report", type=str, help="Path to the report file")

  sub_commands = parser.add_subparsers(dest="command", required=True)

  summary_parser = sub_commands.add_parser("summary", help="Show markdown summary report")
  summary_parser.add_argument("-l", "--limit", type=int, default=10, help="Limit number of top/bottom objs")
  summary_parser.set_defaults(func=do_summary)

  stat_parser = sub_commands.add_parser("stat", help="Show statistics")
  stat_parser.set_defaults(func=do_stat)

  sort_parser = sub_commands.add_parser("sort", help="Sort by a metric")
  sort_parser.add_argument("-m", "--metric", type=str, help="Metric to sort by")
  sort_parser.add_argument("-o", "--order", type=str, help="Order, desc or asc", default="desc")
  sort_parser.add_argument("-l", "--limit", type=int, help="Limit number of objs")
  sort_parser.set_defaults(func=do_sort)

  detail_parser = sub_commands.add_parser("detail", help="Show detail for a obj")
  detail_parser.add_argument("-s", "--obj", type=str, help="obj to show")
  detail_parser.set_defaults(func=do_detail)

  trades_parser = sub_commands.add_parser("trades", help="Show trade cycles / holding periods")
  trades_parser.add_argument("-s", "--obj", type=str, default=None, help="Filter by obj")
  trades_parser.add_argument("-l", "--limit", type=int, default=None, help="Limit number of rows")
  trades_parser.set_defaults(func=do_trades)

  attrib_parser = sub_commands.add_parser("attribution", help="Show stock attribution / contribution ranking")
  attrib_parser.add_argument("-l", "--limit", type=int, default=None, help="Limit number of objs")
  attrib_parser.add_argument("-o", "--order", type=str, default="desc", choices=["desc", "asc"], help="Sort order by profit (desc/asc)")
  attrib_parser.set_defaults(func=do_attribution)
  
  return parser.parse_args()

def do_summary(args):
  reports = load_reports(args.report)
  print(build_markdown_summary(list(reports.values()), top_k=args.limit), end="")


def do_stat(args):
  reports = load_reports(args.report)
  df = pl.DataFrame([r.metrics(True) for r in reports.values()])
  if "obj" in df.columns:
    df = df.drop("obj")
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
  obj_report = reports[args.obj]
  d = {}
  d.update(obj_report.detailed(False))
  d.update(obj_report.metrics(True))
  print(orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY))

def do_sort(args):
  reports = load_reports(args.report)
  records = [{"obj": r.obj, "val": getattr(r, args.metric, 0)} for r in reports.values()]
  records.sort(key=lambda x: x["val"], reverse=args.order != "asc")
  if args.limit:
    records = records[:args.limit]
  res = {args.metric: {r["obj"]: r["val"] for r in records}}
  print(orjson.dumps(res).decode())
  
def do_trades(args):
  reports = load_reports(args.report)
  all_trades = []
  for obj, r in reports.items():
    trades = getattr(r, "trades", None)
    if trades:
      all_trades.extend(trades)
  if args.obj:
    all_trades = [t for t in all_trades if t.get("obj") == args.obj]
  if args.limit:
    all_trades = all_trades[:args.limit]
  print(orjson.dumps(all_trades, option=orjson.OPT_INDENT_2).decode())


def do_attribution(args):
  reports = load_reports(args.report)
  all_attrib = []
  for obj, r in reports.items():
    attrib = getattr(r, "attribution", None)
    if attrib:
      all_attrib.extend(attrib)
  if not all_attrib:
    for obj, r in reports.items():
      trades = getattr(r, "trades", None)
      if trades:
        all_attrib.extend(compute_stock_attribution(trades))
  all_attrib.sort(key=lambda x: x.get("total_profit", 0.0), reverse=args.order != "asc")
  if args.limit:
    all_attrib = all_attrib[:args.limit]
  print(orjson.dumps(all_attrib, option=orjson.OPT_INDENT_2).decode())


def load_reports(report_path: str) -> dict[str, Report]:
  with open(report_path, "r", encoding='utf-8') as f:
    data = orjson.loads(f.read())
  return {obj: Report.from_dict(obj, data[obj]) for obj in data}


def main():
  args = parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
  