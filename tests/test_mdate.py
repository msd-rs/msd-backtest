import matplotlib.dates as mdates
import datetime


def test_mdate_build():
  x = 20089
  dt = mdates.num2date(x, tz=datetime.timezone.utc)
  print(dt)
  assert dt.year == 2025
  assert dt.month == 1
  assert dt.day == 1


