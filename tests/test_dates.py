import datetime as dt

from usbot.utils.dates import is_last_trading_day_of_month, is_trading_day, market_status


def test_weekend_is_not_trading_day():
    # 2024-01-06 is a Saturday
    assert not is_trading_day(dt.date(2024, 1, 6))
    assert market_status(dt.date(2024, 1, 6)) == "weekend"


def test_known_weekday_is_trading_day():
    # 2024-01-03 is a Wednesday, normal session
    assert is_trading_day(dt.date(2024, 1, 3))
    assert market_status(dt.date(2024, 1, 3)) == "open"


def test_new_year_is_holiday():
    # 2024-01-01 is a Monday holiday
    assert not is_trading_day(dt.date(2024, 1, 1))


def test_last_trading_day_of_month():
    # 2024-01-31 is a Wednesday and the last trading day of January 2024
    assert is_last_trading_day_of_month(dt.date(2024, 1, 31))
    assert not is_last_trading_day_of_month(dt.date(2024, 1, 30))
