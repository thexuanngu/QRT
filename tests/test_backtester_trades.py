import pandas as pd
import pytest

from backtest.backtesting import Backtester
from backtest.data_models import Trade
from backtest.strategy import Strategy


class StaticStrategy(Strategy):
    def predict(self, tradable_assets, history, trading_state):
        return pd.Series(0.0, index=tradable_assets)


@pytest.fixture(name="sample_prices")
def fixture_sample_prices():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0, 202.0],
        },
        index=index,
    )


@pytest.fixture(name="backtester")
def fixture_backtester(sample_prices):
    return Backtester(
        tradable_assets=["AAPL", "MSFT"],
        strategy_fn=StaticStrategy(),
        strategy_name="static",
        data=sample_prices.copy(),
        prices=sample_prices.copy(),
        execute_on_next_tick=False,
        allow_short=True,
    )


def test_record_trade_stores_positions_and_costs(backtester):
    trade = Trade(
        date=backtester.dates[0],
        position=pd.Series({"AAPL": 125.0, "MSFT": -40.0}),
        price=pd.Series({"AAPL": 100.0, "MSFT": 200.0}),
        commission=2.5,
        slippage=1.25,
    )

    getattr(backtester, "_record_trade")(trade)

    assert len(backtester.trades) == 1
    recorded_trade = backtester.trades[0]
    assert recorded_trade.date == trade.date
    assert recorded_trade.position["AAPL"] == pytest.approx(125.0)
    assert recorded_trade.position["MSFT"] == pytest.approx(-40.0)
    assert recorded_trade.commission == pytest.approx(2.5)
    assert recorded_trade.slippage == pytest.approx(1.25)


def test_record_trade_rejects_unknown_assets(backtester):
    trade = Trade(
        date=backtester.dates[0],
        position=pd.Series({"NVDA": 50.0}),
        price=pd.Series({"NVDA": 800.0}),
        commission=1.0,
        slippage=0.5,
    )

    with pytest.raises(ValueError, match="unknown assets"):
        getattr(backtester, "_record_trade")(trade)


def test_execute_target_portfolio_adds_small_rebalance_note(backtester):
    date = backtester.dates[0]
    exec_price = backtester.prices.loc[date]
    target_asset_amt = pd.Series({"AAPL": 0.0, "MSFT": 0.0})

    getattr(backtester, "_execute_target_portfolio")(target_asset_amt, exec_price, date)

    assert backtester.trade_notes[date] == ["Position rebalance too small - trade not executed"]


def test_execute_target_portfolio_adds_no_short_note(sample_prices):
    backtester = Backtester(
        tradable_assets=["AAPL", "MSFT"],
        strategy_fn=StaticStrategy(),
        strategy_name="no-short",
        data=sample_prices.copy(),
        prices=sample_prices.copy(),
        execute_on_next_tick=False,
        allow_short=False,
    )

    date = backtester.dates[0]
    exec_price = backtester.prices.loc[date]
    target_asset_amt = pd.Series({"AAPL": -100.0, "MSFT": 250.0})

    getattr(backtester, "_execute_target_portfolio")(target_asset_amt, exec_price, date)

    assert "Asset Weights clipped to 0 because of no shorting allowed" in backtester.trade_notes[date]


def test_execute_target_portfolio_adds_position_limit_note(backtester):
    date = backtester.dates[0]
    exec_price = backtester.prices.loc[date]
    target_asset_amt = pd.Series({"AAPL": 2_500_000.0, "MSFT": 0.0})

    getattr(backtester, "_execute_target_portfolio")(target_asset_amt, exec_price, date)

    assert f"Trade rejected by position limits at {date}." in backtester.trade_notes[date]
    assert backtester.current_position.eq(0.0).all()